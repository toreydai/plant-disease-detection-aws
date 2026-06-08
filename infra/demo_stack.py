"""单 Stack：DynamoDB 历史表 + EC2（自动跑 Streamlit）+ 最小 IAM。

一键起停：
    cdk deploy     # 起表 + EC2，输出访问 URL
    cdk destroy    # 拆表 + EC2（Endpoint 单独 delete-endpoint）
"""
from aws_cdk import (
    Stack,
    RemovalPolicy,
    CfnOutput,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_iam as iam,
)
from constructs import Construct

ENDPOINT_NAME = "potato-disease-demo"
TABLE_NAME = "potato-disease-demo-records"
KIMI_MODEL = "moonshotai.kimi-k2.5"
# 部署前替换为你的代码仓库地址
REPO_URL = "https://github.com/<your-org>/plant-disease-detection-aws.git"


class DemoStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # ---- DynamoDB（demo 用 DESTROY，cdk destroy 时一并删除）----
        table = dynamodb.Table(
            self, "Records",
            table_name=TABLE_NAME,
            partition_key=dynamodb.Attribute(
                name="id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ---- EC2 角色：调用 Endpoint / Bedrock + 读写表 + SSM 登录 ----
        role = iam.Role(
            self, "AppRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMManagedInstanceCore")],
        )
        role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint"],
            resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:"
                       f"endpoint/{ENDPOINT_NAME}"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[f"arn:aws:bedrock:{self.region}::"
                       f"foundation-model/{KIMI_MODEL}"],
        ))
        table.grant_read_write_data(role)

        # ---- 默认 VPC + 安全组（放行 8501，仅临时 demo）----
        vpc = ec2.Vpc.from_lookup(self, "Vpc", is_default=True)
        sg = ec2.SecurityGroup(self, "Sg", vpc=vpc, allow_all_outbound=True)
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(8501), "Streamlit")

        # ---- user-data：拉代码、装依赖、起 Streamlit ----
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "dnf install -y python3-pip git",
            f"git clone {REPO_URL} /opt/app || true",
            "pip3 install -r /opt/app/app/requirements.txt",
            f"cd /opt/app/app && AWS_REGION={self.region} "
            "ENDPOINT_NAME=" + ENDPOINT_NAME + " "
            "DYNAMODB_TABLE=" + TABLE_NAME + " "
            "nohup streamlit run main.py --server.address 0.0.0.0 "
            "> /var/log/streamlit.log 2>&1 &",
        )

        instance = ec2.Instance(
            self, "App",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType("t3.small"),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(),
            role=role,
            security_group=sg,
            user_data=user_data,
            associate_public_ip_address=True,
        )

        CfnOutput(self, "StreamlitUrl",
                  value=f"http://{instance.instance_public_ip}:8501")
        CfnOutput(self, "Reminder",
                  value="演示完别忘了 delete-endpoint："
                        f"aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME}")
