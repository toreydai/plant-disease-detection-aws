#!/usr/bin/env python3
"""CDK 入口：单 Stack 轻量 demo 基础设施（DynamoDB + IAM + EC2）。

训练与 SageMaker Endpoint 仍由 notebooks/ 用 SageMaker SDK 部署，
不在 CDK 管理范围内（见 docs/design-demo.md 第 7 节）。
"""
import os

import aws_cdk as cdk

from demo_stack import DemoStack

app = cdk.App()
DemoStack(
    app,
    "PotatoDiseaseDemo",
    env=cdk.Environment(
        account=os.environ["CDK_DEFAULT_ACCOUNT"],
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
)
app.synth()
