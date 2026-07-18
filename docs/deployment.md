# 马铃薯叶片疾病智能诊断 — Demo 设计（轻量版）

> **定位**：最小可行 demo，目标是快速跑通、效果直观。
> 本文档只保留必要组件，省略生产级中间件与运维设施。

---

## 1. Demo 一句话

> 上传一张马铃薯叶片照片 → SageMaker 秒级分类（健康 / 早疫病 / 晚疫病）→ Bedrock Kimi K2.5 实时生成中文防治建议 → 自动存入历史记录。

整个 demo 只有 **3 个 AWS 触点**：SageMaker Endpoint、Bedrock、DynamoDB。一个 Streamlit 应用把它们串起来。

---

## 2. 架构（极简）

```
       ┌─────────────────────────────────────────┐
       │   Streamlit 应用                          │
       │   (跑在 SageMaker Notebook 或一台 EC2)    │
       │                                           │
       │   [诊断] tab        [历史] tab            │
       └───┬──────────────┬─────────────┬─────────┘
           │ boto3        │ boto3       │ boto3
           ▼              ▼             ▼
   ┌───────────────┐ ┌──────────┐ ┌────────────┐
   │ SageMaker     │ │ Bedrock  │ │ DynamoDB   │
   │ Real-time     │ │ Kimi K2.5│ │ (历史记录) │
   │ Endpoint      │ │          │ │            │
   │ YOLO11n-cls   │ └──────────┘ └────────────┘
   └───────────────┘
```

**为什么用 SageMaker Endpoint**：体现 AWS 托管推理能力，与原博文一致。demo 默认用 **Real-time `ml.m5.large` 小实例**：延迟稳定、演示几小时成本可忽略、结束即删。比 Serverless 更适合演示场景——Serverless 冷启动 + ultralytics 重依赖首次调用可能十几秒到分钟级。
> 备选：流量极低且不介意冷启动可换 Serverless；想更极简可不部署 endpoint，直接在 Streamlit 进程里 `model = YOLO('best.pt')` 本地推理（模型仅 ~3MB，CPU 2ms），连 endpoint 都省了。

---

## 3. 训练（`notebooks/train.ipynb`）

数据用公开的 PlantVillage 马铃薯子集（三分类），同时训练四个模型并输出对比表。按环境选择：

- **Section A（有 GPU Notebook 实例）**：直接在 Notebook 里运行，四模型约 15 min；仅 YOLO 改顶部 `MODELS_TO_TRAIN=['yolo']` 约 10 min
- **Section B（无本地 GPU）**：提交 SageMaker Training Job（ml.g5.2xlarge），训完打印 `MODEL_DATA_S3` 路径供 deploy.ipynb 使用

> 原博文实测 YOLO11n-cls 验证/测试准确率均达 100%（300 epochs，ml.g5.2xlarge）。

---

## 4. 部署 Endpoint

### 4.1 推理脚本与依赖

完整实现见 [`sagemaker/inference.py`](../sagemaker/inference.py)。关键点：

- 返回 `names`（类别标签数组）而非硬编码顺序，调用方按 `names[top1]` 取标签
- `sagemaker/requirements.txt` 必须用 `opencv-python-headless`——SageMaker 无头容器缺 `libGL.so.1`，普通 opencv 会 ImportError

### 4.2 打包 model.tar.gz 并上传

把训练产出的 `best.pt` 与推理脚本（`sagemaker/inference.py` + `sagemaker/requirements.txt`）打成符合 SageMaker 约定的结构：

```
model.tar.gz
├── best.pt
└── code/
    ├── inference.py
    └── requirements.txt
```

```bash
mkdir -p model/code
cp runs/classify/train/weights/best.pt model/
cp sagemaker/inference.py sagemaker/requirements.txt model/code/
tar czf model.tar.gz -C model .
aws s3 cp model.tar.gz s3://<your-bucket>/model.tar.gz
```

### 4.3 部署 Real-time Endpoint

完整流程见 [`notebooks/deploy.ipynb`](../notebooks/deploy.ipynb)。支持两种模型来源：

- `MODEL_SOURCE='local'`：打包本地 `best.pt` + `sagemaker/` 推理脚本上传 S3，再部署（Section A 训练后用）
- `MODEL_SOURCE='s3'`：直接填 Training Job 输出的 `MODEL_DATA_S3` 路径部署（Section B 训练后用）

> 选 Real-time `ml.m5.large` 是为了**现场稳定**：常驻、无冷启动、延迟可预期，demo 几小时成本可忽略，结束 `delete-endpoint` 即清。

### 4.4 演示前预热（重要）

首次调用会触发模型加载。**正式开讲前，先静默调一次 endpoint**，让它就绪，避免现场第一次点"诊断"时干等：

```python
import json, base64
with open("sample_leaf.jpg", "rb") as f:
    predictor.sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName="potato-disease-demo", ContentType="application/json",
        Body=json.dumps({"image": base64.b64encode(f.read()).decode()}))
```

---

## 5. Streamlit 应用（核心）

完整实现见 [`app/main.py`](../app/main.py)。一个文件搞定：YOLO 分类 + Kimi 建议 + 历史记录，boto3 直连三个服务，两个 tab（诊断 / 历史）。

**跑在哪 / 怎么访问**：推荐一台 **EC2**（最省事）。安全组放行 **8501** 端口，给实例挂第 6 节的 IAM 角色，然后：

```bash
pip install streamlit boto3 pillow
streamlit run app/main.py --server.address 0.0.0.0
# 浏览器打开 http://<EC2公网IP>:8501
```

> 不建议在 SageMaker Notebook 实例里跑 Streamlit——8501 端口默认无法直接浏览器访问，需要 jupyter-server-proxy 或 SSH 隧道，演示现场容易卡在"起好了却打不开"。EC2 直连最稳。

---

## 6. IAM（一个角色搞定）

demo 不拆角色。Streamlit 跑在 SageMaker Notebook（用其执行角色）或 EC2（用实例角色），该角色挂如下权限即可：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "sagemaker:InvokeEndpoint",
      "Resource": "arn:aws:sagemaker:us-east-1:*:endpoint/potato-disease-demo" },
    { "Effect": "Allow", "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/moonshotai.kimi-k2.5" },
    { "Effect": "Allow",
      "Action": ["dynamodb:PutItem", "dynamodb:Scan"],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/potato-disease-demo-records" }
  ]
}
```

> Bedrock 第三方模型的 InvokeModel 资源 ARN 以控制台「API 请求」示例为准，部署前试调一次确认。

DynamoDB 建表（一条命令，On-demand）：

```bash
aws dynamodb create-table \
  --table-name potato-disease-demo-records \
  --attribute-definitions AttributeName=id,AttributeType=S \
  --key-schema AttributeName=id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

---

## 7. 可选：单 Stack 轻量 CDK（反复演示用）

手动方式（第 6 节 + CLI 建表 + EC2 手动跑）最快上手，适合一次性演示。若要**反复演示**、希望一键起停且清理干净，用一个**单 Stack** CDK 管理 DynamoDB + IAM + EC2（含自动跑 Streamlit）。

**混合边界**（demo 的合理切法）：
- **Notebook 负责**：训练 + `model.deploy()` 部署 Endpoint —— SDK 自动处理推理镜像与 code 打包，比 CDK L1 简洁得多。
- **CDK 负责**：DynamoDB 表 + EC2（IAM 角色 / 安全组 / user-data 起 Streamlit）—— 一键 `cdk deploy` / `cdk destroy`。
- **清理**：`cdk destroy` 拆表和 EC2；Endpoint 单独 `delete-endpoint`。

完整实现见 [`infra/app.py`](../infra/app.py) 和 [`infra/demo_stack.py`](../infra/demo_stack.py)。Stack 包含：DynamoDB 表（RemovalPolicy.DESTROY）、EC2 实例角色（最小权限）、安全组（8501）、user-data 自动拉代码起 Streamlit。

部署 / 销毁：

```bash
cd infra && pip install aws-cdk-lib constructs
cdk deploy        # 起 DynamoDB + EC2（自动跑 Streamlit），输出访问 URL
# ...演示...
cdk destroy       # 拆掉表和 EC2
aws sagemaker delete-endpoint --endpoint-name potato-disease-demo   # Endpoint 单独删
```

> 安全组对 `0.0.0.0/0` 放行 8501 仅用于临时 demo；正式环境应限制来源 IP 或前置 ALB。user-data 用 `nohup &` 起 Streamlit，实例重启不自动拉起，demo 够用；要更稳可改 systemd。

---

## 8. 成本与清理

| 资源 | demo 成本 |
|------|----------|
| SageMaker Real-time Endpoint | `ml.m5.large` ~$0.115/h，演示几小时 ≈ 几毛钱（用完即删）|
| Bedrock Kimi K2.5 | 按 token，每次建议 ≈ 几分钱 |
| DynamoDB On-demand | 演示量级 ≈ $0 |
| 训练（一次性 ml.g5.2xlarge ~1h）| ~$1.5 |
| EC2（跑 Streamlit，t3.small）| 几小时 ≈ 几毛钱 |
| **演示期间合计** | **≈ $2-3** |

演示完一键清理：

```bash
aws sagemaker delete-endpoint --endpoint-name potato-disease-demo
aws dynamodb delete-table --table-name potato-disease-demo-records
```

---

## 9. 搭建速查（约 40 分钟）

| 步骤 | 操作 | 时间 |
|------|------|------|
| 1 | 准备 PlantVillage 马铃薯三分类数据，上传 Notebook | ~15min |
| 2 | 跑 `train.ipynb`（四模型对比；仅 YOLO 约 10min，全部约 15min）| ~10-15min |
| 3 | 打包 `model.tar.gz`（best.pt + code/inference.py + requirements.txt）上传 S3 | ~5min |
| 4 | `deploy.ipynb` 部署 Real-time Endpoint | ~10min |
| 5 | 建 DynamoDB 表 + 给 EC2/角色加权限 | ~5min |
| 6 | EC2 上 `streamlit run`，开放 8501，**先预热 endpoint** 再演示 | ~5min |

> 步骤 5-6 可用第 7 节的单 Stack CDK 一键替代（`cdk deploy` 起表 + EC2 + Streamlit），反复演示时更省事。

---

## 10. 项目目录（demo）

```
plant-disease-detection-aws/
├── docs/
│   ├── deployment.md              # 本文档（轻量 demo，含设计+部署）
│   └── testing.md             # 端到端验证结果 + 踩坑记录
├── notebooks/
│   ├── train.ipynb                 # 训练 & 四模型对比（Section A 本地 / Section B Training Job）
│   └── deploy.ipynb                # 部署 Real-time Endpoint（local / s3 两种来源）
├── sagemaker/
│   ├── train_job.py       # Training Job 容器入口（Section B 依赖）
│   ├── inference.py       # endpoint 推理脚本
│   └── requirements.txt   # 训练 + 推理共用依赖
├── app/
│   └── main.py                     # Streamlit（分类+建议+历史）
├── infra/                          # 可选：单 Stack 轻量 CDK（第 7 节）
│   ├── app.py
│   └── demo_stack.py               # DynamoDB + IAM + EC2(自动跑 Streamlit)
└── run_pipeline.py                 # 端到端一键流水线（训练 → 打包 → 部署）
```

---

*Demo 设计版本：v1.3 | 日期：2026-06-08*
