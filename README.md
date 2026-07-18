# 马铃薯叶片疾病智能诊断

## 项目简介

本项目基于 [AWS 博文](https://aws.amazon.com/cn/blogs/china/potato-leaf-disease-recognition-and-classification-using-the-deepseek-r1-model-and-a-computer-vision-fine-tuning-model/) 实现马铃薯叶片疾病的自动识别与防治建议生成。

上传一张马铃薯叶片照片，系统自动完成三件事：

1. **SageMaker Real-time Endpoint**（YOLO11n-cls）：识别叶片是否患有早疫病、晚疫病或健康
2. **Amazon Bedrock Kimi K2.5**：根据诊断结果实时生成中文防治建议
3. **Amazon DynamoDB**：记录每次诊断历史，支持回溯查询

---

## 目录结构

```
plant-disease-detection-aws/
├── notebooks/
│   ├── train.ipynb              # 训练 & 四模型对比（方式A本地 / 方式B Training Job）
│   └── deploy.ipynb             # 部署 SageMaker Real-time Endpoint
├── sagemaker/
│   ├── train_job.py             # Training Job 容器入口脚本
│   ├── inference.py             # SageMaker 推理入口脚本
│   └── requirements.txt         # 训练与推理共用依赖
├── app/
│   ├── main.py                  # Streamlit 前端（分类 + 建议 + 历史）
│   └── requirements.txt
├── infra/                       # 可选：CDK 一键部署（DynamoDB + IAM + EC2）
│   ├── app.py
│   ├── demo_stack.py
│   ├── cdk.json
│   └── requirements.txt
├── docs/
│   ├── deployment.md           # 架构与技术设计、完整分步部署指南
│   └── testing.md               # 验证结果与踩坑记录
├── run_pipeline.py              # 一键流水线：训练 → 打包 → 部署 Endpoint
└── test_100.py                  # 批量推理精度验证脚本
```

---

## 功能说明

- 使用 PlantVillage 公开数据集（三分类，共 2152 张）训练四个模型并进行横向对比：
  - YOLO11n-cls（原博文基准模型）
  - ResNet50（ImageNet 预训练 + 迁移学习）
  - MobileNetV3（轻量化，ImageNet 预训练）
  - PotatoCNN（从零训练自定义 CNN）
- 将 YOLO11n-cls 部署为 SageMaker Real-time Endpoint
- 通过 Streamlit 构建一键诊断 Web 界面，串联 SageMaker、Bedrock、DynamoDB 三个服务
- 提供 `run_pipeline.py` 一键完成训练→打包→部署全流程

---

## 适用场景

- 学习 SageMaker Training Job 与 Real-time Endpoint 的完整使用流程
- 探索 Amazon Bedrock 接入第三方大模型（Kimi K2.5）的方式
- 了解计算机视觉模型训练与推理的工程实践
- 构建 CV + LLM 组合应用的参考架构

---

## 处理流程

```
准备数据集
    ↓
train.ipynb：训练四模型 & 对比（约 15 min，ml.g5.2xlarge）
    ↓
deploy.ipynb：部署 SageMaker Real-time Endpoint（约 10 min）
    ↓
建 DynamoDB 表 + 配置 IAM 权限
    ↓
EC2 启动 Streamlit 前端（http://<IP>:8501）
    ↓
上传叶片图片 → 分类 → 建议 → 记录历史
```

架构：

```
Streamlit (EC2)
   ├─ boto3 → SageMaker Real-time Endpoint  (YOLO11n-cls 分类)
   ├─ boto3 → Bedrock Kimi K2.5             (防治建议)
   └─ boto3 → DynamoDB                      (历史记录)
```

---

## 前提条件

| 条件 | 说明 |
|------|------|
| AWS 账号 | 具备 SageMaker、Bedrock、DynamoDB、EC2 的操作权限 |
| Bedrock 模型访问 | 在 us-east-1 控制台开通 `moonshotai.kimi-k2.5` 的访问权限 |
| SageMaker 执行角色 | 账号下存在 SageMaker ExecutionRole（控制台创建 Notebook 实例时会自动生成） |
| GPU 实例配额 | us-east-1 `ml.g5.2xlarge` Training Job 配额 ≥ 1（默认通常已满足） |
| EC2 | 用于运行 Streamlit 前端，`t3.small` 即可 |

---

## 快速开始

大致 40 分钟跑通端到端：

```bash
git clone <本仓库地址> && cd plant-disease-detection-aws

# 1. SageMaker Notebook 中运行 notebooks/train.ipynb，训练四模型（约 10-15min）
# 2. notebooks/deploy.ipynb 部署 Real-time Endpoint（约 10min）
# 3. 建 DynamoDB 表 + 给 EC2/角色配置 IAM 权限（约 5min）
aws dynamodb create-table \
  --table-name potato-disease-demo-records \
  --attribute-definitions AttributeName=id,AttributeType=S \
  --key-schema AttributeName=id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST --region us-east-1

# 4. EC2 上启动 Streamlit 前端（约 5min，先预热一次 Endpoint 再演示）
pip install -r app/requirements.txt
streamlit run app/main.py --server.address 0.0.0.0
```

数据集准备、四模型训练对比、Endpoint 部署（含 local/s3 两种模型来源）、IAM 最小权限 policy、单 Stack CDK 一键部署等完整分步说明见 [`docs/deployment.md`](docs/deployment.md)，端到端验证结果与踩坑记录见 [`docs/testing.md`](docs/testing.md)。

## 可选脚本

```bash
python3 run_pipeline.py              # 端到端一键流水线：训练 → 打包 → 部署 Endpoint（四模型全训）
python3 run_pipeline.py --yolo-only  # 仅训 YOLO，约 10 min
python3 test_100.py                  # 批量精度验证：val 集随机抽 100 张跑推理
```

---

## 结果总结

| 指标 | 结果 |
|------|------|
| YOLO11n-cls val_acc | **100%**（300 epochs，1.53M 参数） |
| ResNet50 val_acc | **100%**（ImageNet 预训练，23.51M 参数） |
| MobileNetV3 val_acc | **100%**（ImageNet 预训练，4.21M 参数，训练最快 0.6 min） |
| PotatoCNN val_acc | 93.95%（从零训练，1.21M 参数） |
| Endpoint 推理延迟 | ~0.11s/张（含网络，ml.m5.large） |
| 端到端总耗时 | 约 40 分钟 |

三个预训练模型在该数据集上均达到 100% val_acc，与原博文结论一致。数据集中 Healthy 类仅 152 张，样本分布不均衡，实际场景部署前建议补充更多健康叶片样本。

---

## 清理

```bash
aws sagemaker delete-endpoint --endpoint-name potato-disease-demo --region us-east-1
aws dynamodb delete-table --table-name potato-disease-demo-records --region us-east-1
# 若用 CDK 部署前端：cd infra && cdk destroy
# 若手动启动 EC2，在控制台终止实例
```

费用参考（完整运行一次约 $2-3）见 [`docs/deployment.md`](docs/deployment.md#8-成本与清理)。

---

## License

MIT - see the [LICENSE](LICENSE) file for details.

---

## 免责声明

- 本项目仅供学习与技术参考，不构成生产部署方案。
- 运行过程中会创建 AWS 资源并产生费用，请在实验结束后及时清理。
- 作者不对因使用本项目产生的任何费用或损失承担责任。
- 本项目与 Amazon Web Services 无官方关联，相关服务的可用性与定价以 AWS 官方文档为准。
- 生产环境使用前请根据实际需求进行安全评估与调整。
