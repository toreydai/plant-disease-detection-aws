# 马铃薯叶片疾病智能诊断

## 实验简介

本实验基于 [AWS 博文](https://aws.amazon.com/cn/blogs/china/potato-leaf-disease-recognition-and-classification-using-the-deepseek-r1-model-and-a-computer-vision-fine-tuning-model/) 实现马铃薯叶片疾病的自动识别与防治建议生成。

上传一张马铃薯叶片照片，系统自动完成三件事：

1. **SageMaker Real-time Endpoint**（YOLO11n-cls）：识别叶片是否患有早疫病、晚疫病或健康
2. **Amazon Bedrock Kimi K2.5**：根据诊断结果实时生成中文防治建议
3. **Amazon DynamoDB**：记录每次诊断历史，支持回溯查询

---

## 实验内容

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

## 实验流程

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

## 实验步骤

### 步骤 1：克隆代码并准备数据集

```bash
git clone <本仓库地址>
cd plant-disease-detection-aws
```

从 [PlantVillage 数据集](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) 下载马铃薯三分类图片，按如下结构放置后上传到 SageMaker Notebook 实例：

```
data/
├── train/
│   ├── Early Blight/   # 800 张
│   ├── Healthy/        # 122 张
│   └── Late Blight/    # 800 张
└── val/
    ├── Early Blight/   # 200 张
    ├── Healthy/        # 30 张
    └── Late Blight/    # 200 张
```

**预期输出**：`data/` 目录结构就绪，train 共 1722 张，val 共 430 张。

---

### 步骤 2：训练四模型并对比

在 SageMaker Notebook 实例中打开 `notebooks/train.ipynb`，按环境选择执行路径：

- **方式 A（有 GPU Notebook 实例）**：运行 Section A，四模型约 15 min；仅训 YOLO 可将顶部 `MODELS_TO_TRAIN` 改为 `['yolo']`，约 10 min
- **方式 B（无本地 GPU）**：运行 Section B，提交 SageMaker Training Job 到 ml.g5.2xlarge（约 15 min），训完自动打印 `MODEL_DATA_S3` 路径

**预期输出**：

```
=== 四模型对比 ===
        model val_acc_pct  train_min  params_m
  YOLO11n-cls     100.00%        7.7      1.53
      ResNet50     100.00%        1.7     23.51
  MobileNetV3     100.00%        0.6      4.21
    PotatoCNN      93.95%        0.8      1.21
```

方式 A 产出 `runs/classify/train/weights/best.pt`；方式 B 打印 `MODEL_DATA_S3` S3 路径。

---

### 步骤 3：部署 SageMaker Real-time Endpoint

打开 `notebooks/deploy.ipynb`，根据训练方式设置 Cell 1 顶部变量：

```python
MODEL_SOURCE = 'local'   # 方式 A：打包本地 best.pt 上传 S3 再部署
# 或
MODEL_SOURCE = 's3'      # 方式 B：填入上一步打印的 MODEL_DATA_S3 路径
```

逐格运行，约 10 min。

**预期输出**：

```
Endpoint 已部署：potato-disease-demo
# 预热调用返回：
{"names": ["Early Blight", "Healthy", "Late Blight"], "top1": 1, "top1conf": 0.998, ...}
```

SageMaker 控制台中 `potato-disease-demo` 状态为 **InService**。

---

### 步骤 4：创建 DynamoDB 表并配置 IAM

**建表**（On-demand 计费，演示量级成本可忽略）：

```bash
aws dynamodb create-table \
  --table-name potato-disease-demo-records \
  --attribute-definitions AttributeName=id,AttributeType=S \
  --key-schema AttributeName=id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

**EC2 实例角色最小权限**（新建或附加到已有角色）：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "sagemaker:InvokeEndpoint",
      "Resource": "arn:aws:sagemaker:us-east-1:*:endpoint/potato-disease-demo" },
    { "Effect": "Allow", "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/moonshotai.kimi-k2.5" },
    { "Effect": "Allow", "Action": ["dynamodb:PutItem", "dynamodb:Scan"],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/potato-disease-demo-records" }
  ]
}
```

**预期输出**：DynamoDB 控制台中表状态为 **ACTIVE**。

---

### 步骤 5：启动 Streamlit 前端

在已挂好 IAM 角色的 EC2 上（安全组放行 TCP 8501）：

```bash
pip install -r app/requirements.txt
streamlit run app/main.py --server.address 0.0.0.0
```

**预期输出**：

```
You can now view your Streamlit app in your browser.
Network URL: http://<私有IP>:8501
External URL: http://<公网IP>:8501
```

浏览器打开 `http://<EC2公网IP>:8501`，出现「诊断」和「历史记录」两个标签页。

> **首次使用前先预热 Endpoint**：在「诊断」Tab 随意上传一张图片调用一次，等待返回结果（约 3-5s），此后响应将稳定在 0.1s 级别。

---

### 步骤 6（可选）：一键流水线 `run_pipeline.py`

如需重新训练并自动更新 Endpoint，在任意有 AWS 凭证的机器上运行：

```bash
python3 run_pipeline.py              # 四模型全训
python3 run_pipeline.py --yolo-only  # 仅训 YOLO（约 10 min）
```

**预期输出**：

```
[HH:MM:SS] Step 1: 打包 sagemaker/ 源代码 → S3  ✓
[HH:MM:SS] Step 2: 提交 Training Job: potato-4models-<timestamp>  ✓
[HH:MM:SS] Step 3: 等待训练完成...  ✓ 计费 919s
[HH:MM:SS] Step 4: === 四模型对比结果 ===  ✓
[HH:MM:SS] Step 5-7: 创建 Model → EndpointConfig → 更新 Endpoint  ✓
[HH:MM:SS] ✓ 全流程完成！Endpoint: potato-disease-demo (InService)
```

---

### 步骤 7（可选）：批量精度验证

```bash
python3 test_100.py
```

**预期输出**：

```
val 集共 430 张，随机抽 100 张
...
总体准确率: 100/100 = 100.0%
总耗时: 11.2s  均值: 0.11s/张
```

---

## 实验总结

| 指标 | 结果 |
|------|------|
| YOLO11n-cls val_acc | **100%**（300 epochs，1.53M 参数） |
| ResNet50 val_acc | **100%**（ImageNet 预训练，23.51M 参数） |
| MobileNetV3 val_acc | **100%**（ImageNet 预训练，4.21M 参数，训练最快 0.6 min） |
| PotatoCNN val_acc | 93.95%（从零训练，1.21M 参数） |
| Endpoint 推理延迟 | ~0.11s/张（含网络，ml.m5.large） |
| 端到端总耗时 | 约 40 分钟 |

三个预训练模型在该数据集上均达到 100% val_acc，与原博文结论一致。数据集中 Healthy 类仅 152 张，样本分布不均衡，实际场景部署前建议补充更多健康叶片样本。

详细验证结果与踩坑记录见 [`docs/test-results.md`](docs/test-results.md)，技术设计细节见 [`docs/design-demo.md`](docs/design-demo.md)。

---

## 环境清理

实验结束后删除所有资源，避免持续计费：

```bash
# 删除 SageMaker Endpoint（主要计费项）
aws sagemaker delete-endpoint --endpoint-name potato-disease-demo --region us-east-1

# 删除 DynamoDB 表
aws dynamodb delete-table --table-name potato-disease-demo-records --region us-east-1

# 若使用 CDK 部署前端
cd infra && cdk destroy

# 若手动启动 EC2，在控制台终止实例
```

**费用参考**（完整实验一次）：

| 资源 | 费用 |
|------|------|
| Training Job（ml.g5.2xlarge，~15 min） | ~$0.5 |
| Endpoint（ml.m5.large，按小时） | ~$0.115/h |
| EC2 前端（t3.small，按小时） | ~$0.023/h |
| Bedrock Kimi K2.5 | 按 token，每次建议 < $0.01 |
| DynamoDB On-demand | 演示量级 ≈ $0 |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## 免责声明

- 本项目仅供学习与技术参考，不构成任何商业建议或生产部署方案。
- 运行本实验将产生 AWS 资源费用，请参考上方费用表并在实验结束后及时清理资源。
- 模型基于 PlantVillage 公开数据集训练，仅在该数据集上验证，实际农业场景中的准确率可能存在差异。
- 本项目与 Amazon Web Services、Moonshot AI 等厂商无官方关联，相关服务的可用性和定价以官方文档为准。
