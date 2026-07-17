# 端到端测试结果 —— 马铃薯叶片疾病诊断 Demo

> 最新验证于 2026-06-08 · 区域 us-east-1

## 训练结果（第 4 轮，2026-06-08 08:07 UTC）

- Training Job：`potato-4models-<timestamp>` (ml.g5.2xlarge) · Completed（计费 919s / 15.3 min）
- 部署包：`s3://sagemaker-<region>-<account-id>/potato-demo/deploy/model_v2.tar.gz`

| 模型 | val_acc | 训练时长 | 参数量 | 说明 |
|------|---------|---------|--------|------|
| YOLO11n-cls | **100%** | 7.7 min | 1.53M | 300 epochs，原博文基准模型 |
| ResNet50 | **100%** | 1.7 min | 23.51M | ImageNet 预训练 + warmup 5 epoch |
| MobileNetV3 | **100%** | 0.6 min | 4.21M | ImageNet 预训练，最快收敛 |
| PotatoCNN | 93.95% | 0.8 min | 1.21M | 从零训练，参数最少 |

## 推理测试（100 张随机 val 图）

- Endpoint：`potato-disease-demo` (ml.m5.large) · **InService**
- 测试时间：2026-06-08 · 从 S3 val 集随机抽取

| 类别 | 样本数 | 正确 | 准确率 |
|------|--------|------|--------|
| Early Blight | 46 | 46 | **100%** |
| Healthy | 7 | 7 | **100%** |
| Late Blight | 47 | 47 | **100%** |
| **合计** | **100** | **100** | **100%** |

均值推理延迟：**0.11s/张**（含 S3 下载）

---

## 数据集

- 来源：GitHub 公开镜像 `spMohanty/PlantVillage-Dataset`
- 三类原始数量：Early Blight 1000 / Late Blight 1000 / Healthy 152
- 80/20 切分：train 1722 张 / val 430 张
- 已上传：`s3://sagemaker-<region>-<account-id>/potato-demo/data/{train,val}/`

> val 集与训练集同源，100% 准确率说明管线正确，非泛化精度指标。Healthy 类样本偏少，生产建议补充。

---

## 踩坑记录

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | Training Job 失败：`manifest unknown` | PyTorch 2.1+ DLC 用 **cu121** 而非 cu118 | 改用 `pytorch-training:2.3.0-gpu-py311-cu121` |
| 2 | Training Job 失败：`RuntimeError: Numpy is not available` | ultralytics 将 numpy 升到 2.x，与 DLC 内 torch 不兼容 | `sagemaker/requirements.txt` 加 `numpy<2` |
| 3 | EC2 user-data `dnf install python3-pip` 失败 | cloud-init 启动时 rpm 锁未释放 | 开机后经 SSM 补装 |
| 4 | `pip install streamlit` 报 `Cannot uninstall requests` | 系统 rpm 装的 requests pip 无法卸载 | 用 venv（`/opt/venv`）隔离安装 |
| 5 | Training Job 失败：`FileNotFoundError: 'yolo26n.pt'` | ultralytics 8.4.x 将 `yolo11n-cls.pt` 内部重映射 | `sagemaker/requirements.txt` 锁定 `ultralytics==8.3.135` |

---

## 清理

```bash
aws sagemaker delete-endpoint --endpoint-name potato-disease-demo --region us-east-1
aws dynamodb delete-table --table-name potato-disease-demo-records --region us-east-1
# 若用 CDK 部署前端：cd infra && cdk destroy
# 若手动 EC2：aws ec2 terminate-instances --instance-ids <ID> --region us-east-1
```
