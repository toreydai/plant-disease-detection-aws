"""SageMaker PyTorch 推理脚本：YOLO11n-cls 马铃薯叶片三分类。

打包结构（model.tar.gz）：
    best.pt
    code/inference.py
    code/requirements.txt

SageMaker 托管 PyTorch 容器不预装 ultralytics，由 code/requirements.txt
在部署时自动安装；opencv 必须用 headless 版以避免无头容器缺 libGL.so.1。
"""
import base64
import io
import json

from PIL import Image
from ultralytics import YOLO


def model_fn(model_dir):
    """加载训练产出的 best.pt。"""
    return YOLO(f"{model_dir}/best.pt")


def input_fn(request_body, content_type="application/json"):
    """请求体：{"image": "<base64>"} → PIL.Image（RGB）。"""
    if content_type != "application/json":
        raise ValueError(f"不支持的 content type：{content_type}")
    img_b64 = json.loads(request_body)["image"]
    return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")


def predict_fn(image, model):
    """返回完整类别标签与概率，避免客户端硬编码类别顺序。"""
    result = model.predict(image, verbose=False)[0]
    names = [result.names[i] for i in range(len(result.names))]
    return {
        "names": names,
        "probs": result.probs.data.tolist(),
        "top1": int(result.probs.top1),
        "top1conf": float(result.probs.top1conf),
    }


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
