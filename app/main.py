"""马铃薯叶片疾病诊断 Demo —— Streamlit 前端。

直接用 boto3 调用三个服务：
  - SageMaker Real-time Endpoint (YOLO11n-cls 分类)
  - Amazon Bedrock Kimi K2.5      (防治建议)
  - DynamoDB                      (历史记录)

启动：
  pip install -r requirements.txt
  streamlit run main.py --server.address 0.0.0.0
"""
import os
import io
import json
import uuid
import base64
from datetime import datetime, timezone

import boto3
import streamlit as st
from PIL import Image

# ---------- 配置（可用环境变量覆盖）----------
REGION = os.environ.get("AWS_REGION", "us-east-1")
ENDPOINT = os.environ.get("ENDPOINT_NAME", "potato-disease-demo")
TABLE = os.environ.get("DYNAMODB_TABLE", "potato-disease-demo-records")
KIMI_MODEL = os.environ.get("KIMI_MODEL", "moonshotai.kimi-k2.5")

smr = boto3.client("sagemaker-runtime", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)

CLASS_COLORS = {"Healthy": "🟢", "Early Blight": "🟡", "Late Blight": "🔴"}
FALLBACK_ADVICE = "⚠️ LLM 暂不可用，已记录检测结果。请咨询当地农技人员。"


# ---------- 后端调用 ----------
def classify(image_bytes: bytes) -> dict:
    resp = smr.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps({"image": base64.b64encode(image_bytes).decode()}),
    )
    r = json.loads(resp["Body"].read())
    names, probs = r["names"], r["probs"]
    return {
        "class": names[r["top1"]],
        "confidence": float(r["top1conf"]),
        "all_probs": dict(zip(names, probs)),
    }


def advise(disease: str, confidence: float) -> tuple[str, str]:
    """返回 (建议文本, 状态)。LLM 失败时降级，不阻断主流程。"""
    prompt = f"""你是一名农业专家，请根据以下马铃薯叶片检测结果给出防治建议：

检测结果：{disease}（置信度：{confidence:.1%}）

请提供：1.疾病描述（健康则跳过）2.危害程度 3.防治措施(3-5条) 4.预防建议
用简洁专业、农户易懂的语言。"""
    try:
        resp = bedrock.converse(
            modelId=KIMI_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.3},
        )
        return resp["output"]["message"]["content"][0]["text"], "ok"
    except Exception as e:  # noqa: BLE001 - demo 降级
        print(f"[LLM 降级] {e}")
        return FALLBACK_ADVICE, "degraded"


def save(disease, confidence, all_probs, advice, llm_status):
    table.put_item(Item={
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disease_class": disease,
        "confidence": str(round(confidence, 4)),
        "all_probs": {k: str(round(v, 4)) for k, v in all_probs.items()},
        "advice": advice,
        "llm_status": llm_status,
    })


def history(limit: int = 20) -> list:
    items = table.scan().get("Items", [])  # demo 量小，scan 足够
    return sorted(items, key=lambda x: x["timestamp"], reverse=True)[:limit]


# ---------- 页面 ----------
st.set_page_config(page_title="马铃薯叶片疾病诊断 Demo", layout="wide")
tab_diag, tab_hist = st.tabs(["🔬 诊断", "📋 历史记录"])

with tab_diag:
    st.title("🌿 马铃薯叶片疾病智能诊断")
    st.caption("YOLO11n-cls (SageMaker) + Kimi K2.5 (Amazon Bedrock)")

    up = st.file_uploader("上传叶片图片", type=["jpg", "jpeg", "png"])
    if up:
        col1, col2 = st.columns(2)
        col1.image(Image.open(up), caption="待检测", use_container_width=True)

        if col2.button("开始诊断", type="primary"):
            img_bytes = up.getvalue()
            try:
                with st.spinner("SageMaker 分类中..."):
                    cv = classify(img_bytes)
                with st.spinner("Kimi 生成建议中..."):
                    advice, status = advise(cv["class"], cv["confidence"])
                save(cv["class"], cv["confidence"], cv["all_probs"], advice, status)
            except Exception as e:  # noqa: BLE001
                col2.error(f"诊断失败：{e}")
            else:
                with col2:
                    st.markdown(f"## {CLASS_COLORS.get(cv['class'], '')} **{cv['class']}**")
                    st.progress(cv["confidence"], text=f"置信度 {cv['confidence']:.1%}")
                    for k, v in cv["all_probs"].items():
                        st.metric(k, f"{v:.1%}")
                    st.markdown("### 🌱 防治建议")
                    (st.warning if status == "degraded" else st.info)(advice)

with tab_hist:
    st.title("📋 历史诊断记录")
    if st.button("刷新"):
        st.rerun()
    records = history()
    if not records:
        st.info("暂无记录，先到「诊断」页上传一张图片。")
    for it in records:
        c = it["disease_class"]
        with st.expander(
            f"{CLASS_COLORS.get(c, '')} {c} · "
            f"{float(it['confidence']):.1%} · {it['timestamp'][:19]}"
        ):
            st.write(it["advice"])
