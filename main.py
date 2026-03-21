import os
import cv2
import uuid
import base64
import numpy as np
import requests
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# 加载 ONNX 模型
# ==========================
MODEL_PATH = "yolov5n.onnx"
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# 只保留危险类别
DANGER_CLASS_MAP = {
    0: "person",
    2: "car",
    5: "bus",
    7: "truck"
}

INPUT_SIZE = 640
CONF_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45


# ==========================
# 上传音频到 Supabase
# ==========================
def upload_audio(audio_content):
    filename = f"{uuid.uuid4()}.mp3"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "audio/mpeg"
    }

    upload_url = f"{SUPABASE_URL}/storage/v1/object/ai-files/tts/{filename}"
    r = requests.put(upload_url, data=audio_content, headers=headers, timeout=30)

    # Supabase 这里可能返回 200 或 201
    if r.status_code in [200, 201]:
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/ai-files/tts/{filename}"
        print("Supabase upload success:", public_url)
        return public_url

    print("Supabase upload failed:", r.status_code, r.text)
    return ""


# ==========================
# 图片预处理
# ==========================
def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return image, r, dw, dh


# ==========================
# YOLO ONNX 推理
# ==========================
def detect_objects(image):
    original = image.copy()
    h0, w0 = original.shape[:2]

    img, r, dw, dh = letterbox(original, (INPUT_SIZE, INPUT_SIZE))
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1 / 255.0,
        size=(INPUT_SIZE, INPUT_SIZE),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward()
    predictions = outputs[0]

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        obj_conf = float(pred[4])
        if obj_conf < CONF_THRESHOLD:
            continue

        class_scores = pred[5:]
        class_id = int(np.argmax(class_scores))
        class_score = float(class_scores[class_id])

        confidence = obj_conf * class_score
        if confidence < SCORE_THRESHOLD:
            continue

        cx, cy, w, h = pred[0:4]

        x = (cx - w / 2 - dw) / r
        y = (cy - h / 2 - dh) / r
        w = w / r
        h = h / r

        x = max(0, min(x, w0 - 1))
        y = max(0, min(y, h0 - 1))
        w = max(0, min(w, w0 - x))
        h = max(0, min(h, h0 - y))

        boxes.append([int(x), int(y), int(w), int(h)])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        SCORE_THRESHOLD,
        NMS_THRESHOLD
    )

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            class_id = class_ids[i]

            if class_id in DANGER_CLASS_MAP:
                detections.append({
                    "class_id": class_id,
                    "class_name": DANGER_CLASS_MAP[class_id],
                    "confidence": confidences[i],
                    "box": boxes[i]
                })

    return detections


# ==========================
# 自动危险检测
# ==========================
@app.route("/detect", methods=["POST"])
def detect():
    try:
        img_bytes = request.get_data()

        if not img_bytes:
            return jsonify({
                "danger": False,
                "text": "",
                "audio_url": ""
            })

        npimg = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "danger": False,
                "text": "Bild konnte nicht gelesen werden.",
                "audio_url": ""
            })

        detections = detect_objects(image)

        danger = False
        warning_text = ""

        for det in detections:
            class_name = det["class_name"]

            if class_name == "car":
                danger = True
                warning_text = "Achtung, ein Auto nähert sich."
                break
            elif class_name == "bus":
                danger = True
                warning_text = "Achtung, ein Bus kommt."
                break
            elif class_name == "truck":
                danger = True
                warning_text = "Achtung, ein Lastwagen nähert sich."
                break
            elif class_name == "person":
                danger = True
                warning_text = "Person vor Ihnen, bitte vorsichtig gehen."
                break

        audio_url = ""

        if danger and warning_text.strip():
            speech = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=warning_text
            )

            # 调试日志
            print("[detect] warning_text =", warning_text)
            print("[detect] tts bytes =", len(speech.content) if speech and speech.content else 0)

            audio_url = upload_audio(speech.content)
            print("[detect] audio_url =", audio_url)

        return jsonify({
            "danger": danger,
            "text": warning_text,
            "audio_url": audio_url
        })

    except Exception as e:
        print("[detect] error:", str(e))
        return jsonify({
            "danger": False,
            "text": str(e),
            "audio_url": ""
        }), 500


# ==========================
# 按钮触发 AI 场景描述
# ==========================
@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    try:
        img_bytes = request.get_data()

        if not img_bytes:
            return jsonify({
                "text": "Kein Bild empfangen.",
                "audio_url": ""
            }), 400

        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein hilfreicher Assistent für blinde Nutzer. "
                        "Beschreibe die aktuelle Umgebung kurz, klar, praktisch und auf Deutsch."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Beschreibe bitte die aktuelle Szene kurz und hilfreich."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        text = response.choices[0].message.content.strip()
        audio_url = ""

        if text:
            speech = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )

            # 调试日志
            print("[ask_ai] text =", text)
            print("[ask_ai] tts bytes =", len(speech.content) if speech and speech.content else 0)

            audio_url = upload_audio(speech.content)
            print("[ask_ai] audio_url =", audio_url)

        return jsonify({
            "text": text,
            "audio_url": audio_url
        })

    except Exception as e:
        print("[ask_ai] error:", str(e))
        return jsonify({
            "text": str(e),
            "audio_url": ""
        }), 500


# ==========================
# 测试接口
# ==========================
@app.route("/test")
def test():
    return jsonify({"status": "server running"})

 port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
