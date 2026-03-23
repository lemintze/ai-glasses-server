import os
import cv2
import base64
import wave
import io
import numpy as np

from flask import Flask, request, jsonify, send_file, send_from_directory
from openai import OpenAI

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# 路径配置
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov5n.onnx")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
LATEST_AUDIO_PATH = os.path.join(BASE_DIR, "latest_audio.wav")

# 预录危险提示音频
DANGER_AUDIO_MAP = {
    "car": "car.wav",
    "bus": "bus.wav",
    "truck": "truck.wav",
    "person": "person.wav",
}

# ==========================
# 加载 ONNX 模型
# ==========================
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

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
# PCM -> WAV
# OpenAI PCM: 24kHz, 16-bit, mono
# ==========================
def pcm_to_wav_bytes(pcm_bytes, sample_rate=16000, channels=1, sample_width=2):
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_buffer.getvalue()


# ==========================
# 生成 AI 语音并写到本地 latest_audio.wav
# ==========================
def generate_latest_tts_file(text, voice="alloy", speed=1.5):
    speech = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        speed=speed,
        response_format="pcm"
    )

    pcm_bytes = speech.content if speech and speech.content else b""
    print("[tts] pcm bytes =", len(pcm_bytes))

    if not pcm_bytes:
        return False

    wav_bytes = pcm_to_wav_bytes(pcm_bytes)
    print("[tts] wav bytes =", len(wav_bytes))

    with open(LATEST_AUDIO_PATH, "wb") as f:
        f.write(wav_bytes)

    print("[tts] latest audio saved to", LATEST_AUDIO_PATH)
    return True


# ==========================
# 获取最新音频固定地址
# ==========================
def get_latest_audio_url():
    return request.host_url.rstrip("/") + "/latest_audio.wav"


# ==========================
# 获取预录危险音频固定地址
# ==========================
def get_danger_audio_url(filename):
    return request.host_url.rstrip("/") + f"/audio/{filename}"


# ==========================
# 文件服务
# ==========================
@app.route("/latest_audio.wav", methods=["GET"])
def latest_audio():
    if not os.path.exists(LATEST_AUDIO_PATH):
        return jsonify({"error": "latest audio not found"}), 404

    return send_file(
        LATEST_AUDIO_PATH,
        mimetype="audio/wav",
        as_attachment=False,
        download_name="latest_audio.wav",
        max_age=0
    )


@app.route("/audio/<path:filename>", methods=["GET"])
def serve_audio(filename):
    return send_from_directory(
        AUDIO_DIR,
        filename,
        as_attachment=False,
        max_age=0
    )


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
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
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
# 不再实时生成 TTS
# 直接返回你预录好的本地音频文件
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

        img_height, img_width = image.shape[:2]
        detections = detect_objects(image)

        danger = False
        warning_text = ""
        warning_class = ""

        for det in detections:
            class_name = det["class_name"]
            box = det["box"]

            box_area = box[2] * box[3]
            box_center_y = box[1] + box[3] / 2
            area_ratio = box_area / (img_width * img_height)
            position_ratio = box_center_y / img_height

            if area_ratio > 0.08 and position_ratio > 0.5:
                if class_name == "car":
                    danger = True
                    warning_class = "car"
                    warning_text = "Achtung, ein Auto nähert sich."
                    break
                elif class_name == "bus":
                    danger = True
                    warning_class = "bus"
                    warning_text = "Achtung, ein Bus kommt."
                    break
                elif class_name == "truck":
                    danger = True
                    warning_class = "truck"
                    warning_text = "Achtung, ein Lastwagen nähert sich."
                    break
                elif class_name == "person":
                    danger = True
                    warning_class = "person"
                    warning_text = "Person vor Ihnen, bitte vorsichtig gehen."
                    break
            else:
                print(
                    f"⚠️ {class_name} 距离较远 "
                    f"(area={area_ratio:.3f}, pos={position_ratio:.3f})，不播报"
                )

        audio_url = ""
        if danger and warning_class in DANGER_AUDIO_MAP:
            filename = DANGER_AUDIO_MAP[warning_class]
            audio_url = get_danger_audio_url(filename)
            print("[detect] use prerecorded audio:", filename)

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
# 生成 latest_audio.wav 并返回固定地址
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
                        "REGELN: "
                        "1. Maximal 1-2 Sätze (unter 30 Wörtern). "
                        "2. Nur wichtige Informationen: Objekte, Personen, Gefahren, Hindernisse. "
                        "3. Keine Füllwörter wie 'ich sehe', 'es gibt', 'auf dem Bild'. "
                        "4. Direkt und praktisch. "
                        "5. Auf Deutsch."
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
            print("[ask_ai] text =", text)
            ok = generate_latest_tts_file(
                text,
                voice="alloy",
                speed=1.5
            )
            if ok:
                audio_url = get_latest_audio_url()
                print("[ask_ai] latest audio url =", audio_url)

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
