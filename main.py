# ============================================================================
# AI智能眼镜服务器 - HTTP版本（WAV稳定版）
# 改进重点：
# 1. TTS 直接请求 wav，而不是 pcm 后手动封装
# 2. latest_audio.wav 返回时禁缓存
# 3. 增加更详细的日志，方便确认文件大小和生成状态
# ============================================================================

import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from openai import OpenAI

app = Flask(__name__)

# ==========================
# 环境变量读取
# ==========================
print("=" * 70)
print("正在读取环境变量...")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
print(f"OPENAI_API_KEY: {'***' + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'NOT SET'}")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 环境变量未设置！")

print("✓ 环境变量读取成功")
print("=" * 70)

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# 路径配置
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov5n.onnx")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
LATEST_AUDIO_PATH = os.path.join(BASE_DIR, "latest_audio.wav")

for directory in [AUDIO_DIR]:
    os.makedirs(directory, exist_ok=True)

DANGER_AUDIO_MAP = {
    "car": "car.wav",
    "bus": "bus.wav",
    "truck": "truck.wav",
    "person": "person.wav",
}

# ==========================
# 加载模型
# ==========================
print("正在加载YOLO模型...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("✓ 模型加载成功")

DANGER_CLASS_MAP = {0: "person", 2: "car", 5: "bus", 7: "truck"}
INPUT_SIZE = 640
CONF_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45

# ==========================
# 辅助函数
# ==========================
def get_latest_audio_url():
    return request.host_url.rstrip("/") + "/latest_audio.wav"

def get_danger_audio_url(filename):
    return request.host_url.rstrip("/") + f"/audio/{filename}"

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

    boxes, confidences, class_ids = [], [], []

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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

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

def generate_latest_tts_file(text, voice="alloy", speed=1.0):
    """
    直接向 OpenAI 请求 WAV，避免 pcm -> 手动封装 wav 造成兼容性问题
    """
    try:
        print(f"[TTS] 开始生成语音，文本前60字: {text[:60]}")
        print(f"[TTS] voice={voice}, speed={speed}, format=wav")

        speech = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=speed,
            response_format="wav"
        )

        wav_bytes = speech.content if speech and speech.content else b""

        if not wav_bytes:
            print("[TTS] ❌ 未收到音频内容")
            return False

        with open(LATEST_AUDIO_PATH, "wb") as f:
            f.write(wav_bytes)

        if not os.path.exists(LATEST_AUDIO_PATH):
            print("[TTS] ❌ latest_audio.wav 未成功写入")
            return False

        file_size = os.path.getsize(LATEST_AUDIO_PATH)
        print(f"[TTS] ✅ latest_audio.wav 已生成，大小: {file_size} bytes")

        if file_size < 100:
            print("[TTS] ❌ 音频文件过小，疑似无效")
            return False

        return True

    except Exception as e:
        print(f"[TTS] 错误：{e}")
        return False

# ==========================
# HTTP 路由
# ==========================
@app.route("/latest_audio.wav", methods=["GET"])
def latest_audio():
    if not os.path.exists(LATEST_AUDIO_PATH):
        return jsonify({"error": "latest audio not found"}), 404

    file_size = os.path.getsize(LATEST_AUDIO_PATH)
    print(f"[AUDIO] 提供 latest_audio.wav, 大小: {file_size} bytes")

    response = make_response(
        send_file(
            LATEST_AUDIO_PATH,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="latest_audio.wav"
        )
    )

    # 禁缓存，避免 ESP32 或中间层拿到旧文件
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response

@app.route("/audio/<path:filename>", methods=["GET"])
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        img_bytes = request.get_data()

        if not img_bytes:
            return jsonify({"danger": False, "text": "", "audio_url": ""})

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

        audio_url = ""
        if danger and warning_class in DANGER_AUDIO_MAP:
            filename = DANGER_AUDIO_MAP[warning_class]
            audio_url = get_danger_audio_url(filename)

        return jsonify({
            "danger": danger,
            "text": warning_text,
            "audio_url": audio_url
        })

    except Exception as e:
        print(f"[detect] 错误：{str(e)}")
        return jsonify({
            "danger": False,
            "text": str(e),
            "audio_url": ""
        }), 500

@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    try:
        img_bytes = request.get_data()

        if not img_bytes:
            return jsonify({"text": "Kein Bild empfangen.", "audio_url": ""}), 400

        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Du bist ein hilfreicher Assistent für blinde Nutzer. Maximal 1-2 Sätze. Auf Deutsch."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Beschreibe bitte die aktuelle Szene kurz und hilfreich."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=150
        )

        text = response.choices[0].message.content.strip()
        print(f"[AI] 生成文本: {text}")

        audio_url = ""
        if text:
            ok = generate_latest_tts_file(text, voice="alloy", speed=1.0)
            if ok:
                audio_url = get_latest_audio_url()
                print(f"[AI] ✅ 音频地址: {audio_url}")
            else:
                print("[AI] ❌ TTS生成失败")
        else:
            print("[AI] ⚠️ 文本为空，跳过TTS")

        return jsonify({
            "text": text,
            "audio_url": audio_url
        })

    except Exception as e:
        print(f"[ask_ai] 错误：{str(e)}")
        return jsonify({
            "text": f"Error: {str(e)}",
            "audio_url": ""
        }), 500

@app.route("/test")
def test():
    return jsonify({"status": "server running"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": net is not None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
