# ============================================================================
# AI智能眼镜服务器 - HTTP版本（真实TTS + YOLO调试修正版）
# 功能：
# 1. /ask_ai 使用真实图像理解 + 真实 TTS 生成
# 2. TTS 直接请求 wav，speed=1.5
# 3. /latest_audio.wav 使用更适合 ESP32 的完整 Response 返回
# 4. /detect 保存最近原图、带框图、检测信息
# 5. /debug/view 调试网页可查看原图、带框图、检测信息
# 6. YOLO 输出解析兼容常见 ONNX 形状，避免 class_id 变成 8000+
# ============================================================================

import os
import cv2
import time
import base64
import numpy as np
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    Response,
    render_template_string
)
from openai import OpenAI

app = Flask(__name__)

# ==========================
# 调试用：保存最近一帧
# ==========================
latest_raw_frame = None
latest_annotated_frame = None
latest_debug_info = {
    "timestamp": "",
    "danger": False,
    "warning_text": "",
    "detections": []
}

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

os.makedirs(AUDIO_DIR, exist_ok=True)

DANGER_AUDIO_MAP = {
    "car": "car.wav",
    "bus": "bus.wav",
    "truck": "truck.wav",
    "person": "person.wav",
}

# ==========================
# COCO 类别名
# ==========================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

DANGER_CLASS_NAMES = {"person", "car", "bus", "truck"}

# ==========================
# 加载模型
# ==========================
print("正在加载YOLO模型...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("✓ 模型加载成功")

INPUT_SIZE = 640

# 先放宽阈值，优先验证“能不能看到人”
CONF_THRESHOLD = 0.20
SCORE_THRESHOLD = 0.20
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

def normalize_predictions(outputs):
    """
    兼容常见 ONNX 输出：
    - (1, N, 85)   YOLOv5 风格
    - (1, 85, N)   需要转置
    - (N, 85)
    - (85, N)
    - 84 维（YOLOv8 风格：4 box + 80 classes，无 obj_conf）
    """
    # OpenCV DNN 通常直接返回 ndarray，但保险起见兼容 list/tuple
    if isinstance(outputs, (list, tuple)):
        arr = outputs[0]
    else:
        arr = outputs

    arr = np.array(arr)
    print(f"[YOLO SHAPE] raw outputs.shape={arr.shape}")

    # 去掉 batch 维
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
        print(f"[YOLO SHAPE] removed batch -> {arr.shape}")

    # 现在希望变成 (N, C)
    if arr.ndim == 2:
        # 已经是 (N, 85) 或 (N, 84)
        if arr.shape[1] in (84, 85):
            preds = arr
        # 是 (85, N) 或 (84, N)
        elif arr.shape[0] in (84, 85):
            preds = arr.transpose(1, 0)
        else:
            raise ValueError(f"无法识别的二维输出形状: {arr.shape}")
    elif arr.ndim == 3:
        # 某些模型可能还是三维
        # 尝试最后一维是 84/85
        if arr.shape[-1] in (84, 85):
            preds = arr.reshape(-1, arr.shape[-1])
        # 尝试第二维是 84/85
        elif arr.shape[1] in (84, 85):
            preds = arr.transpose(0, 2, 1).reshape(-1, arr.shape[1])
        else:
            raise ValueError(f"无法识别的三维输出形状: {arr.shape}")
    else:
        raise ValueError(f"不支持的输出维度: {arr.shape}")

    print(f"[YOLO SHAPE] normalized predictions.shape={preds.shape}")
    return preds

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
    predictions = normalize_predictions(outputs)

    boxes, confidences, class_ids = [], [], []

    # 调试计数器
    raw_count = 0
    passed_conf_count = 0
    passed_score_count = 0
    passed_candidates = []

    pred_dim = predictions.shape[1]
    is_yolov5_style = (pred_dim == 85)  # 4 + obj + 80
    is_yolov8_style = (pred_dim == 84)  # 4 + 80，无 obj

    if not (is_yolov5_style or is_yolov8_style):
        print(f"[YOLO DEBUG] 非预期维度 pred_dim={pred_dim}")
        return []

    for pred in predictions:
        raw_count += 1

        if is_yolov5_style:
            obj_conf = float(pred[4])
            if obj_conf < CONF_THRESHOLD:
                continue
            passed_conf_count += 1

            class_scores = pred[5:]
            class_id = int(np.argmax(class_scores))
            class_score = float(class_scores[class_id])
            confidence = obj_conf * class_score

        else:
            # YOLOv8 风格：没有单独 obj_conf
            obj_conf = 1.0
            passed_conf_count += 1

            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            class_score = float(class_scores[class_id])
            confidence = class_score

        if confidence < SCORE_THRESHOLD:
            continue
        passed_score_count += 1

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

        class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f"class_{class_id}"

        passed_candidates.append({
            "class_id": class_id,
            "class_name": class_name,
            "class_score": round(class_score, 3),
            "confidence": round(confidence, 3),
            "box": [int(x), int(y), int(w), int(h)]
        })

    print(f"[YOLO DEBUG] raw={raw_count}, passed_conf={passed_conf_count}, passed_score={passed_score_count}, boxes={len(boxes)}")
    print(f"[YOLO DEBUG] passed_candidates={passed_candidates}")

    if len(boxes) == 0:
        print("[YOLO DEBUG] boxes为空")
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)
    after_nms = len(indices) if len(indices) > 0 else 0
    print(f"[YOLO DEBUG] after_nms={after_nms}")

    detections = []
    all_kept_after_nms = []

    if len(indices) > 0:
        for i in indices.flatten():
            class_id = class_ids[i]
            class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f"class_{class_id}"

            kept = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidences[i], 3),
                "box": boxes[i]
            }
            all_kept_after_nms.append(kept)

            if class_name in DANGER_CLASS_NAMES:
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidences[i],
                    "box": boxes[i]
                })

    print(f"[YOLO DEBUG] kept_after_nms={all_kept_after_nms}")
    print(f"[YOLO DEBUG] final_detections={detections}")

    return detections

def draw_detections(image, detections):
    vis = image.copy()

    for det in detections:
        x, y, w, h = det["box"]
        class_name = det["class_name"]
        conf = det["confidence"]

        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            vis,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return vis

def generate_latest_tts_file(text, voice="alloy", speed=1.5):
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

    try:
        with open(LATEST_AUDIO_PATH, "rb") as f:
            wav_data = f.read()

        file_size = len(wav_data)
        print(f"[AUDIO] 提供 latest_audio.wav, 大小: {file_size} bytes")

        response = Response(wav_data, mimetype="audio/wav")
        response.headers["Content-Length"] = str(file_size)
        response.headers["Content-Type"] = "audio/wav"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Connection"] = "close"
        response.headers["Accept-Ranges"] = "none"

        return response

    except Exception as e:
        print(f"[AUDIO] 提供 latest_audio.wav 失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<path:filename>", methods=["GET"])
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)

@app.route("/detect", methods=["POST"])
def detect():
    global latest_raw_frame, latest_annotated_frame, latest_debug_info

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

        # 保存原图
        latest_raw_frame = image.copy()

        img_height, img_width = image.shape[:2]
        detections = detect_objects(image)

        danger = False
        warning_text = ""

        debug_detections = []

        for det in detections:
            class_name = det["class_name"]
            box = det["box"]

            box_area = box[2] * box[3]
            box_center_y = box[1] + box[3] / 2
            area_ratio = box_area / (img_width * img_height)
            position_ratio = box_center_y / img_height

            debug_detections.append({
                "class_name": class_name,
                "confidence": round(det["confidence"], 3),
                "box": box,
                "area_ratio": round(area_ratio, 3),
                "position_ratio": round(position_ratio, 3)
            })

            # 先用更宽松的逻辑验证整条链
            if class_name == "person" and det["confidence"] > 0.35:
                danger = True
                warning_text = "Person vor Ihnen."
                break
            elif class_name == "car" and det["confidence"] > 0.35:
                danger = True
                warning_text = "Achtung, ein Auto vor Ihnen."
                break
            elif class_name == "bus" and det["confidence"] > 0.35:
                danger = True
                warning_text = "Achtung, ein Bus vor Ihnen."
                break
            elif class_name == "truck" and det["confidence"] > 0.35:
                danger = True
                warning_text = "Achtung, ein Lastwagen vor Ihnen."
                break

        # 保存带框图
        latest_annotated_frame = draw_detections(image, detections)

        # 保存调试信息
        latest_debug_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "danger": danger,
            "warning_text": warning_text,
            "detections": debug_detections
        }

        print(f"[DETECT] detections={debug_detections}")
        print(f"[DETECT] danger={danger}, warning_text={warning_text}")

        audio_url = ""

        # 关键：危险提示也改走 TTS，不再使用 /audio/car.wav 这类本地文件
        if danger and warning_text:
            ok = generate_latest_tts_file(warning_text, voice="alloy", speed=1.5)
            if ok:
                audio_url = get_latest_audio_url()
                print(f"[DETECT] ✅ 危险提示TTS已生成: {audio_url}")
            else:
                print("[DETECT] ❌ 危险提示TTS生成失败")

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

        print(f"[ASK_AI] 收到图片，大小: {len(img_bytes)} bytes")

        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Du bist ein hilfreicher Assistent für blinde Nutzer. Antworte auf Deutsch, kurz, klar und praktisch. Maximal 1 bis 2 Sätze."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Beschreibe bitte die aktuelle Szene kurz und hilfreich."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=120
        )

        text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message.content else ""
        print(f"[ASK_AI] 生成文本: {text}")

        audio_url = ""

        if text:
            ok = generate_latest_tts_file(text, voice="alloy", speed=1.5)
            if ok:
                audio_url = get_latest_audio_url()
                print(f"[ASK_AI] ✅ 返回真实TTS音频: {audio_url}")
            else:
                print("[ASK_AI] ❌ TTS生成失败")
        else:
            print("[ASK_AI] ⚠️ 文本为空，跳过TTS")

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

@app.route("/debug/raw.jpg")
def debug_raw_jpg():
    global latest_raw_frame

    if latest_raw_frame is None:
        return jsonify({"error": "no raw frame yet"}), 404

    ok, buf = cv2.imencode(".jpg", latest_raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if not ok:
        return jsonify({"error": "encode failed"}), 500

    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.route("/debug/annotated.jpg")
def debug_annotated_jpg():
    global latest_annotated_frame

    if latest_annotated_frame is None:
        return jsonify({"error": "no annotated frame yet"}), 404

    ok, buf = cv2.imencode(".jpg", latest_annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if not ok:
        return jsonify({"error": "encode failed"}), 500

    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.route("/debug/status")
def debug_status():
    return jsonify(latest_debug_info)

@app.route("/debug/view")
def debug_view():
    html = """
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AI Glasses Debug View</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f7f7f7; }
            h1 { margin-bottom: 10px; }
            .row { display: flex; gap: 20px; flex-wrap: wrap; }
            .card { background: white; padding: 16px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
            img { width: 480px; max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }
            pre { background: #111; color: #0f0; padding: 12px; border-radius: 8px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>AI Glasses Debug View</h1>
        <p>这个页面会显示最近一次 detect 请求的原图、带框图和检测信息。</p>

        <div class="row">
            <div class="card">
                <h3>原始画面</h3>
                <img id="raw" src="/debug/raw.jpg">
            </div>

            <div class="card">
                <h3>带框画面</h3>
                <img id="ann" src="/debug/annotated.jpg">
            </div>
        </div>

        <div class="card" style="margin-top:20px;">
            <h3>检测信息</h3>
            <pre id="status">loading...</pre>
        </div>

        <script>
            async function refreshDebug() {
                const t = Date.now();
                document.getElementById("raw").src = "/debug/raw.jpg?t=" + t;
                document.getElementById("ann").src = "/debug/annotated.jpg?t=" + t;

                try {
                    const res = await fetch("/debug/status?t=" + t);
                    const data = await res.json();
                    document.getElementById("status").textContent = JSON.stringify(data, null, 2);
                } catch (e) {
                    document.getElementById("status").textContent = "failed to load status";
                }
            }

            refreshDebug();
            setInterval(refreshDebug, 500);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route("/test")
def test():
    return jsonify({"status": "server running"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": net is not None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
