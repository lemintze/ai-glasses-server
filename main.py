# ============================================================================
# AI智能眼镜服务器 - Flask + WebSocket + YOLO + OpenAI
# 完整修复版 - 环境变量 + 图片格式 + 错误处理
# ============================================================================

import os
import cv2
import base64
import wave
import io
import numpy as np
import json
import uuid
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_sock import Sock
from openai import OpenAI
from collections import deque
import threading

# ============================================================================
# 配置部分
# ============================================================================

app = Flask(__name__)
sock = Sock(app)

# ==========================
# 环境变量读取（修复版）
# ==========================
print("=" * 70)
print("正在读取环境变量...")

# 方法1：从环境变量读取
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 调试输出
print(f"OPENAI_API_KEY: {'***' + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'NOT SET'}")
print(f"SUPABASE_URL: {SUPABASE_URL}")
print(f"SUPABASE_KEY: {'***' + SUPABASE_KEY[-4:] if SUPABASE_KEY else 'NOT SET'}")

# 验证必要的环境变量
if not OPENAI_API_KEY:
    print("❌ 错误：OPENAI_API_KEY 环境变量未设置！")
    print("   请在 Railway 后台 Variables 中添加 OPENAI_API_KEY")
    raise ValueError("OPENAI_API_KEY 环境变量未设置！")

print("✓ 环境变量读取成功")
print("=" * 70)

# 初始化 OpenAI 客户端
client = OpenAI(api_key=OPENAI_API_KEY)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov5n.onnx")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
LATEST_AUDIO_PATH = os.path.join(BASE_DIR, "latest_audio.wav")
TTS_CACHE_DIR = os.path.join(BASE_DIR, "tts_cache")

# 创建必要目录
for directory in [AUDIO_DIR, TTS_CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

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
print("正在加载YOLO模型...")
try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    print("✓ YOLO模型加载成功")
except Exception as e:
    print(f"✗ YOLO模型加载失败：{e}")
    net = None

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

# WebSocket连接管理
esp32_connections = {}
esp32_connections_lock = threading.Lock()
danger_cooldown = {}
cooldown_lock = threading.Lock()
DANGER_COOLDOWN_SECONDS = 3

# ============================================================================
# 辅助函数
# ============================================================================

def pcm_to_wav_bytes(pcm_bytes, sample_rate=16000, channels=1, sample_width=2):
    """PCM转WAV"""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_buffer.getvalue()


def generate_latest_tts_file(text, voice="aria", speed=1.5):
    """生成TTS音频"""
    try:
        print(f"[TTS] 正在生成语音：{text[:50]}...")
        speech = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=1.5,
            response_format="pcm"
        )

        pcm_bytes = speech.content if speech and speech.content else b""
        print(f"[TTS] PCM 字节数：{len(pcm_bytes)}")

        if not pcm_bytes:
            return False

        wav_bytes = pcm_to_wav_bytes(pcm_bytes)
        print(f"[TTS] WAV 字节数：{len(wav_bytes)}")

        with open(LATEST_AUDIO_PATH, "wb") as f:
            f.write(wav_bytes)

        print(f"[TTS] 音频已保存到：{LATEST_AUDIO_PATH}")
        return True
    except Exception as e:
        print(f"[TTS] 错误：{e}")
        return False


def get_latest_audio_url():
    """获取最新音频URL"""
    return request.host_url.rstrip("/") + "/latest_audio.wav"


def get_danger_audio_url(filename):
    """获取预录音频URL"""
    return request.host_url.rstrip("/") + f"/audio/{filename}"


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """图像预处理"""
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


def detect_objects(image):
    """YOLO检测"""
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


def check_danger_cooldown(danger_type):
    """检查危险冷却时间"""
    current_time = time.time()
    
    with cooldown_lock:
        if danger_type in danger_cooldown:
            if current_time - danger_cooldown[danger_type] < DANGER_COOLDOWN_SECONDS:
                return False
        danger_cooldown[danger_type] = current_time
        return True


def process_video_frame(data):
    """处理视频帧"""
    try:
        npimg = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"视频帧处理错误：{e}")
        return None


# ============================================================================
# WebSocket 路由
# ============================================================================

@sock.route('/ws')
def ws(websocket):
    """ESP32 WebSocket连接"""
    connection_id = str(uuid.uuid4())
    
    with esp32_connections_lock:
        esp32_connections[connection_id] = {
            'websocket': websocket,
            'connected_at': datetime.now(),
            'frame_count': 0
        }
    
    print(f"✓ ESP32已连接：{connection_id}")
    
    try:
        while True:
            data = websocket.receive()
            image = process_video_frame(data)
            
            if image is None:
                continue
            
            with esp32_connections_lock:
                if connection_id in esp32_connections:
                    esp32_connections[connection_id]['frame_count'] += 1
            
            detections = detect_objects(image)
            img_height, img_width = image.shape[:2]
            
            for det in detections:
                class_name = det["class_name"]
                box = det["box"]
                
                box_area = box[2] * box[3]
                box_center_y = box[1] + box[3] / 2
                area_ratio = box_area / (img_width * img_height)
                position_ratio = box_center_y / img_height
                
                if area_ratio > 0.08 and position_ratio > 0.5:
                    if check_danger_cooldown(class_name):
                        filename = DANGER_AUDIO_MAP.get(class_name, "attention.wav")
                        audio_url = get_danger_audio_url(filename)
                        
                        warning_message = json.dumps({
                            'type': 'DANGER',
                            'class': class_name,
                            'confidence': det['confidence'],
                            'audio_url': audio_url,
                            'text': f"Achtung, ein {class_name} nähert sich.",
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        websocket.send(warning_message)
                        print(f"⚠️ 危险警告：{class_name}")
            
    except Exception as e:
        print(f"✗ ESP32断开：{connection_id}, 错误：{e}")
    
    finally:
        with esp32_connections_lock:
            if connection_id in esp32_connections:
                del esp32_connections[connection_id]


# ============================================================================
# HTTP 路由
# ============================================================================

@app.route("/latest_audio.wav", methods=["GET"])
def latest_audio():
    """提供最新TTS音频"""
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
    """提供预录音频"""
    return send_from_directory(
        AUDIO_DIR,
        filename,
        as_attachment=False,
        max_age=0
    )


@app.route("/tts/<path:filename>", methods=["GET"])
def serve_tts(filename):
    """提供TTS缓存音频"""
    return send_from_directory(
        TTS_CACHE_DIR,
        filename,
        as_attachment=False,
        max_age=0
    )


@app.route("/detect", methods=["POST"])
def detect():
    """自动危险检测 - HTTP版本"""
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

        audio_url = ""
        if danger and warning_class in DANGER_AUDIO_MAP:
            filename = DANGER_AUDIO_MAP[warning_class]
            audio_url = get_danger_audio_url(filename)
            print(f"[detect] 使用预录音频：{filename}")

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
    """按钮触发AI场景描述"""
    try:
        img_bytes = request.get_data()
        if not img_bytes:
            return jsonify({
                "text": "Kein Bild empfangen.",
                "audio_url": ""
            }), 400

        print(f"[ask_ai] 收到图片数据：{len(img_bytes)} 字节")
        
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        print(f"[ask_ai] Base64长度：{len(base64_image)}")

        # ✅ 修复：添加 data: 前缀
        image_url = f"data:image/jpeg;base64,{base64_image}"
        print(f"[ask_ai] 图片URL前缀：{image_url[:50]}...")

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
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=150
        )

        text = response.choices[0].message.content.strip()
        print(f"[ask_ai] AI回复：{text}")
        
        audio_url = ""

        if text:
            ok = generate_latest_tts_file(
                text,
                voice="aria",
                speed=1.5
            )
            if ok:
                audio_url = get_latest_audio_url()
                print(f"[ask_ai] 音频URL：{audio_url}")

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
    """测试接口"""
    return jsonify({"status": "server running"})


@app.route("/health")
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": len(esp32_connections),
        "model_loaded": net is not None
    })


@app.route("/stats")
def stats():
    """服务器统计"""
    with esp32_connections_lock:
        connections_info = []
        for conn_id, conn_data in esp32_connections.items():
            connections_info.append({
                'id': conn_id,
                'connected_at': conn_data['connected_at'].isoformat(),
                'frame_count': conn_data['frame_count']
            })
    
    return jsonify({
        'active_connections': len(esp32_connections),
        'connections': connections_info,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print("AI智能眼镜服务器启动中...")
    print("=" * 70)
    print(f"  监听端口：{port}")
    print(f"  音频目录：{AUDIO_DIR}")
    print(f"  TTS缓存：{TTS_CACHE_DIR}")
    print("=" * 70)
    
    app.run(host="0.0.0.0", port=port, threaded=True)
