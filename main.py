import os
import io
import requests
import base64
import uuid
from flask import Flask, request, jsonify
from openai import OpenAI
from supabase import create_client
from ultralytics import YOLO

app = Flask(__name__)

# --- 从环境变量读取 Key（安全！）---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
# ---------------------------------

# 初始化客户端
client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 加载 YOLO 模型
model = YOLO('yolov8n.pt')

# 定义危险关键词
DANGER_CLASSES = ['person', 'car', 'truck', 'bus']

# ===========================
# 辅助函数：上传音频到 Supabase
# ===========================
def upload_audio_to_supabase(audio_content, filename):
    """上传音频文件到 Supabase Storage 并返回公开 URL"""
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "audio/mpeg"
        }
        upload_url = f"{SUPABASE_URL}/storage/v1/object/ai-files/tts/{filename}"
        resp = requests.put(upload_url, data=audio_content, headers=headers)
        
        if resp.status_code == 200:
            return f"{SUPABASE_URL}/storage/v1/object/public/ai-files/tts/{filename}"
        else:
            print(f"上传失败：{resp.text}")
            return ""
    except Exception as e:
        print(f"上传异常：{str(e)}")
        return ""


# ===========================
# 接口 1：安全识别
# ===========================
@app.route('/detect', methods=['POST'])
def detect_safety():
    # ✅ 兼容两种格式：form-data 或 原始图片
    if 'file' in request.files:
        file = request.files['file']
        img_bytes = file.read()
    else:
        # ESP32 发送的原始图片数据
        img_bytes = request.get_data()
    
    if not img_bytes:
        return jsonify({"error": "No image"}), 400
    
    # 1. YOLO 识别
    results = model(io.BytesIO(img_bytes))
    detections = []
    danger_found = False
    warning_msg = ""
    detected_obj = ""
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            if conf > 0.5:
                detections.append(class_name)
                detected_obj = class_name
                if class_name in DANGER_CLASSES:
                    danger_found = True
                    if class_name == 'car': 
                        warning_msg = "前方有车辆靠近，请注意安全"
                    elif class_name == 'person': 
                        warning_msg = "前方有人，请慢行"
                    elif class_name == 'truck': 
                        warning_msg = "前方有卡车，请注意安全"
                    elif class_name == 'bus': 
                        warning_msg = "前方有公交车，请注意安全"
    
    # 2. 生成 TTS 语音
    audio_url = ""
    if danger_found and warning_msg:
        speech_response = client.audio.speech.create(
            model="tts-1", 
            voice="alloy", 
            input=warning_msg
        )
        
        filename = f"{uuid.uuid4()}.mp3"
        audio_url = upload_audio_to_supabase(speech_response.content, filename)
        
        try:
            supabase.table("logs").insert({
                "type": "danger",
                "detected": detected_obj,
                "message": warning_msg
            }).execute()
        except Exception as e:
            print(f"日志记录失败：{str(e)}")
    
    return jsonify({
        "status": "success",
        "danger": danger_found,
        "text": warning_msg if danger_found else "安全",
        "audio_url": audio_url
    })

# ===========================
# 接口 2：AI 助手
# ===========================
@app.route('/ask_ai', methods=['POST'])
def ask_ai_vision():
    # ✅ 兼容两种格式：form-data 或 原始图片
    if 'file' in request.files:
        file = request.files['file']
        img_bytes = file.read()
    else:
        img_bytes = request.get_data()
    
    if not img_bytes:
        return jsonify({"error": "No image"}), 400
    
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用简短的一句话描述这张图片里的环境，特别是是否有危险。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        )
        
        ai_text = response.choices[0].message.content
        
        speech_response = client.audio.speech.create(
            model="tts-1", 
            voice="alloy", 
            input=ai_text
        )
        
        filename = f"{uuid.uuid4()}.mp3"
        audio_url = upload_audio_to_supabase(speech_response.content, filename)
        
        return jsonify({
            "status": "success",
            "text": ai_text,
            "audio_url": audio_url
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "text": str(e),
            "audio_url": ""
        }), 500

# ===========================
# 接口 3：测试接口
# ===========================
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "server is running!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
