
import os
import io
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from supabase import create_client
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# --- 从环境变量读取 Key（安全！）---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
# ---------------------------------

# 初始化客户端
client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 加载 YOLO 模型 (第一次运行会自动下载，大概 100MB，服务器启动会慢一点)
# yolov8n.pt 是最小最快的模型，适合免费服务器
model = YOLO('yolov8n.pt') 

# 定义危险关键词
DANGER_CLASSES = ['person', 'car', 'truck', 'bus'] 
# 注意：YOLO 预训练模型里没有 'zebra crossing'，毕设演示建议先用 car/person 代替

@app.route('/detect', methods=['POST'])
def detect_safety():
    """
    安全监护模式：ESP32 上传图片，服务器识别危险，返回警告语音
    """
    if 'file' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    
    # 1. YOLO 识别
    results = model(io.BytesIO(img_bytes))
    detections = []
    danger_found = False
    warning_msg = ""
    
    # 解析识别结果
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            if conf > 0.5: # 置信度大于 50% 才认为识别到了
                detections.append(class_name)
                if class_name in DANGER_CLASSES:
                    danger_found = True
                    # 简单逻辑：识别到车就说车，识别到人就说人
                    if class_name == 'car': warning_msg = "前方有车辆靠近，请注意安全"
                    elif class_name == 'person': warning_msg = "前方有人，请慢行"
    
    # 2. 如果有危险，生成语音
    audio_url = ""
    if danger_found:
        speech_response = client.audio.speech.create(model="tts-1", voice="alloy", input=warning_msg)
        # 将语音存入 Supabase (参考之前的上传逻辑，此处简化)
        # 实际需编写上传函数，这里假设返回一个固定链接或 base64 (为了代码简洁，暂略上传步骤，直接返回文本让 ESP32 读)
        # 毕设建议：这里调用之前的上传逻辑，返回 audio_url
        audio_url = "https://你的服务器.com/tts/warning.mp3" # 占位符
        
        # 3. 记录日志到 Supabase 数据库
        try:
            supabase.table("logs").insert({
                "type": "danger",
                "detected": detections,
                "message": warning_msg
            }).execute()
        except:
            pass # 数据库表没创建也不影响运行
            
    return jsonify({
        "status": "success",
        "danger": danger_found,
        "text": warning_msg if danger_found else "安全",
        "audio_url": audio_url
    })

@app.route('/ask_ai', methods=['POST'])
def ask_ai_vision():
    """
    AI 助手模式：用户语音提问 + 图片，GPT-4o 视觉分析
    """
    # 接收图片和音频 (简化为只接收图片，假设触发词已由 ESP32 判断)
    if 'file' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    
    # 1. 调用 GPT-4o 视觉模型
    # 需要将图片转为 base64 发送给 OpenAI
    import base64
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用简短的一句话描述这张图片里的环境，特别是是否有危险。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    )
    
    ai_text = response.choices[0].message.content
    
    # 2. 文字转语音
    speech_response = client.audio.speech.create(model="tts-1", voice="alloy", input=ai_text)
    # 同样，这里需要上传语音文件到存储桶并返回 URL
    # 为了演示，我们假设直接返回文本，ESP32 用离线 TTS 或者这里简化处理
    # 完善方案：上传到 Supabase Storage，返回 URL
    
    return jsonify({
        "status": "success",
        "text": ai_text,
        "audio_url": "https://你的服务器.com/tts/ai_reply.mp3"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)