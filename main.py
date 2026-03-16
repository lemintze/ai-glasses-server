import os
import base64
import uuid
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from supabase import create_client

app = Flask(__name__)

# ==========================
# 读取环境变量
# ==========================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 初始化客户端
client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ==========================
# 上传音频到 Supabase
# ==========================
def upload_audio(audio_content, filename):

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "audio/mpeg"
    }

    upload_url = f"{SUPABASE_URL}/storage/v1/object/ai-files/tts/{filename}"

    r = requests.put(upload_url, data=audio_content, headers=headers)

    if r.status_code == 200:
        return f"{SUPABASE_URL}/storage/v1/object/public/ai-files/tts/{filename}"

    return ""


# ==========================
# 接口1：安全识别
# ==========================
@app.route("/detect", methods=["POST"])
def detect():

    try:

        img_bytes = request.get_data()

        if not img_bytes:
            return jsonify({"error": "No image"}), 400

        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        # 使用 GPT4o Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请判断图片中是否存在危险（例如车辆靠近、行人、公交车等），只用一句话回答。"
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

        result = response.choices[0].message.content

        # 简单危险判断
        danger = False

        if "车" in result or "危险" in result or "行人" in result:
            danger = True

        audio_url = ""

        if danger:

            speech = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=result
            )

            filename = f"{uuid.uuid4()}.mp3"

            audio_url = upload_audio(speech.content, filename)

        return jsonify({
            "danger": danger,
            "text": result,
            "audio_url": audio_url
        })

    except Exception as e:

        return jsonify({
            "danger": False,
            "text": str(e),
            "audio_url": ""
        })


# ==========================
# 接口2：AI助手
# ==========================
@app.route("/ask_ai", methods=["POST"])
def ask_ai():

    try:

        img_bytes = request.get_data()

        if not img_bytes:
            return jsonify({"error": "No image"}), 400

        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "描述图片环境。"
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

        result = response.choices[0].message.content

        speech = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=result
        )

        filename = f"{uuid.uuid4()}.mp3"

        audio_url = upload_audio(speech.content, filename)

        return jsonify({
            "text": result,
            "audio_url": audio_url
        })

    except Exception as e:

        return jsonify({
            "text": str(e),
            "audio_url": ""
        })


# ==========================
# 测试接口
# ==========================
@app.route("/test")
def test():
    return jsonify({"status": "server running"})


# ==========================
# 启动
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
