import os
import io
import base64
import uuid
import requests
from flask import Flask, request, jsonify
from openai import OpenAI
from ultralytics import YOLO

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# 使用更省内存的模型
model = YOLO("yolov5n.pt")

DANGER_CLASSES = ["person", "car", "bus", "truck"]

# 上传音频
def upload_audio(audio_content):

    filename = f"{uuid.uuid4()}.mp3"

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
# 危险检测接口
# ==========================
@app.route("/detect", methods=["POST"])
def detect():

    img_bytes = request.get_data()

    if not img_bytes:
        return jsonify({"danger": False, "text": "", "audio_url": ""})

    results = model(io.BytesIO(img_bytes))

    danger = False
    warning_text = ""

    for r in results:

        for box in r.boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if conf > 0.5 and class_name in DANGER_CLASSES:

                danger = True

                if class_name == "car":
                    warning_text = "Achtung, ein Auto nähert sich."

                elif class_name == "bus":
                    warning_text = "Achtung, ein Bus kommt."

                elif class_name == "truck":
                    warning_text = "Achtung, ein Lastwagen nähert sich."

                elif class_name == "person":
                    warning_text = "Person vor Ihnen, bitte vorsichtig gehen."

                break

    audio_url = ""

    if danger:

        speech = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=warning_text
        )

        audio_url = upload_audio(speech.content)

    return jsonify({
        "danger": danger,
        "text": warning_text,
        "audio_url": audio_url
    })


# ==========================
# 视觉助手（按钮触发）
# ==========================
@app.route("/ask_ai", methods=["POST"])
def ask_ai():

    img_bytes = request.get_data()

    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Beschreibe kurz die Umgebung für eine blinde Person."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }]
    )

    text = response.choices[0].message.content

    speech = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    audio_url = upload_audio(speech.content)

    return jsonify({
        "text": text,
        "audio_url": audio_url
    })


@app.route("/test")
def test():
    return {"status": "server running"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
