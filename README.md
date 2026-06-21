
🕶️ Sens — Smart Glasses for On-Demand Hazard Awareness

Sens is a human-centered assistive smart glasses system designed for visually impaired users.
It combines local real-time object detection with on-demand AI assistance, reducing cognitive overload while improving situational awareness.

Built around an ESP32 camera module + YOLO-based detection pipeline, the system only triggers AI feedback when explicitly requested by the user.


🌍 Overview

Unlike traditional assistive systems that constantly narrate the environment, Sens follows a minimal-interruption philosophy:

- Continuous lightweight local detection runs on-device
- Only critical hazards are tracked automatically
- Detailed AI feedback is activated via a user button
- Audio output is delivered in German via TTS


⚙️ Key Features

- 🎯 Real-time hazard detection (person, car, bus, truck)
- 📷 ESP32-CAM-based embedded vision pipeline
- 🧠 YOLO-based local inference (no external API calls during normal operation)
- 🔘 On-demand AI activation via physical button
- 🔊 German audio feedback (Supabase-hosted TTS)
- 🪶 Low-power, edge-first architecture
- 🧭 Designed for visually impaired mobility assistance


 🧠 System Architecture


Camera (ESP32)
   ↓
Local YOLO Detection
   ↓
Hazard Filtering Layer
   ↓
Event Trigger System
   ↓
Button Press Only
   ↓
OpenAI Vision API
   ↓
Text-to-Speech (Supabase TTS)
   ↓
Audio Output

<img width="1126" height="589" alt="Group 19" src="https://github.com/user-attachments/assets/fe61bf3c-ec71-425b-bf2d-db80f9bb7ac9" />


 🔧 Hardware Setup

- ESP32-CAM module
- Push button 
- Battery pack 
- Speaker
- 3D printed glasses frame

<img width="929" height="573" alt="Group 20" src="https://github.com/user-attachments/assets/4c632019-fd23-456d-b813-884d14f7ee2e" />


 💻 Software Stack

- Embedded: ESP32 (Arduino)
- Vision: YOLO (lightweight model)
- Backend: Python 
- AI: OpenAI Vision API (on-demand only)
- TTS: pyttsx3
- Communication: HTTP / REST


🚀 How It Works

1. Device continuously captures low-resolution frames
2. YOLO detects predefined hazard classes locally
3. System remains silent unless a **hazard event** or **user button press** occurs
4. On trigger:

   - Image is sent to AI
   - Scene is analyzed
   - audio response is generated and played back



 🧪 Design Philosophy

Sens is built around three constraints:

- **Cognitive minimalism** → no unnecessary notifications
- **Energy efficiency** → edge-first computation
- **User control** → AI speaks only when asked

It treats AI not as an always-on narrator, but as a **deliberate interaction layer**.


<img width="5459" height="3911" alt="Frame 31 (1)" src="https://github.com/user-attachments/assets/6e56b4ea-7280-48fa-966f-ad2ae1b48615" />

<img width="2812" height="2516" alt="Group 24" src="https://github.com/user-attachments/assets/a9173a94-4bc2-4681-a202-ef496bf0cdc6" />

