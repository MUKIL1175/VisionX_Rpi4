import torch
import cv2
import platform
import pandas as pd
import subprocess
import time
import threading
import os
import queue
import edge_tts
import asyncio
import tempfile
from playsound import playsound
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# 🔧 Config
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 1.5
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_SKIP = 2  # Skip 1 out of every 2 frames for performance
speech_queue = queue.Queue()

# 🔐 Firebase setup
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://raspberryxx-acd06-default-rtdb.firebaseio.com/'
})
firebase_ref = db.reference('detections')

# 🖥️ Display check
def is_display_connected():
    if platform.system() == "Linux":
        try:
            out = subprocess.check_output("xrandr --listmonitors", shell=True).decode()
            return "Monitors" in out and "0:" in out
        except:
            return False
    return True

DISPLAY_ENABLED = is_display_connected()

# 🧠 Raspberry Pi check
def is_raspberry_pi():
    try:
        with open('/proc/cpuinfo') as f:
            return 'Raspberry Pi' in f.read()
    except:
        return False

# 🏁 Model selection
if is_raspberry_pi():
    print("🐍 Using yolov5n on Raspberry Pi for speed")
    model_name = 'yolov5n'
else:
    print("🖥️ Using yolov5l on PC for accuracy")
    model_name = 'yolov5l'

model_name = os.getenv("MODEL_SIZE", model_name)

# 🔍 Load model
print(f"Loading model: {model_name}")
model = torch.hub.load('ultralytics/yolov5', model_name, trust_repo=True)
model.conf = CONFIDENCE_THRESHOLD

# 📢 Async TTS setup
async def async_speak(text):
    try:
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
            await communicate.save(tmp.name)
            playsound(tmp.name)
    except Exception as e:
        print(f"[TTS Error] {e}")

def tts_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        text = speech_queue.get()
        loop.run_until_complete(async_speak(text))
        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    with speech_queue.mutex:
        speech_queue.queue.clear()
    speech_queue.put(text)

# 🧭 Direction helper
def get_direction(x_center, width):
    third = width // 3
    if x_center < third:
        return "on your left"
    elif x_center > 2 * third:
        return "on your right"
    else:
        return "in front of you"

# 🔐 Firebase logger
def log_detection(label, confidence, direction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_folder = datetime.now().strftime("%d%b%Y").lower()
    log = f"{label} : {confidence:.1f}% : {direction} : {timestamp}"
    print("[LOG]", log)
    try:
        firebase_ref.child(date_folder).push(log)
    except Exception as e:
        print("[Firebase Error]", e)

# 📷 Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("❌ Camera not available.")
    exit()

print("✅ Detection started. Press 'q' to exit.")
last_spoken_time = 0
last_sentence_spoken = ""
frame_count = 0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read failed.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        if not is_raspberry_pi():
            frame = cv2.flip(frame, 1)

        results = model(frame, size=FRAME_HEIGHT)
        df = results.pandas().xyxy[0]

        current_time = time.time()
        sentences = []

        # Prioritize non-person objects
        df = df.sort_values(by='name', key=lambda x: x == 'person')

        for _, row in df.iterrows():
            conf = float(row['confidence'])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            label = row['name']
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            x_center = (x1 + x2) // 2
            direction = get_direction(x_center, frame.shape[1])
            spoken_label = f"a {label}" if label.strip() else "an object"
            sentence = f"{spoken_label} {direction}"
            sentences.append(sentence)
            log_detection(label, conf * 100, direction)

            if DISPLAY_ENABLED:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({int(conf*100)}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if (current_time - last_spoken_time > COOLDOWN_SECONDS):
            if sentences:
                full_sentence = "I see " + ", ".join(sentences)
                if full_sentence != last_sentence_spoken:
                    speak(full_sentence)
                    last_spoken_time = current_time
                    last_sentence_spoken = full_sentence
            else:
                speak("I can't see anything")
                last_spoken_time = current_time
                last_sentence_spoken = ""

        if DISPLAY_ENABLED:
            cv2.imshow("YOLOv5 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.1)

# 🧹 Cleanup
cap.release()
if DISPLAY_ENABLED:
    cv2.destroyAllWindows()
