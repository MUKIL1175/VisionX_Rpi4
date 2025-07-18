# ✅ Base: Python 3.9 slim for ARM64 (Raspberry Pi 4)
FROM arm64v8/python:3.9-slim

# Disable interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# 🧰 Install system-level dependencies
RUN apt update && apt install -y \
    ffmpeg espeak libespeak1 \
    libglib2.0-0 libgl1-mesa-glx libgl1 \
    libatlas-base-dev \
    build-essential cmake libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev libcanberra-gtk* \
    && apt clean && rm -rf /var/lib/apt/lists/*

# 🔁 Copy files
COPY requirements.txt .
COPY detect_tts.py .
COPY firebase_key.json .

# 🔧 Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 🔁 Pre-download YOLOv5 model (optional)
RUN python3 -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5n', trust_repo=True)"

# ✅ Entry point
CMD ["python3", "detect_tts.py"]
