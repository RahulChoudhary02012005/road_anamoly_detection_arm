# Road Anomaly Detection using YOLOv8 and TensorFlow Lite on Raspberry Pi 4

## Overview

This project implements a real-time Road Anomaly Detection system using a custom-trained YOLOv8 model. The system detects road anomalies such as potholes, cracks, and surface irregularities from video or camera input.

The trained PyTorch model (yolo26n.pt) is converted into an optimized TensorFlow Lite INT8 model for efficient deployment on Raspberry Pi 4 (8GB RAM). This enables fast, lightweight, and efficient edge AI inference suitable for real-world deployment.

---

## Features

- Custom trained YOLOv8 model for road anomaly detection
- TensorFlow Lite INT8 optimized model for edge deployment
- Raspberry Pi 4 deployment support
- Real-time video and camera inference
- Lightweight and efficient edge AI system
- Optimized for speed and low memory usage

---

## Model Information

Model Name: yolo26n.pt  
Model Architecture: YOLOv8 (Custom trained)  
Framework: PyTorch (Ultralytics)  
Converted Format: TensorFlow Lite (INT8)  
Deployment Device: Raspberry Pi 4 (8GB RAM)  

Input Resolution: 640x640  
Output: Bounding boxes with class labels and confidence scores  

---

## Project Structure

road-anomaly-detection/

yolo26n.pt  
yolov8n.pt  
convert_int8.py  
calibration_image_sample_data_20x128x128x3_float32.npy  
2.yaml  
README.md  
requirements.txt  

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/road-anomaly-detection.git

Navigate to the project folder:

cd road-anomaly-detection

Install dependencies:

pip install ultralytics tensorflow opencv-python numpy tflite-runtime onnx onnxruntime

---

## Model Training

Train the YOLOv8 model using:

yolo detect train data=2.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16

The trained model will be saved as:

yolo26n.pt

---

## Model Conversion to TensorFlow Lite

Convert the trained model to INT8 TensorFlow Lite format using:

python convert_int8.py

This generates an optimized TensorFlow Lite model for Raspberry Pi deployment.

---

## Laptop Inference

Run detection using the PyTorch model:

yolo predict model=yolo26n.pt source=video.mp4

---

## Raspberry Pi Inference

Install required libraries:

pip install tflite-runtime opencv-python numpy

Run inference:

python inference.py

Supports camera or video input.

---

## Optimization Techniques

INT8 quantization  
TensorFlow Lite conversion  
Lightweight YOLOv8 architecture  
Edge optimized inference  

These optimizations improve performance on Raspberry Pi.

---

## Hardware Used

Raspberry Pi 4 (8GB RAM)  
USB Camera or Pi Camera  
Laptop (for training and conversion)  
MicroSD Card  

---

## Software Used

Python 3.10  
YOLOv8 (Ultralytics)  
TensorFlow Lite  
OpenCV  
NumPy  

---

## Applications

Road condition monitoring  
Smart transportation systems  
Autonomous vehicle assistance  
Road safety monitoring  
Edge AI deployment  

---

## How It Works

1. Capture video from camera or file  
2. Extract frames using OpenCV  
3. Preprocess frames  
4. Run inference using YOLOv8 or TensorFlow Lite model  
5. Detect road anomalies  
6. Draw bounding boxes and labels  
7. Display or save output  

---

## Author

Rahul Choudhary  
Road Anomaly Detection Project  

---

## License

MIT License
