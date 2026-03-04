# Hand Gesture Recognition with Volume Control 🎥🤖

This project uses Deep Learning (CNN) to recognize hand gestures in real-time using a webcam and control system volume.

## Features
- Real-time hand gesture detection
- CNN model trained using TensorFlow
- Controls system volume using gestures
- OpenCV webcam integration

## Gestures
✋ Palm → Maximum Volume  
✊ Fist → Mute  
👍 Thumb → Medium Volume  

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Pycaw (Volume Control)

## How to Run
1. Install dependencies:
   python -m pip install tensorflow opencv-python numpy scikit-learn pycaw comtypes

2. Train model:
   python train.py

3. Run volume control:
   python volume_control.py