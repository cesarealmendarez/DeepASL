# DeepASL

DeepASL is pretty straightforward, it uses a live webcam feed to interpret American Sign Language hand gestures in real time.

**Installation:**

1. pip install -> opencv-python, mediapipe, numpy
2. run app.py

**Usage:**

After running, two windows will appear, the Analytics window displays extracted data points used in triggering the interpretation snapshot, and depth perception. Hand Segmentation shows what the CNN "sees"!

**Credits:**

1. MediaPipe(https://github.com/google/mediapipe) - Used to percieve shape and motion of hand, create skeleton-like outline of hands to segment useful classification features

2. Streamlit-WebRTC(https://github.com/whitphx/streamlit-webrtc) - Used to connect Python ML backend to Streamlit web hosting. At first I wanted to host this project with just Flask, but task proved to be rather difficult when trying to send webcam frames to Python backend. Streamlit-WebRTC solved this!

