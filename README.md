# Face Emotion Detector

## Project Overview
Face Emotion Detector is a real-time computer vision application that detects human faces from a webcam feed and classifies their emotions using a pre-trained deep learning model. The detected emotion is displayed on the video feed with a bounding box around the face.

## Features
- Real-time face detection using OpenCV
- Emotion classification (Happy, Sad, Angry, Neutral, Surprise)
- Smooth, flicker-free emotion predictions
- Clean exit on key press
- Modular, production-quality Python code

## Installation
1. Clone this repository or download the source code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
```bash
python main.py
```

## How it Works
- The webcam feed is captured using OpenCV.
- Faces are detected using Haar cascades.
- Each detected face is passed to a pre-trained emotion recognition model.
- The predicted emotion is displayed on the video feed with a bounding box.
- Predictions are smoothed to avoid flickering.
- Press 'q' to exit the application cleanly.

## How to Fork and Contribute
1. Fork this repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to your fork.
4. Open a pull request describing your changes.

## Developer
Developer: tubakhxn
