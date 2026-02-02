import cv2
import numpy as np
from emotion_model import EmotionModel
from utils import smooth_predictions

class Camera:
    def __init__(self, model_path, cascade_path):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.emotion_model = EmotionModel(model_path)
        self.emotion_labels = self.emotion_model.emotion_labels
        self.prediction_history = []

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                emotion, prob = self.emotion_model.predict(face_img)
                self.prediction_history.append(emotion)
                smoothed_emotion = smooth_predictions(self.prediction_history, window=7)
                label = f"{smoothed_emotion}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow('Face Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
