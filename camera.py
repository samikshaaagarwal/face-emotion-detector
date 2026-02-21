import cv2
import numpy as np
from emotion_model import EmotionModel
from utils import smooth_predictions

EMOTION_COLORS = {
    "Angry": (0, 0, 255),        # Red
    "Disgust": (0, 140, 255),    # Orange
    "Fear": (128, 0, 128),       # Purple
    "Happy": (0, 255, 0),        # Green
    "Sad": (255, 0, 0),          # Blue
    "Surprise": (0, 255, 255),   # Yellow
    "Neutral": (255, 255, 255)   # White
}

EMOTION_VALUES = {
    "Angry": -2,
    "Disgust": -2,
    "Fear": -1,
    "Sad": -1,
    "Neutral": 0,
    "Happy": 2,
    "Surprise": 1
}

class Camera:
    def __init__(self, model_path, cascade_path):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.emotion_model = EmotionModel(model_path)
        self.emotion_labels = self.emotion_model.emotion_labels
        self.prediction_history = []
        self.emotion_counts = {
            "Angry": 0,
            "Disgust": 0,
            "Fear": 0,
            "Happy": 0,
            "Sad": 0,
            "Surprise": 0,
            "Neutral": 0
        }
        self.emotion_signal = []
        self.session_data = []
        self.signal_length = 100

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
                emotion, conf = self.emotion_model.predict(face_img)
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                self.prediction_history.append(emotion)
                smoothed_emotion = smooth_predictions(self.prediction_history, window=7)
                self.session_data.append(smoothed_emotion)
                if smoothed_emotion in self.emotion_counts:
                    self.emotion_counts[smoothed_emotion] += 1
                value = EMOTION_VALUES.get(smoothed_emotion, 0)
                self.emotion_signal.append(value)
                if len(self.emotion_signal) > self.signal_length:
                    self.emotion_signal.pop(0)
                label = f"{smoothed_emotion} ({conf:.1f}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2)
                y_offset = 30
                for emotion, count in self.emotion_counts.items():
                    text = f"{emotion}: {count}"
                    cv2.putText(frame,
                                text,
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (200, 200, 200),
                                1)
                    y_offset += 20
            self.draw_emotion_graph(frame)
            cv2.imshow('Face Emotion Detector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.show_summary()
            if key == ord('q'):
                self.generate_report()
                break
        self.cap.release()
        cv2.destroyAllWindows()
    
    def show_summary(self):
        total = sum(self.emotion_counts.values())
        print("\n--- SESSION SUMMARY ---")

        if total == 0:
            print("No emotions detected.")
            return

        for emotion, count in self.emotion_counts.items():
            percentage = (count / total) * 100
            print(f"{emotion}: {percentage:.2f}%")

        most_frequent = max(self.emotion_counts, key=self.emotion_counts.get)
        print(f"\nDominant Emotion: {most_frequent}")
        print("-----------------------\n")
    
    def draw_emotion_graph(self, frame):
        height, width, _ = frame.shape

        graph_height = 120
        graph_width = width
        y_base = height - graph_height - 20

        # Draw background
        cv2.rectangle(frame,
                    (0, y_base),
                    (graph_width, y_base + graph_height),
                    (30, 30, 30),
                    -1)

        # Draw center line (neutral)
        center_y = y_base + graph_height // 2
        cv2.line(frame,
                (0, center_y),
                (graph_width, center_y),
                (100, 100, 100),
                1)

        if len(self.emotion_signal) < 2:
            return

        step = graph_width / self.signal_length

        for i in range(1, len(self.emotion_signal)):
            x1 = int((i - 1) * step)
            x2 = int(i * step)

            y1 = int(center_y - self.emotion_signal[i - 1] * 20)
            y2 = int(center_y - self.emotion_signal[i] * 20)

            cv2.line(frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2)
    
    def generate_report(self):
        if not self.session_data:
            print("No reactions recorded.")
            return

        from collections import Counter

        counter = Counter(self.session_data)
        total = len(self.session_data)

        print("\n--- PRODUCT REACTION REPORT ---")

        for emotion, count in counter.items():
            percent = (count / total) * 100
            print(f"{emotion}: {percent:.2f}%")

        dominant = counter.most_common(1)[0][0]
        print(f"\nMost Prominent Reaction: {dominant}")

        positive = counter.get("Happy", 0) + counter.get("Surprise", 0)
        engagement_score = (positive / total) * 100

        print(f"Engagement Score: {engagement_score:.2f}%")
        print("------------------------------\n")
        
        import csv

        with open("reaction_report.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Emotion", "Percentage"])

            for emotion, count in counter.items():
                percent = (count / total) * 100
                writer.writerow([emotion, percent])

            writer.writerow([])
            writer.writerow(["Most Prominent Reaction", dominant])
            writer.writerow(["Engagement Score", engagement_score])