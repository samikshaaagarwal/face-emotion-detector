import numpy as np
from keras.models import load_model
import cv2

class EmotionModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def preprocess(self, face_img):
        # Resize to 48x48, normalize, expand dims for model
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def predict(self, face_img):
        processed = self.preprocess(face_img)
        preds = self.model.predict(processed, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        emotion = self.emotion_labels[emotion_idx]
        prob = preds[emotion_idx]
        return emotion, prob
