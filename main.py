import os
import cv2
from camera import Camera

# Paths
CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emotion_model.h5')

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Pre-trained model not found at {MODEL_PATH}.")
        print("Download a Keras .h5 model trained on FER2013 or similar and place it as 'emotion_model.h5' in the project directory.")
        return
    cam = Camera(model_path=MODEL_PATH, cascade_path=CASCADE_PATH)
    cam.run()

if __name__ == "__main__":
    main()
