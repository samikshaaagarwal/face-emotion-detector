from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
from camera import Camera

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emotion_model.h5')

camera_instance = Camera(model_path=MODEL_PATH, cascade_path=CASCADE_PATH)

@app.get("/")
def home():
    return {"message": "Emotion API running"}

@app.post("/start")
def start_detection():
    camera_instance.run()
    return {"status": "Session ended"}

@app.get("/stats")
def get_stats():
    return camera_instance.get_session_stats()