from flask import Flask, request, jsonify
from emotion_model import EmotionModel
import numpy as np
import cv2
import base64

app = Flask(__name__)

model = EmotionModel("emotion_model.h5")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    image_data = data["image"]

    # Decode base64 image
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    emotion, conf = model.predict(frame)

    return jsonify({
        "emotion": emotion,
        "confidence": conf
    })

if __name__ == "__main__":
    app.run(debug=True)