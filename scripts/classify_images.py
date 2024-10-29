import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

model = MobileNetV2(weights="imagenet")

def classify_frame(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))  # Resize to match model input size
    image_preprocessed = preprocess_input(np.expand_dims(image_resized, axis=0))
    
    preds = model.predict(image_preprocessed)
    return decode_predictions(preds, top=1)[0][0]

if __name__ == "__main__":
    frames_dir = "data/frames"
    for frame in os.listdir(frames_dir):
        frame_path = os.path.join(frames_dir, frame)
        label = classify_frame(frame_path)
        print(f"{frame}: {label}")
