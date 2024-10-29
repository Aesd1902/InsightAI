import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1
    print(f"Extracted {count} frames.")

if __name__ == "__main__":
    extract_frames("data/video.mp4", "data/frames", frame_rate=30)
