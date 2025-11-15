import cv2
import numpy as np

def extract_frames(video_path, frame_rate=1):
    """
    Extract frames from video at 1 frame per second (default).
    Returns a list of frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(int(fps / frame_rate), 1)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    cap.release()
    return frames
