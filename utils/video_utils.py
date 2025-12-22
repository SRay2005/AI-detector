import cv2
import tempfile
from utils.preprocessing import preprocess_image

def extract_frames(file, fps=2):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(file.file.read())
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    frames = []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps // fps)

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
