import threading
import time

import cv2
from deepface import DeepFace
from utils.log import logger

_tracking_thread: threading.Thread | None = None
_stop_event = threading.Event()

_face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def start_tracking(camera_index: int = 0) -> None:
    """Start face detection + expression recognition in a background thread."""
    global _tracking_thread, _stop_event

    _stop_event.clear()
    _tracking_thread = threading.Thread(
        target=_tracking_loop,
        args=(camera_index,),
        daemon=True,
    )
    _tracking_thread.start()


def _tracking_loop(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error(f"Failed to open camera at index {camera_index}")
        return

    last_emotion_time = 0.0
    emotion_interval = 1.5  # run DeepFace at most once per 1.5s (it's slow)

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]

        now = time.time()
        if now - last_emotion_time >= emotion_interval:
            last_emotion_time = now
            try:
                face_crop = frame[y:y + h, x:x + w]
                result = DeepFace.analyze(
                    face_crop,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                emotion = result[0]["dominant_emotion"]
                if emotion != "neutral":
                    logger.info(f"Expression: {emotion.upper()}")
            except Exception:
                pass

    cap.release()
