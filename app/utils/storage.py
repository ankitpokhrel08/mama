import os
import uuid
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from app.config import config


def _ensure_dirs():
    os.makedirs(config["evidence"]["clips_dir"], exist_ok=True)
    os.makedirs(config["evidence"]["plates_dir"], exist_ok=True)


def save_plate_image(plate_crop: np.ndarray) -> str:
    _ensure_dirs()
    name = f"plate_{uuid.uuid4().hex}.jpg"
    path = os.path.join(config["evidence"]["plates_dir"], name)
    cv2.imwrite(path, plate_crop)
    return path


def save_screenshot(frame: np.ndarray) -> str:
    _ensure_dirs()
    name = f"screenshot_{uuid.uuid4().hex}.jpg"
    path = os.path.join(config["evidence"]["plates_dir"], name)
    cv2.imwrite(path, frame)
    return path


def save_clip(frame_buffer: deque, fps: float) -> str:
    _ensure_dirs()
    if not frame_buffer:
        return ""

    name = f"clip_{uuid.uuid4().hex}.mp4"
    path = os.path.join(config["evidence"]["clips_dir"], name)

    h, w = frame_buffer[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frame_buffer:
        writer.write(f)
    writer.release()
    return path
