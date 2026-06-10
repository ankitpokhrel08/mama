import csv
import os
import uuid
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from app.config import config

EVIDENCE_DIR = "evidence"
CSV_PATH = os.path.join(EVIDENCE_DIR, "violations.csv")

_CSV_FIELDS = [
    "violation_id", "timestamp", "camera_id", "location",
    "speed_kmh", "speed_limit_kmh", "plate_text", "vehicle_color",
    "vehicle_image", "plate_image", "clip",
]


def new_violation_id() -> str:
    return uuid.uuid4().hex


def _to_file_url(path: str) -> str:
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    return f"file://{abs_path}"


def append_to_csv(
    violation_id: str,
    camera_id: str,
    location: str,
    speed_kmh: float,
    speed_limit_kmh: int,
    plate_text: str,
    vehicle_color: str,
    vehicle_path: str,
    plate_path: str,
    clip_path: str,
):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "violation_id": violation_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": camera_id,
            "location": location,
            "speed_kmh": round(speed_kmh, 1),
            "speed_limit_kmh": speed_limit_kmh,
            "plate_text": plate_text,
            "vehicle_color": vehicle_color,
            "vehicle_image": _to_file_url(vehicle_path),
            "plate_image": _to_file_url(plate_path),
            "clip": _to_file_url(clip_path),
        })


def violation_dir(violation_id: str) -> str:
    path = os.path.join(EVIDENCE_DIR, violation_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_vehicle_image(frame: np.ndarray, bbox: np.ndarray, violation_id: str) -> str:
    x1, y1, x2, y2 = bbox.astype(int)
    pad = 20
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad); y2 = min(frame.shape[0], y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    path = os.path.join(violation_dir(violation_id), "vehicle.jpg")
    cv2.imwrite(path, crop)
    return path


def save_plate_image(plate_crop: np.ndarray, violation_id: str) -> str:
    if plate_crop.size == 0:
        return ""
    path = os.path.join(violation_dir(violation_id), "plate.jpg")
    cv2.imwrite(path, plate_crop)
    return path


def save_clip(
    frame_buffer: deque,
    fps: float,
    violation_id: str,
    bbox: np.ndarray | None = None,
    speed: float = 0,
    plate_text: str = "",
    speed_limit: int = 0,
) -> str:
    if not frame_buffer:
        return ""

    path = os.path.join(violation_dir(violation_id), "clip.mp4")
    h, w = frame_buffer[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    font = cv2.FONT_HERSHEY_DUPLEX
    bar_h = max(36, h // 18)

    for f in frame_buffer:
        frame = f.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            spd_label = f"{int(speed)} km/h"
            (lw, lh), _ = cv2.getTextSize(spd_label, font, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 8, y1), (0, 0, 255), -1)
            cv2.putText(frame, spd_label, (x1 + 4, y1 - 4), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        scale = bar_h / 42
        thick = max(1, bar_h // 20)
        pad = bar_h // 6
        y = h - pad

        left = f"SPEEDING  {int(speed)} km/h  (limit {speed_limit} km/h)"
        cv2.putText(frame, left, (pad, y), font, scale, (0, 80, 255), thick, cv2.LINE_AA)

        if plate_text:
            right = f"Plate: {plate_text}"
            (rw, _), _ = cv2.getTextSize(right, font, scale, thick)
            cv2.putText(frame, right, (w - rw - pad, y), font, scale, (255, 220, 0), thick, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    return path
