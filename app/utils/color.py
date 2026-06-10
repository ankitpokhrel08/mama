import cv2
import numpy as np

# HSV ranges for common vehicle colors
_COLOR_RANGES = [
    ("red",    [(0, 70, 70),   (10, 255, 255)]),
    ("red",    [(170, 70, 70), (180, 255, 255)]),
    ("orange", [(11, 100, 100),(20, 255, 255)]),
    ("yellow", [(21, 100, 100),(35, 255, 255)]),
    ("green",  [(36, 60, 60),  (89, 255, 255)]),
    ("blue",   [(90, 60, 60),  (128, 255, 255)]),
    ("purple", [(129, 50, 50), (158, 255, 255)]),
    ("white",  [(0, 0, 200),   (180, 30, 255)]),
    ("black",  [(0, 0, 0),     (180, 255, 50)]),
    ("silver", [(0, 0, 120),   (180, 30, 200)]),
]


def detect_vehicle_color(vehicle_crop: np.ndarray) -> str:
    if vehicle_crop.size == 0:
        return "unknown"

    # ignore top 20% (sky/background) and bottom 10% (road/shadow)
    h = vehicle_crop.shape[0]
    body = vehicle_crop[int(h * 0.20):int(h * 0.90)]
    if body.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)

    counts: dict[str, int] = {}
    for color_name, (lo, hi) in _COLOR_RANGES:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        counts[color_name] = counts.get(color_name, 0) + int(mask.sum() // 255)

    total = sum(counts.values())
    if total == 0:
        return "unknown"

    best = max(counts, key=lambda c: counts[c])
    if counts[best] / total < 0.15:
        return "unknown"

    return best
