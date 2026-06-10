import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from paddleocr import PaddleOCR
from ultralytics import YOLO

_ocr = None
_plate_detector = None


def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr


def get_plate_detector() -> YOLO:
    global _plate_detector
    if _plate_detector is None:
        model_path = hf_hub_download(
            repo_id="Koushim/yolov8-license-plate-detection",
            filename="best.pt",
        )
        _plate_detector = YOLO(model_path)
    return _plate_detector


def _preprocess(img: np.ndarray) -> np.ndarray:
    target_w = 300
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img
    scale = target_w / w
    resized = cv2.resize(img, (target_w, max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def detect_and_read_plate(frame: np.ndarray, vehicle_bbox: np.ndarray) -> tuple[str, np.ndarray]:
    """Detect plate inside vehicle bbox, OCR it. Returns (plate_text, plate_crop)."""
    empty = ("", np.array([]))

    x1, y1, x2, y2 = vehicle_bbox.astype(int)
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)

    vehicle_crop = frame[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        return empty

    detector = get_plate_detector()
    results = detector(vehicle_crop, conf=0.3, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return empty

    best_idx = int(results.boxes.conf.argmax())
    px1, py1, px2, py2 = results.boxes.xyxy[best_idx].cpu().numpy().astype(int)
    plate_crop = vehicle_crop[py1:py2, px1:px2]

    if plate_crop.size == 0:
        return empty

    return _ocr_plate(plate_crop), plate_crop


def _ocr_plate(plate_crop: np.ndarray) -> str:
    plate_crop = _preprocess(plate_crop)
    ocr = get_ocr()
    results = ocr.predict(plate_crop)

    if not results:
        return ""

    best_text, best_conf = "", 0.0
    for res in results:
        for text, conf in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
            if conf > best_conf:
                best_conf = conf
                best_text = text

    return best_text.strip() if best_conf > 0.5 else ""
