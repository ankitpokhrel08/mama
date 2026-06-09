import numpy as np
from paddleocr import PaddleOCR

_ocr = None

def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        # use_angle_cls: handles rotated plates
        # lang: en covers roman script on Nepali plates; add 'devanagari' support via ch model
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr


def crop_plate_region(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Crop lower-center 30% of vehicle bbox as likely plate region."""
    x1, y1, x2, y2 = bbox.astype(int)
    h = y2 - y1
    w = x2 - x1

    # lower 30% height, center 60% width
    crop_y1 = y2 - int(h * 0.30)
    crop_x1 = x1 + int(w * 0.20)
    crop_x2 = x2 - int(w * 0.20)

    crop_y1 = max(0, crop_y1)
    crop_x1 = max(0, crop_x1)
    crop_x2 = min(frame.shape[1], crop_x2)
    crop_y2 = min(frame.shape[0], y2)

    return frame[crop_y1:crop_y2, crop_x1:crop_x2]


def read_plate(plate_crop: np.ndarray) -> str:
    """Run PaddleOCR on plate crop, return best text or empty string."""
    if plate_crop.size == 0:
        return ""

    ocr = get_ocr()
    result = ocr.ocr(plate_crop, cls=True)

    if not result or not result[0]:
        return ""

    # pick line with highest confidence
    best_text = ""
    best_conf = 0.0
    for line in result[0]:
        text, conf = line[1]
        if conf > best_conf:
            best_conf = conf
            best_text = text

    return best_text.strip() if best_conf > 0.5 else ""
