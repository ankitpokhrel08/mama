# mama — Vehicle Speed Detection System

Automated roadside speed enforcement system built for Nepal traffic police. Detects speeding vehicles from a fixed camera, reads number plates, and stores evidence for police review.

## How it works

1. Video feed (MP4 or RTSP) is processed frame by frame
2. YOLOv11x detects vehicles inside a calibrated road zone
3. ByteTrack tracks each vehicle across frames
4. Speed is calculated using perspective-corrected coordinates between two virtual reference lines
5. When a vehicle exceeds the speed limit:
   - Number plate is detected (dedicated YOLOv8 plate detector) and read via PaddleOCR
   - Vehicle color is detected
   - Evidence is saved: vehicle image, plate crop, annotated 5s clip
   - Row is appended to `evidence/violations.csv`

## Stack

| Component | Library |
|---|---|
| Detection | YOLOv11x (Ultralytics) |
| Tracking | ByteTrack (via `trackers`) |
| Plate detection | YOLOv8 (Hugging Face: `Koushim/yolov8-license-plate-detection`) |
| OCR | PaddleOCR (English / Devanagari-ready) |
| Video / annotation | OpenCV, Supervision |
| Backend | FastAPI + SQLAlchemy + PostgreSQL |
| Deployment | MacBook local → RTSP production |

## Setup

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `DATABASE_URL`.

## Configuration

Edit `config.yaml`:

```yaml
camera:
  process_width: 1280      # resize 4K input to this before processing
  process_every_nth_frame: 3

speed:
  limit_kmh: 40
  source_points: [...]     # 4 pixel coords on the original frame (calibrate on install)
  target_width_m: 10.0     # real-world road width in meters
  target_length_m: 25.0    # real-world road length in meters
```

## Running

```bash
python scripts/run_processor.py data/vehicles.mp4 data/output.mp4
```

## Evidence structure

Each violation gets its own folder:

```
evidence/
  <uuid>/
    vehicle.jpg   — cropped vehicle at violation moment
    plate.jpg     — cropped number plate (if detected)
    clip.mp4      — annotated 5s clip with speed overlay
  violations.csv  — all violations with file:// links to evidence
```

## Database migration

If upgrading from an older schema:

```bash
python scripts/migrate.py
```

## Phase 2 roadmap

- RTSP live stream support (one-line swap in `config.yaml`)
- Multi-camera dashboard
- DOTM vehicle owner lookup
- SMS alerts to vehicle owner
- PDF ticket generation
- Fine-tuned OCR model for Nepali plates (needs 500+ labeled images)
