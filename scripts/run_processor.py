"""Quick test runner — runs SpeedProcessor on a video, prints violations, saves output."""
import sys
import cv2
import supervision as sv

sys.path.insert(0, ".")
from app.core.processor import SpeedProcessor
from app.config import config

source = sys.argv[1] if len(sys.argv) > 1 else "data/vehicles.mp4"
output = sys.argv[2] if len(sys.argv) > 2 else "data/output_processed.mp4"

processor = SpeedProcessor()
video_info = sv.VideoInfo.from_video_path(source)

with sv.VideoSink(output, video_info) as sink:
    for frame in processor.process(source, db_session=None):
        sink.write_frame(frame)
        cv2.imshow("Speed Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
print(f"\nDone. Output: {output}")
print(f"Evidence saved to: {config['evidence']['clips_dir']} and {config['evidence']['plates_dir']}")
