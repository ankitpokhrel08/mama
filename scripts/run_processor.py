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

# compute actual output resolution (may differ from source if process_width is set)
orig_w, orig_h = video_info.resolution_wh
process_width = config["camera"].get("process_width", orig_w)
if process_width and process_width < orig_w:
    scale = process_width / orig_w
    out_w, out_h = process_width, int(orig_h * scale)
else:
    out_w, out_h = orig_w, orig_h

out_info = sv.VideoInfo(width=out_w, height=out_h, fps=video_info.fps, total_frames=video_info.total_frames)

with sv.VideoSink(output, out_info) as sink:
    for frame in processor.process(source, db_session=None):
        sink.write_frame(frame)
        cv2.imshow("Speed Cam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
print(f"\nDone. Output: {output}")
print(f"Evidence saved to: {config['evidence']['clips_dir']} and {config['evidence']['plates_dir']}")
