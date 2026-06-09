from collections import defaultdict, deque
from typing import Generator

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

from app.config import config, get_source_points, get_target_points
from app.core.plate import crop_plate_region, read_plate
from app.utils.storage import save_clip, save_plate_image, save_screenshot


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)


class SpeedProcessor:
    def __init__(self):
        self.model = YOLO("yolo11x.pt")
        self.tracker = ByteTrackTracker()

        cam = config["camera"]
        spd = config["speed"]

        self.speed_limit = spd["limit_kmh"]
        self.camera_id = cam["id"]
        self.location = cam["location"]
        self.process_nth = cam["process_every_nth_frame"]
        self.clip_duration = config["evidence"]["clip_duration_sec"]

        source = get_source_points()
        target = get_target_points()
        self.polygon_zone = sv.PolygonZone(polygon=source)
        self.view_transformer = ViewTransformer(source=source, target=target)

        self.conf_threshold = 0.3
        self.iou_threshold = 0.7

    def _setup_annotators(self, video_info: sv.VideoInfo):
        thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=int(video_info.fps * 2),
            position=sv.Position.BOTTOM_CENTER,
        )
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone,
            thickness=thickness,
        )

    def process(self, source_video: str, db_session=None) -> Generator:
        video_info = sv.VideoInfo.from_video_path(source_video)
        fps = video_info.fps
        self._setup_annotators(video_info)

        coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
        # rolling frame buffer per tracker for clip saving
        frame_buffers: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=int(fps * self.clip_duration))
        )
        # track which ids already triggered a violation (avoid duplicates)
        alerted: set[int] = set()

        frame_gen = sv.get_video_frames_generator(source_video)

        for frame_idx, frame in enumerate(frame_gen):
            # fill frame buffer for all active trackers before skip
            for buf in frame_buffers.values():
                buf.append(frame.copy())

            if frame_idx % self.process_nth != 0:
                yield frame
                continue

            result = self.model(
                frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[self.polygon_zone.trigger(detections)]
            detections = self.tracker.update(detections)

            if detections.tracker_id is None:
                yield frame
                continue

            points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            points = self.view_transformer.transform_points(points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for i, tracker_id in enumerate(detections.tracker_id):
                if len(coordinates[tracker_id]) < fps / 2:
                    labels.append(f"#{tracker_id}")
                    continue

                coord_start = coordinates[tracker_id][-1]
                coord_end = coordinates[tracker_id][0]
                distance = abs(coord_start - coord_end)
                elapsed = len(coordinates[tracker_id]) / fps
                speed = distance / elapsed * 3.6

                label = f"#{tracker_id} {int(speed)} km/h"

                if speed > self.speed_limit and tracker_id not in alerted:
                    alerted.add(tracker_id)
                    label = f"#{tracker_id} {int(speed)} km/h !"
                    self._handle_violation(
                        frame=frame,
                        detection=detections[i],
                        tracker_id=tracker_id,
                        speed=speed,
                        frame_buffer=frame_buffers[tracker_id],
                        fps=fps,
                        db_session=db_session,
                    )

                labels.append(label)

            annotated = frame.copy()
            annotated = self.zone_annotator.annotate(annotated)
            annotated = self.trace_annotator.annotate(annotated, detections)
            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels)
            yield annotated

    def _handle_violation(
        self,
        frame: np.ndarray,
        detection: sv.Detections,
        tracker_id: int,
        speed: float,
        frame_buffer: deque,
        fps: float,
        db_session,
    ):
        bbox = detection.xyxy[0]
        plate_crop = crop_plate_region(frame, bbox)
        plate_text = read_plate(plate_crop)

        plate_path = save_plate_image(plate_crop) if plate_crop.size > 0 else ""
        screenshot_path = save_screenshot(frame)
        clip_path = save_clip(frame_buffer, fps)

        print(
            f"[VIOLATION] tracker={tracker_id} speed={int(speed)}km/h "
            f"plate='{plate_text}' clip={clip_path}"
        )

        if db_session is not None:
            self._write_to_db(
                db_session=db_session,
                plate_text=plate_text,
                speed=speed,
                plate_path=plate_path,
                screenshot_path=screenshot_path,
                clip_path=clip_path,
            )

    def _write_to_db(self, db_session, plate_text, speed, plate_path, screenshot_path, clip_path):
        from app.models import Violation

        violation = Violation(
            plate_text=plate_text or None,
            speed_kmh=round(speed, 1),
            speed_limit_kmh=self.speed_limit,
            camera_id=self.camera_id,
            location=self.location,
            plate_image_path=plate_path or None,
            screenshot_path=screenshot_path or None,
            clip_path=clip_path or None,
            status="pending",
        )
        db_session.add(violation)
        db_session.commit()
