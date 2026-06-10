from collections import defaultdict, deque
from datetime import datetime
from typing import Generator

import cv2
import numpy as np
import supervision as sv
import torch
from trackers import ByteTrackTracker
from ultralytics import YOLO

from app.config import config, get_source_points, get_target_points
from app.core.plate import detect_and_read_plate
from app.utils.color import detect_vehicle_color
from app.utils.storage import append_to_csv, new_violation_id, save_clip, save_plate_image, save_vehicle_image

COLOR_NORMAL = sv.Color.from_hex("#00E676")   # bright green
COLOR_SPEEDER = sv.Color.from_hex("#FF1744")  # red
COLOR_ZONE = sv.Color.from_hex("#FFEA00")     # yellow


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
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.model = YOLO("yolo11x.pt")
        self.model.to(device)
        self.tracker = ByteTrackTracker()

        cam = config["camera"]
        spd = config["speed"]

        self.speed_limit = spd["limit_kmh"]
        self.camera_id = cam["id"]
        self.location = cam["location"]
        self.process_nth = cam["process_every_nth_frame"]
        self.process_width = cam.get("process_width")
        self.clip_duration = config["evidence"]["clip_duration_sec"]

        self.conf_threshold = 0.3
        self.iou_threshold = 0.7

    def _setup_annotators(self, resolution_wh: tuple, fps: float):
        thickness = sv.calculate_optimal_line_thickness(resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh)

        self.box_normal = sv.BoxAnnotator(color=COLOR_NORMAL, thickness=thickness)
        self.box_speeder = sv.BoxAnnotator(color=COLOR_SPEEDER, thickness=thickness + 1)
        self.label_normal = sv.LabelAnnotator(
            color=COLOR_NORMAL,
            text_color=sv.Color.BLACK,
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.label_speeder = sv.LabelAnnotator(
            color=COLOR_SPEEDER,
            text_color=sv.Color.WHITE,
            text_scale=text_scale * 1.2,
            text_thickness=thickness + 1,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLOR_NORMAL,
            thickness=thickness,
            trace_length=int(fps * 2),
            position=sv.Position.BOTTOM_CENTER,
        )
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone,
            color=COLOR_ZONE,
            thickness=thickness,
        )

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        bar_h = max(36, h // 22)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = bar_h / 42
        thick = max(1, bar_h // 20)
        pad = bar_h // 6
        y = bar_h - pad

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (pad, y), font, scale, (200, 200, 200), thick, cv2.LINE_AA)

        limit_text = f"LIMIT: {self.speed_limit} km/h"
        (lw, _), _ = cv2.getTextSize(limit_text, font, scale, thick)
        cv2.putText(frame, limit_text, (w - lw - pad, y), font, scale, (255, 220, 0), thick, cv2.LINE_AA)

        loc_text = f"{self.camera_id}  |  {self.location}"
        (locw, _), _ = cv2.getTextSize(loc_text, font, scale * 0.85, thick)
        cv2.putText(frame, loc_text, ((w - locw) // 2, y), font, scale * 0.85, (180, 180, 180), thick, cv2.LINE_AA)

        return frame

    def _annotate(self, frame: np.ndarray, detections: sv.Detections, labels: list, alerted: set) -> np.ndarray:
        annotated = frame.copy()
        annotated = self.zone_annotator.annotate(annotated)
        annotated = self.trace_annotator.annotate(annotated, detections)

        if detections.tracker_id is not None and len(detections) > 0:
            speeder_mask = np.array([tid in alerted for tid in detections.tracker_id])
            normal_mask = ~speeder_mask

            if normal_mask.any():
                annotated = self.box_normal.annotate(annotated, detections[normal_mask])
                annotated = self.label_normal.annotate(annotated, detections[normal_mask],
                                                       [l for l, m in zip(labels, normal_mask) if m])
            if speeder_mask.any():
                annotated = self.box_speeder.annotate(annotated, detections[speeder_mask])
                annotated = self.label_speeder.annotate(annotated, detections[speeder_mask],
                                                        [l for l, m in zip(labels, speeder_mask) if m])

        return self._draw_hud(annotated)

    def process(self, source_video: str, db_session=None) -> Generator:
        video_info = sv.VideoInfo.from_video_path(source_video)
        fps = video_info.fps
        orig_w, orig_h = video_info.resolution_wh

        if self.process_width and self.process_width < orig_w:
            scale = self.process_width / orig_w
            process_wh = (self.process_width, int(orig_h * scale))
        else:
            scale = 1.0
            process_wh = (orig_w, orig_h)

        scaled_source = get_source_points() * scale
        self.polygon_zone = sv.PolygonZone(polygon=scaled_source)
        self.view_transformer = ViewTransformer(source=scaled_source, target=get_target_points())

        self._setup_annotators(process_wh, fps)

        coordinates = defaultdict(lambda: deque(maxlen=int(fps)))
        frame_buffers: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=int(fps * self.clip_duration))
        )
        alerted: set[int] = set()

        # cached state for non-processed frames so annotations don't blink
        last_detections = sv.Detections.empty()
        last_labels: list = []

        frame_gen = sv.get_video_frames_generator(source_video)

        for frame_idx, orig_frame in enumerate(frame_gen):
            if scale != 1.0:
                frame = cv2.resize(orig_frame, process_wh)
            else:
                frame = orig_frame

            for buf in frame_buffers.values():
                buf.append(frame.copy())

            if frame_idx % self.process_nth != 0:
                yield self._annotate(frame, last_detections, last_labels, alerted)
                continue

            result = self.model(
                frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[self.polygon_zone.trigger(detections)]
            detections = self.tracker.update(detections)

            if detections.tracker_id is None:
                yield self._annotate(frame, last_detections, last_labels, alerted)
                continue

            valid_mask = detections.tracker_id >= 0
            detections = detections[valid_mask]
            for tid in detections.tracker_id:
                _ = frame_buffers[tid]

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

                label = f"#{tracker_id}  {int(speed)} km/h"

                if speed > self.speed_limit and tracker_id not in alerted:
                    alerted.add(tracker_id)
                    label = f"#{tracker_id}  {int(speed)} km/h  SPEEDING"
                    self._handle_violation(
                        frame=orig_frame,
                        bbox_scale=scale,
                        detection=detections[i],
                        tracker_id=tracker_id,
                        speed=speed,
                        frame_buffer=frame_buffers[tracker_id],
                        fps=fps,
                        db_session=db_session,
                    )

                labels.append(label)

            last_detections = detections
            last_labels = labels

            yield self._annotate(frame, detections, labels, alerted)

    def _handle_violation(self, frame, bbox_scale, detection, tracker_id, speed, frame_buffer, fps, db_session):
        # bbox is in scaled-down coordinates; invert scale to get original-resolution coords
        orig_bbox = (detection.xyxy[0] / bbox_scale) if bbox_scale != 1.0 else detection.xyxy[0]
        plate_text, plate_crop = detect_and_read_plate(frame, orig_bbox)

        # vehicle crop from original-res frame for color detection + image save
        ox1, oy1, ox2, oy2 = orig_bbox.astype(int)
        vehicle_crop = frame[max(0,oy1):oy2, max(0,ox1):ox2]
        vehicle_color = detect_vehicle_color(vehicle_crop)

        vid = new_violation_id()
        vehicle_path = save_vehicle_image(frame, orig_bbox, vid)
        plate_path = save_plate_image(plate_crop, vid)
        scaled_bbox = detection.xyxy[0]
        clip_path = save_clip(
            frame_buffer, fps, vid,
            bbox=scaled_bbox,
            speed=speed,
            plate_text=plate_text,
            speed_limit=self.speed_limit,
        )

        print(
            f"[VIOLATION] tracker={tracker_id} speed={int(speed)}km/h "
            f"plate='{plate_text}' color={vehicle_color} dir=evidence/{vid}/"
        )

        append_to_csv(
            violation_id=vid,
            camera_id=self.camera_id,
            location=self.location,
            speed_kmh=speed,
            speed_limit_kmh=self.speed_limit,
            plate_text=plate_text,
            vehicle_color=vehicle_color,
            vehicle_path=vehicle_path,
            plate_path=plate_path,
            clip_path=clip_path,
        )

        if db_session is not None:
            self._write_to_db(
                db_session=db_session,
                plate_text=plate_text,
                speed=speed,
                vehicle_color=vehicle_color,
                vehicle_path=vehicle_path,
                plate_path=plate_path,
                clip_path=clip_path,
            )

    def _write_to_db(self, db_session, plate_text, speed, vehicle_color, vehicle_path, plate_path, clip_path):
        from app.models import Violation

        violation = Violation(
            plate_text=plate_text or None,
            speed_kmh=round(speed, 1),
            speed_limit_kmh=self.speed_limit,
            camera_id=self.camera_id,
            location=self.location,
            vehicle_color=vehicle_color or None,
            vehicle_image_path=vehicle_path or None,
            plate_image_path=plate_path or None,
            clip_path=clip_path or None,
            status="pending",
        )
        db_session.add(violation)
        db_session.commit()
