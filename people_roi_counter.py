#!/usr/bin/env python3
"""
People ROI Entry/Exit Counter
Uses YOLOv8 and ByteTrack to count people entering/exiting a gate region

Usage:
  python people_roi_counter.py --video input.mp4 
  
ROI formats:
  - Rectangle: "x1,y1,x2,y2" (percentages 0-1)
  - Rotated: --roi_rot "cx,cy,w,h,angle" (center, size in %, angle in degrees)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import argparse
import sys
from typing import Tuple


class PeopleROICounter:
    def __init__(self, model_name: str = 'yolov8m.pt'):
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.tracker = sv.ByteTrack()

        # Keep track of where each person has been
        self.track_history = defaultdict(list)

        # Track who's currently inside the gate area
        self.ids_currently_inside = set()
        
        # Remember where people were in the last frame to know which direction they're moving
        self.previous_positions = {}
        self.ids_counted_in_crossing = set()

        self.in_count = 0
        self.out_count = 0
        self.total_count = 0

        # colors that look nice on video (BGR format)
        self.color_detection_default = (150, 160, 180)
        self.color_detection_in = (120, 190, 160)
        self.color_detection_out = (140, 160, 220)
        self.color_roi = (170, 210, 200)
        self.color_text = (240, 240, 240)
        self.color_panel = (35, 40, 45)

        print("People ROI Counter initialized")

    def _point_in_rect(self, x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        """Check if a point is inside a rectangle"""
        rx1, ry1, rx2, ry2 = rect
        return (rx1 <= x <= rx2) and (ry1 <= y <= ry2)

    def _point_in_polygon(self, x: int, y: int, polygon: np.ndarray) -> bool:
        """Check if point is inside a polygon (for rotated rectangles)"""
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0

    def _update_entry_exit(self, track_id: int, center_x: int, center_y: int, roi_shape):
        """
        Main logic for counting entries and exits
        Looks at which direction the person is moving when they cross into the gate area
        """
        # Check if person is currently inside the gate
        if isinstance(roi_shape, tuple):
            is_inside = self._point_in_rect(center_x, center_y, roi_shape)
        else:
            is_inside = self._point_in_polygon(center_x, center_y, roi_shape)
        
        was_inside = track_id in self.ids_currently_inside
        direction = None
        
        # Figure out which way they're moving based on previous position
        if track_id in self.previous_positions:
            prev_x, prev_y = self.previous_positions[track_id]
            dy = center_y - prev_y  # positive = moving down, negative = moving up
            
            # Only count when they first enter the gate area
            if not was_inside and is_inside:
                # Make sure they're actually moving, not just standing there
                if abs(dy) > 3:
                    if dy > 0:  # Moving down (front to back) counts as IN
                        direction = 'in'
                        if track_id not in self.ids_counted_in_crossing:
                            self.in_count += 1
                            self.ids_counted_in_crossing.add(track_id)
                    elif dy < 0:  # Moving up (back to front) counts as OUT
                        direction = 'out'
                        if track_id not in self.ids_counted_in_crossing:
                            self.out_count += 1
                            self.ids_counted_in_crossing.add(track_id)

        # Update tracking state
        if is_inside:
            self.ids_currently_inside.add(track_id)
        else:
            self.ids_currently_inside.discard(track_id)
            # Reset their counting flag once they've completely left the area
            if track_id not in self.ids_currently_inside:
                self.ids_counted_in_crossing.discard(track_id)

        self.previous_positions[track_id] = (center_x, center_y)
        self.total_count = self.in_count - self.out_count
        return direction

    def _draw_statistics_panel(self, frame):
        """Draw the counter display in the top left"""
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, panel_height), self.color_panel, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"TOTAL (IN-OUT): {self.total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_text, 2)
        cv2.putText(frame, f"IN (front->back): {self.in_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_detection_in, 2)
        cv2.putText(frame, f"OUT (back->front): {self.out_count}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_detection_out, 2)

    def _draw_detection(self, frame, bbox, track_id, conf, class_name, crossing):
        """Draw bounding box and label for each detected person"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on what they're doing
        if crossing == 'in':
            color = self.color_detection_in
        elif crossing == 'out':
            color = self.color_detection_out
        else:
            color = self.color_detection_default

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name} ID:{track_id} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center dot
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        return center_x, center_y

    def _draw_track_history(self, frame, track_id, color):
        """Draw the trail showing where someone has walked"""
        if track_id in self.track_history:
            points = self.track_history[track_id]
            for j in range(1, len(points)):
                cv2.line(frame, points[j - 1], points[j], color, 2)

    def _parse_roi_percent(self, roi_percent: str, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert percentage ROI coordinates to actual pixel coordinates"""
        try:
            x1p, y1p, x2p, y2p = [float(v) for v in roi_percent.split(',')]
            x1 = int(np.clip(x1p, 0.0, 1.0) * width)
            y1 = int(np.clip(y1p, 0.0, 1.0) * height)
            x2 = int(np.clip(x2p, 0.0, 1.0) * width)
            y2 = int(np.clip(y2p, 0.0, 1.0) * height)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            return x1, y1, x2, y2
        except Exception:
            print("Invalid --roi format. Expected \"x1,y1,x2,y2\" with values between 0 and 1.")
            sys.exit(2)

    def _parse_roi_rot(self, roi_rot: str, width: int, height: int) -> np.ndarray:
        """Parse rotated rectangle ROI and return polygon points"""
        try:
            cxp, cyp, wp, hp, angled = [float(v) for v in roi_rot.split(',')]
        except Exception:
            print("Invalid --roi_rot format. Expected \"cx,cy,w,h,angle\" with percentages and angle in degrees.")
            sys.exit(2)
        
        cx = np.clip(cxp, 0.0, 1.0) * width
        cy = np.clip(cyp, 0.0, 1.0) * height
        w = np.clip(wp, 0.0, 1.0) * width
        h = np.clip(hp, 0.0, 1.0) * height
        rect = ((cx, cy), (max(1.0, w), max(1.0, h)), angled)
        box = cv2.boxPoints(rect)
        return np.int32(box)

    def process_video(self, video_source, output_path='output_people_roi.mp4',
                      display=True, roi_percent: str | None = None, roi_rot: str | None = None):
        """Main video processing loop"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            sys.exit(1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None

        # Set up the gate region - use custom ROI if provided, otherwise use default
        roi_shape = None
        if roi_rot is not None:
            roi_shape = self._parse_roi_rot(roi_rot, width, height)
        elif roi_percent is not None:
            roi_shape = self._parse_roi_percent(roi_percent, width, height)
        else:
            # Default gate position if nothing specified
            roi_shape = self._parse_roi_percent("0.20,0.65,0.75,0.70", width, height)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("\n" + "=" * 60)
        print("Starting people detection and ROI counting...")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        if total_frames:
            print(f"Total frames: {total_frames}")
        if isinstance(roi_shape, tuple):
            print(f"ROI rectangle (pixels): {roi_shape}")
        else:
            print(f"ROI polygon (pixels): {roi_shape.reshape(-1,2).tolist()}")
        print(f"Output will be saved to: {output_path}")
        if display:
            print("Press 'q' to quit")
        print("=" * 60 + "\n")

        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Detect people in the current frame
                results = self.model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Only keep person detections (class 0 in COCO dataset)
                if detections.class_id is not None:
                    keep_indices = [i for i, cid in enumerate(detections.class_id) if int(cid) == 0]
                    detections = detections[keep_indices]

                # Track people across frames
                detections = self.tracker.update_with_detections(detections)

                # Draw the gate region on the frame
                if isinstance(roi_shape, tuple):
                    rx1, ry1, rx2, ry2 = roi_shape
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), self.color_roi, 3)
                    cv2.putText(frame, "GATE ROI", (rx1, max(ry1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.color_roi, 1)
                    
                    # Draw red center line in the gate
                    cy = (ry1 + ry2) // 2
                    full_len = max(0, (rx2 - rx1) - 12)
                    half_scaled = int(0.8 * full_len / 2)
                    cx_mid = (rx1 + rx2) // 2
                    x_start = cx_mid - half_scaled
                    x_end = cx_mid + half_scaled
                    cv2.line(frame, (x_start, cy), (x_end, cy), (0, 0, 255), 2)
                else:
                    # Draw polygon gate
                    poly = roi_shape.reshape(-1, 2)
                    cv2.polylines(frame, [poly], isClosed=True, color=self.color_roi, thickness=3)
                    tx, ty = int(poly[0][0]), int(max(20, poly[0][1] - 10))
                    cv2.putText(frame, "GATE ROI", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.color_roi, 1)
                    
                    # Draw red line through center of rotated gate
                    edges = [
                        (poly[0], poly[1]),
                        (poly[1], poly[2]),
                        (poly[2], poly[3]),
                        (poly[3], poly[0])
                    ]
                    lengths = [np.hypot(e[0][0]-e[1][0], e[0][1]-e[1][1]) for e in edges]
                    idxs = np.argsort(lengths)[-2:]
                    m1 = np.array(((edges[idxs[0]][0][0] + edges[idxs[0]][1][0]) / 2.0,
                                    (edges[idxs[0]][0][1] + edges[idxs[0]][1][1]) / 2.0), dtype=np.float32)
                    m2 = np.array(((edges[idxs[1]][0][0] + edges[idxs[1]][1][0]) / 2.0,
                                    (edges[idxs[1]][0][1] + edges[idxs[1]][1][1]) / 2.0), dtype=np.float32)
                    c = (m1 + m2) / 2.0
                    v = (m2 - m1)
                    scaled_half = 0.25 * v
                    p1 = (int(c[0] - scaled_half[0]), int(c[1] - scaled_half[1]))
                    p2 = (int(c[0] + scaled_half[0]), int(c[1] + scaled_half[1]))
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)

                current_ids = set()
                if detections.tracker_id is not None:
                    for bbox, track_id, conf, class_id in zip(
                            detections.xyxy,
                            detections.tracker_id,
                            detections.confidence,
                            detections.class_id if detections.class_id is not None else [0] * len(detections.xyxy)):
                        
                        current_ids.add(track_id)
                        class_name = results.names[class_id] if (hasattr(results, 'names') and class_id in results.names) else 'person'

                        # Draw the person's bounding box
                        center_x, center_y = self._draw_detection(frame, bbox, track_id, conf, class_name, None)

                        # Check if they entered or exited the gate
                        event = self._update_entry_exit(track_id, center_x, center_y, roi_shape)
                        if event in ('in', 'out'):
                            # Redraw with appropriate color if they just entered/exited
                            self._draw_detection(frame, bbox, track_id, conf, class_name, event)

                        # Keep track of their path
                        self.track_history[track_id].append((center_x, center_y))
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id].pop(0)
                        self._draw_track_history(frame, track_id, self.color_detection_default)

                # Draw counter panel and save frame
                self._draw_statistics_panel(frame)
                out.write(frame)

                if display:
                    cv2.imshow('People ROI Counter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopping due to user input...")
                        break

                # Print progress update every 30 frames
                if total_frames and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames}) - "
                          f"IN: {self.in_count}, OUT: {self.out_count}, TOTAL: {self.total_count}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()
            print("\n" + "=" * 60)
            print("Processing complete!")
            print(f"  IN: {self.in_count}")
            print(f"  OUT: {self.out_count}")
            print(f"  TOTAL: {self.total_count}")
            print(f"Frames processed: {frame_count}")
            print(f"Output saved to: {output_path}")
            print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='People Detection with ROI Enter/Exit Counting (YOLOv8 + ByteTrack)')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_people_roi.mp4')
    parser.add_argument('--model', type=str, default='yolov8m.pt')
    parser.add_argument('--roi', type=str, default=None, help='Gate region as percentages: "x1,y1,x2,y2"')
    parser.add_argument('--roi_rot', type=str, default=None, help='Rotated gate: "cx,cy,w,h,angle"')
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()

    counter = PeopleROICounter(model_name=args.model)
    counter.process_video(video_source=args.video,
                          output_path=args.output,
                          display=not args.no_display,
                          roi_percent=args.roi,
                          roi_rot=args.roi_rot)


if __name__ == '__main__':

    main()

