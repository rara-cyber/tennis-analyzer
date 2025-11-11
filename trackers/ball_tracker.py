"""Ball detection and tracking module for tennis videos."""

import cv2
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Optional
from ultralytics import YOLO


class BallTracker:
    """Detects and tracks tennis ball across video frames."""

    def __init__(self, model_path: str):
        """
        Initialize ball tracker with fine-tuned YOLO model.

        Args:
            model_path: Path to fine-tuned YOLO model (e.g., 'models/yolov5_tennis_ball.pt')
        """
        self.model = YOLO(model_path)
        self.use_generic = ('yolov8n.pt' in model_path or 'yolov5' not in model_path)

        if self.use_generic:
            print("⚠ Using generic YOLO model. Detection accuracy may be lower.")
            print("  Download fine-tuned model for better results.")
        else:
            print(f"✓ Ball detector ({model_path}) loaded")

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Detect ball in a single frame.

        Args:
            frame: Video frame (np.ndarray)

        Returns:
            Dictionary with single key-value pair: {1: [x_min, y_min, x_max, y_max]}
            Returns empty dict {} if no ball detected
        """
        # Run YOLO prediction (no tracking, just detection)
        # Lower confidence threshold because balls are small and fast
        results = self.model.predict(frame, conf=0.15, verbose=False)

        # Extract ball detection
        ball_dict = {}

        if len(results[0].boxes) > 0:
            # Take first detection (highest confidence)
            box = results[0].boxes[0]
            bbox = box.xyxy[0].tolist()

            # Always use ID 1 for ball (only one ball on court)
            ball_dict[1] = bbox

        return ball_dict

    def detect_frames(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None
    ) -> List[dict]:
        """
        Detect ball in all frames with caching.

        Args:
            frames: List of video frames
            read_from_stub: Load from cache if True
            stub_path: Path to cache file

        Returns:
            List of dictionaries (one per frame)
        """
        # Import cache utilities
        from utils.video_utils import load_cache, save_cache, cache_exists, display_progress

        # Load from cache if requested
        if read_from_stub and stub_path and cache_exists(stub_path):
            print(f"Loading cached ball detections from {stub_path}")
            return load_cache(stub_path)

        # Run detection on all frames
        print("Detecting ball in all frames...")
        ball_detections = []
        start_time = time.time()

        for i, frame in enumerate(frames):
            detections = self.detect_frame(frame)
            ball_detections.append(detections)

            # Display progress
            display_progress(i + 1, len(frames), "Progress", start_time)

        # Save to cache
        if stub_path:
            save_cache(ball_detections, stub_path)

        # Calculate detection rate
        frames_with_ball = sum(1 for d in ball_detections if len(d) > 0)
        detection_rate = frames_with_ball / len(ball_detections) * 100
        print(f"✓ Ball detected in {frames_with_ball}/{len(ball_detections)} frames ({detection_rate:.1f}%)")

        return ball_detections

    def interpolate_ball_positions(self, ball_detections: List[dict]) -> List[dict]:
        """
        Fill in missing ball detections using linear interpolation.

        Args:
            ball_detections: List of detection dicts (may have empty dicts for missing frames)

        Returns:
            List of detection dicts with gaps filled in
        """
        # Convert to list of bounding boxes (None for missing frames)
        ball_positions = [d.get(1, None) for d in ball_detections]

        # Create DataFrame
        df_ball_positions = pd.DataFrame(
            ball_positions,
            columns=['x1', 'y1', 'x2', 'y2']
        )

        # Interpolate missing values linearly
        df_ball_positions = df_ball_positions.interpolate()

        # Back-fill start (if first frames have no detection)
        df_ball_positions = df_ball_positions.bfill()

        # Convert back to list of dicts
        ball_detections_interpolated = [
            {1: row.tolist()}
            for _, row in df_ball_positions.iterrows()
        ]

        # Report interpolation results
        original_detections = sum(1 for d in ball_detections if len(d) > 0)
        final_detections = sum(1 for d in ball_detections_interpolated if len(d) > 0)
        print(f"✓ Interpolated {final_detections - original_detections} missing ball positions")
        print(f"✓ Final ball coverage: {final_detections}/{len(ball_detections)} frames ({final_detections/len(ball_detections)*100:.1f}%)")

        return ball_detections_interpolated

    def get_ball_shot_frames(self, ball_detections: List[dict]) -> List[int]:
        """
        Detect frames where player hit the ball (shot events).

        Args:
            ball_detections: List of ball detection dicts (with interpolation)

        Returns:
            List of frame numbers where shots occurred
            Example: [11, 58, 95, 131, 182]
        """
        # Convert to DataFrame for easier analysis
        ball_positions = [d.get(1, [0, 0, 0, 0]) for d in ball_detections]
        df_ball = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Calculate ball center y-position
        df_ball['mid_y'] = (df_ball['y1'] + df_ball['y2']) / 2

        # Smooth trajectory with rolling mean (reduces noise)
        df_ball['mid_y_rolling'] = df_ball['mid_y'].rolling(
            window=5,
            min_periods=1,
            center=False
        ).mean()

        # Calculate change in y-position (velocity)
        df_ball['delta_y'] = df_ball['mid_y_rolling'].diff()

        # Initialize shot detection
        shot_frames = []
        minimum_change_frames_for_hit = 25  # Ball must change direction for 25 frames

        for i in range(len(df_ball) - int(minimum_change_frames_for_hit * 1.2)):
            # Detect negative to positive change (ball going down, then up = hit)
            negative_position_change = df_ball['delta_y'].iloc[i] > 0 and df_ball['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball['delta_y'].iloc[i] < 0 and df_ball['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                # Count how many frames the change persists
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit)):
                    if change_frame >= len(df_ball):
                        break

                    negative_change = df_ball['delta_y'].iloc[change_frame] < 0 and negative_position_change
                    positive_change = df_ball['delta_y'].iloc[change_frame] > 0 and positive_position_change

                    if negative_change or positive_change:
                        change_count += 1

                # If change persists for minimum frames, it's a shot
                if change_count >= minimum_change_frames_for_hit - 1:
                    shot_frames.append(i)

        print(f"✓ Detected {len(shot_frames)} shots at frames: {shot_frames}")
        return shot_frames

    def draw_bounding_boxes(
        self,
        frames: List[np.ndarray],
        ball_detections: List[dict]
    ) -> List[np.ndarray]:
        """
        Draw ball bounding boxes on frames (yellow color).

        Args:
            frames: List of video frames
            ball_detections: List of detection dicts

        Returns:
            List of frames with ball bounding boxes drawn
        """
        output_frames = []

        for frame, ball_dict in zip(frames, ball_detections):
            frame_copy = frame.copy()

            if 1 in ball_dict:  # Ball detected in this frame
                bbox = ball_dict[1]
                x1, y1, x2, y2 = map(int, bbox)

                # Draw rectangle (yellow color in BGR: 0, 255, 255)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Draw label
                cv2.putText(
                    frame_copy,
                    "Ball",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

            output_frames.append(frame_copy)

        return output_frames
