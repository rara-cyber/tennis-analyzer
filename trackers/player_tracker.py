"""Player detection and tracking using YOLO."""

import numpy as np
import cv2
import time
from typing import List, Dict
from ultralytics import YOLO


class PlayerTracker:
    """Detects and tracks tennis players across video frames."""

    def __init__(self, model_path: str = 'yolov8x.pt'):
        """
        Initialize player tracker with YOLO model.

        Args:
            model_path: Path to YOLO model weights (default uses pre-trained YOLOv8x)
        """
        self.model = YOLO(model_path)
        print(f"✓ Player detector ({model_path}) loaded")

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        Detect and track players in a single frame.

        Args:
            frame: Video frame (np.ndarray, shape (height, width, 3))

        Returns:
            Dictionary mapping player_id (int) to bounding box (list of 4 floats)
            Example: {1: [100.5, 200.3, 150.2, 300.8], 2: [400.1, 210.5, 450.3, 310.2]}
            Bounding box format: [x_min, y_min, x_max, y_max]
        """
        # Run YOLO tracking (persist=True maintains IDs across frames)
        results = self.model.track(frame, persist=True, verbose=False)

        # Extract detections
        player_dict = {}

        for box in results[0].boxes:
            # Get class ID and name
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            # Only process "person" class (ID 0 in COCO dataset)
            if class_name == 'person':
                # Get tracking ID
                if box.id is not None:
                    track_id = int(box.id[0])

                    # Get bounding box in xyxy format
                    bbox = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

                    player_dict[track_id] = bbox

        return player_dict

    def detect_frames(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: str = None
    ) -> List[dict]:
        """
        Detect and track players across all video frames.

        Args:
            frames: List of video frames
            read_from_stub: If True, load cached results instead of running detection
            stub_path: Path to cache file (e.g., 'tracker_stubs/player_detections.pkl')

        Returns:
            List of dictionaries (one per frame), each mapping player_id to bbox
        """
        # Import cache utilities
        from utils.video_utils import load_cache, save_cache, cache_exists
        from utils.video_utils import display_progress

        # Load from cache if requested
        if read_from_stub and stub_path and cache_exists(stub_path):
            print(f"Loading cached player detections from {stub_path}")
            return load_cache(stub_path)

        # Run detection on all frames
        print("Detecting players in all frames...")
        player_detections = []
        start_time = time.time()

        for i, frame in enumerate(frames):
            # Detect players in this frame
            detections = self.detect_frame(frame)
            player_detections.append(detections)

            # Display progress
            display_progress(i + 1, len(frames), "Progress", start_time)

        # Save to cache if path provided
        if stub_path:
            save_cache(player_detections, stub_path)

        return player_detections

    def choose_and_filter_players(
        self,
        court_keypoints: List[float],
        player_detections: List[dict]
    ) -> List[dict]:
        """
        Filter detections to only include the two competing players.

        Args:
            court_keypoints: List of 28 floats (14 x/y coordinate pairs)
            player_detections: List of detection dicts (may include >2 people per frame)

        Returns:
            Filtered list of detection dicts (exactly 2 players per frame)
        """
        # Get first frame with detections
        first_frame_detections = None
        for frame_detections in player_detections:
            if len(frame_detections) >= 2:
                first_frame_detections = frame_detections
                break

        if first_frame_detections is None:
            print("⚠ Warning: No frame with at least 2 players detected")
            return player_detections

        # Choose 2 players closest to any court keypoint
        chosen_players = self._choose_players(court_keypoints, first_frame_detections)

        # Filter all frames to only include chosen players
        filtered_detections = []
        for frame_detections in player_detections:
            filtered_dict = {
                player_id: bbox
                for player_id, bbox in frame_detections.items()
                if player_id in chosen_players
            }
            filtered_detections.append(filtered_dict)

        return filtered_detections

    def _choose_players(self, court_keypoints: List[float], player_dict: dict) -> List[int]:
        """
        Choose 2 players closest to court.

        Args:
            court_keypoints: List of 28 floats (14 x/y coordinate pairs)
            player_dict: Dictionary mapping player_id to bbox

        Returns:
            List of 2 player IDs closest to court
        """
        from utils.bbox_utils import get_foot_position, measure_distance

        distances = []

        for track_id, bbox in player_dict.items():
            # Get player foot position (center-x, bottom-y of bbox)
            player_foot = get_foot_position(bbox)

            # Calculate minimum distance to any court keypoint
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                keypoint_x = court_keypoints[i]
                keypoint_y = court_keypoints[i + 1]
                distance = measure_distance(player_foot, (keypoint_x, keypoint_y))
                min_distance = min(min_distance, distance)

            distances.append((track_id, min_distance))

        # Sort by distance and take 2 closest
        distances.sort(key=lambda x: x[1])
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players

    def draw_bounding_boxes(
        self,
        frames: List[np.ndarray],
        player_detections: List[dict]
    ) -> List[np.ndarray]:
        """
        Draw bounding boxes and player IDs on frames.

        Args:
            frames: List of video frames
            player_detections: List of detection dicts

        Returns:
            List of frames with bounding boxes drawn
        """
        output_frames = []

        for frame, player_dict in zip(frames, player_detections):
            # Create copy to avoid modifying original
            frame_copy = frame.copy()

            for track_id, bbox in player_dict.items():
                # Extract coordinates
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw rectangle (red color in BGR: 0, 0, 255)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw player ID label
                label = f"Player {track_id}"
                cv2.putText(
                    frame_copy,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            output_frames.append(frame_copy)

        return output_frames
