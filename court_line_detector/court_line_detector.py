"""
Tennis Court Line Detector using CNN-based keypoint detection.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np


class CourtLineDetector:
    """Detects tennis court keypoints using CNN model."""

    def __init__(self, model_path: str):
        """
        Initialize court keypoint detector.

        Args:
            model_path: Path to trained model weights (e.g., 'models/court_keypoint_model.pth')
        """
        # Create ResNet50 model
        self.model = models.resnet50(pretrained=False)

        # Replace final fully connected layer
        # Original ResNet50 outputs 1000 classes
        # We need 28 outputs (14 keypoints × 2 coordinates)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 28)

        # Load trained weights
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )

        # Set to evaluation mode
        self.model.eval()

        # Define image preprocessing transforms
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # ResNet50 expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"✓ Court detector ({model_path}) loaded")

    def predict(self, frame: np.ndarray) -> list[float]:
        """
        Detect 14 court keypoints in a frame.

        Args:
            frame: Video frame (np.ndarray, shape (height, width, 3))

        Returns:
            List of 28 floats representing 14 (x, y) coordinate pairs
            Format: [x0, y0, x1, y1, x2, y2, ..., x13, y13]

        Note: Only processes first frame (assumes static camera)

        Keypoint indices:
            0: Top-left outer corner
            1: Top-right outer corner
            2: Top-left service box corner
            3: Top-right service box corner
            4: Left net post
            5: Right net post
            6: Bottom-left service box corner
            7: Bottom-right service box corner
            8: Bottom-left outer corner
            9: Bottom-right outer corner
            10: Top center (baseline)
            11: Bottom center (baseline)
            12: Left center (service line)
            13: Right center (service line)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get original dimensions for denormalization later
        original_height, original_width = frame.shape[:2]

        # Apply transforms
        image_tensor = self.transforms(frame_rgb)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Convert to numpy and remove batch dimension
        keypoints = outputs.squeeze().cpu().numpy()

        # Denormalize coordinates from 224x224 to original resolution
        # Model outputs are for 224x224 image, need to scale to original
        for i in range(0, len(keypoints), 2):
            # Scale x coordinate
            keypoints[i] = keypoints[i] * original_width / 224
            # Scale y coordinate
            keypoints[i + 1] = keypoints[i + 1] * original_height / 224

        print(f"✓ Detected 14 court keypoints")
        return keypoints.tolist()

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: list[float]
    ) -> np.ndarray:
        """
        Draw keypoints on frame for visualization.

        Args:
            frame: Video frame
            keypoints: List of 28 floats (14 x/y pairs)

        Returns:
            Frame with keypoints drawn as red circles with numbers
        """
        frame_copy = frame.copy()

        # Draw each keypoint
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            # Draw red circle
            cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)

            # Draw keypoint number
            keypoint_num = i // 2
            cv2.putText(
                frame_copy,
                str(keypoint_num),
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        return frame_copy

    def draw_keypoints_on_video(
        self,
        video_frames: list[np.ndarray],
        keypoints: list[float]
    ) -> list[np.ndarray]:
        """
        Draw keypoints on all video frames.

        Args:
            video_frames: List of video frames
            keypoints: List of 28 floats (14 x/y pairs)

        Returns:
            List of frames with keypoints drawn
        """
        output_frames = []
        for frame in video_frames:
            frame_with_keypoints = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame_with_keypoints)

        return output_frames
