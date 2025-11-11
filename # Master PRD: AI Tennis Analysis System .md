# Master PRD: AI Tennis Analysis System with Heatmap Visualization

## 1. Overview

**Product Name:** TennisVision Analytics Platform

**Description:** A comprehensive AI-powered tennis match analysis system that processes recorded tennis match videos to track players and balls, generate performance analytics, and visualize spatial patterns through heatmaps. The system provides professional-grade insights including shot speed, player movement, court coverage, and ball bounce patterns.

**Target Users:**
- Tennis coaches analyzing player performance
- Sports analysts studying match patterns
- Amateur players improving their game
- Sports technology enthusiasts

**Key Value Proposition:** Transforms raw tennis match footage into actionable insights through automated AI analysis, eliminating manual video review and providing data-driven coaching recommendations.

**Deployment Model:** 
- Development: Local environment (Windows/Mac/Linux)
- GPU Processing: Cloud-based via RunPod.io or Modal.com
- Future: Web application (Phase 2)
- Current Phase: Desktop application with video file input/output

## 2. Problem Statement

### Current Pain Points
1. **Manual Analysis is Time-Consuming:** Coaches spend 5-10 hours manually reviewing match footage to extract basic statistics
2. **Limited Objective Data:** Visual observation lacks precision for measuring speeds, distances, and positional patterns
3. **No Spatial Visualization:** Understanding court coverage and ball distribution requires tedious manual tracking
4. **Expensive Professional Tools:** Commercial sports analysis software costs $5,000-$50,000 annually

### User Impact
- Coaches cannot provide data-driven feedback to players efficiently
- Players lack objective metrics to identify weaknesses in court positioning
- Amateur analysts have no accessible tools for match analysis

### Business Justification
- Market gap: No affordable solution exists for semi-professional and amateur tennis analysis
- Total addressable market: 87 million tennis players worldwide (ITF 2023)
- Growing demand for sports analytics at grassroots level (42% YoY growth in sports tech)

## 3. Goals & Success Metrics

### Primary Goals
1. **Accurate Detection:** Achieve ≥95% ball detection rate and ≥98% player tracking accuracy across standard tennis match footage
2. **Comprehensive Analytics:** Generate shot speed, player speed, distance covered, and shot count metrics with <5% error margin
3. **Visual Insights:** Produce intuitive heatmaps showing ball bounce patterns and player positioning density
4. **Efficient Processing:** Process 1 minute of video in <2 minutes of compute time (30x speedup vs. manual analysis)
5. **Accessible Development:** Enable developers new to ML to set up and run the system within 4 hours

### Measurable KPIs

| Metric | Baseline | Target | Timeline |
|--------|----------|--------|----------|
| Ball detection recall | 75% (YOLO out-of-box) | ≥95% | Week 4 |
| Player tracking accuracy | 90% | ≥98% | Week 3 |
| Shot speed calculation error | N/A | <5% vs. ground truth | Week 5 |
| Video processing speed | N/A | <2min per match-minute | Week 6 |
| Heatmap generation time | N/A | <10 seconds for full match | Week 6 |
| Setup time for new developers | N/A | <4 hours with docs | Week 2 |
| Model training time (if needed) | N/A | <2 hours on cloud GPU | Week 4 |

### Timeline Milestones

**Phase 1: Foundation (Weeks 1-3)**
- Week 1: Video I/O pipeline + basic player detection
- Week 2: Court keypoint detection system
- Week 3: Player tracking with ID persistence

**Phase 2: Ball Analysis (Weeks 4-5)**
- Week 4: Ball detection model fine-tuning
- Week 5: Ball trajectory tracking and bounce detection

**Phase 3: Analytics & Visualization (Weeks 6-7)**
- Week 6: Speed/distance calculations + heatmap generation
- Week 7: Mini-court visualization + stats overlay

**Phase 4: Integration & Polish (Week 8)**
- Week 8: End-to-end testing, documentation, optimization

## 4. User Stories

### Core Workflow Stories

**US-001:** As a tennis coach, I want to upload a match video and receive automated player tracking so that I can analyze my student's court positioning without manual tagging.

**US-002:** As a player, I want to see a heatmap of where my opponent's balls landed so that I can identify patterns in their shot placement strategy.

**US-003:** As an analyst, I want to view shot speed metrics for each rally so that I can correlate shot power with winning points.

**US-004:** As a coach, I want to see where each player spends most of their time on court so that I can identify positioning weaknesses.

**US-005:** As a developer, I want to run the system on a sample video with a single command so that I can verify my installation is working correctly.

### Feature-Specific Stories

**US-006:** As a user, I want the system to automatically detect when a player hits the ball so that I don't have to manually mark shot frames.

**US-007:** As a coach, I want to export heatmaps as images so that I can include them in player performance reports.

**US-008:** As a developer, I want to use pre-trained models so that I don't need to collect training data or spend GPU hours on model training.

**US-009:** As a user, I want the system to handle videos from different camera angles so that I can analyze matches filmed with standard equipment.

**US-010:** As an analyst, I want to see player movement speed during rallies so that I can assess fitness and court coverage efficiency.

**US-011:** As a user, I want clear error messages when video quality is insufficient so that I understand why analysis failed.

**US-012:** As a developer, I want to process videos using cloud GPUs so that I don't need expensive local hardware.

## 5. Functional Requirements

### 5.1 Video Input Processing

**FR-001: Video File Ingestion**
The system accepts standard video file formats as input and extracts frames for processing.

**Acceptance Criteria:**
- Supports MP4, AVI, MOV video formats (H.264/H.265 codecs)
- Extracts frames at original frame rate (typically 24-30 fps)
- Validates video has minimum resolution of 1280x720 pixels
- Displays error message if video format is unsupported or corrupted
- Preserves frame timestamps for accurate speed calculations

**FR-002: Video Frame Buffering**
The system loads video frames into memory efficiently to enable batch processing.

**Acceptance Criteria:**
- Implements frame caching to avoid repeated disk reads
- Processes videos up to 60 minutes in length without memory overflow
- Releases memory after processing each video segment (5-minute chunks)
- Provides progress indicator showing frames processed / total frames

### 5.2 Player Detection & Tracking

**FR-003: Player Detection**
The system detects all humans visible in each frame using pre-trained YOLO models.

**Acceptance Criteria:**
- Uses YOLOv8x pre-trained on COCO dataset (person class)
- Detects players with confidence threshold ≥0.5
- Returns bounding box coordinates in (x_min, y_min, x_max, y_max) format
- Filters detections to only include persons on the tennis court (excludes audience, umpires)
- Processes each frame in <100ms on GPU

**FR-004: Player Identification**
The system identifies which detected persons are the two competing players versus other people.

**Acceptance Criteria:**
- Selects the 2 persons closest to any court keypoint in the first frame
- Assigns persistent IDs (Player 1, Player 2) based on initial court position
- Maintains player IDs for entire video duration (no ID swaps)
- Handles temporary occlusions (player not visible) by maintaining last known ID

**FR-005: Player Tracking**
The system tracks each player's position across all video frames with consistent identity.

**Acceptance Criteria:**
- Uses YOLO tracking mode with ByteTrack algorithm for ID persistence
- Maintains player ID accuracy ≥98% (≤2% ID switches per 100 frames)
- Tracks players even during fast movements or partial occlusions
- Stores player bounding box coordinates for every frame
- Handles edge cases: player leaving frame, players crossing paths

**FR-006: Player Position Extraction**
The system determines the foot position (ground contact point) of each player for spatial analysis.

**Acceptance Criteria:**
- Calculates foot position as (center_x, max_y) of bounding box
- Returns pixel coordinates relative to video frame dimensions
- Accounts for player height perspective (closer = lower in frame)
- Validates positions are within court boundaries (±10% tolerance)

### 5.3 Ball Detection & Tracking

**FR-007: Ball Detection Model**
The system detects tennis balls in each frame using a fine-tuned YOLO model optimized for small, fast-moving objects.

**Acceptance Criteria:**
- Uses YOLOv5 or YOLOv8 fine-tuned on tennis ball dataset (Roboflow or equivalent)
- Detects balls with confidence threshold ≥0.15 (lower than players due to size)
- Achieves ≥95% detection recall on test videos
- Processes each frame in <150ms on GPU
- Returns bounding box coordinates for ball center point

**FR-008: Ball Trajectory Interpolation**
The system fills in missing ball detections using interpolation to create smooth trajectories.

**Acceptance Criteria:**
- Detects gaps in ball detection timeline (missing frames)
- Interpolates ball position linearly for gaps ≤5 consecutive frames
- Uses rolling mean (window=5 frames) to smooth trajectory noise
- Does not interpolate gaps >5 frames (treats as separate ball events)
- Validates interpolated positions follow physically plausible paths

**FR-009: Ball Bounce Detection**
The system identifies frames where the ball contacts the ground based on trajectory changes.

**Acceptance Criteria:**
- Monitors ball y-coordinate (vertical position) for directional changes
- Detects bounce when y-velocity changes from negative to positive (ball descending then ascending)
- Requires velocity change sustained for ≥25 frames to confirm bounce (reduces false positives)
- Records bounce frame number and pixel coordinates (x, y)
- Stores bounce positions for heatmap generation

**FR-010: Ball Shot Detection**
The system identifies frames where a player hits the ball based on ball trajectory and player proximity.

**Acceptance Criteria:**
- Detects shot when ball y-velocity changes direction (indicates impact)
- Requires velocity change sustained for ≥25 frames
- Identifies which player shot the ball (closest player within 2 meters at shot frame)
- Records shot frame number, player ID, and ball speed
- Handles edge cases: serves, volleys, groundstrokes

### 5.4 Court Keypoint Detection

**FR-011: Court Keypoint Model**
The system detects 14 standard court keypoints (corners, service lines, net intersections) using a custom CNN.

**Acceptance Criteria:**
- Uses ResNet50 backbone with custom output layer (28 outputs: 14 x/y pairs)
- Processes first frame only (assumes static camera)
- Pre-trained model achieves <5 pixel average error on test images
- Returns 14 keypoint coordinates in pixel space: (x0,y0), (x1,y1)...(x13,y13)
- Keypoints represent: 4 outer corners, 4 service line intersections, 4 center line points, 2 net posts

**FR-012: Court Dimensions Mapping**
The system maps pixel coordinates to real-world court dimensions using known tennis court measurements.

**Acceptance Criteria:**
- Stores standard court dimensions as constants: singles width=8.23m, doubles width=10.97m, half-court height=11.88m
- Implements perspective transformation from pixel coordinates to metric coordinates
- Calculates pixel-to-meter conversion ratio using court width as reference
- Validates court dimensions detected match standard tennis court (±5% tolerance)
- Handles minor camera angle variations (up to 15° from perpendicular)

### 5.5 Coordinate Transformation System

**FR-013: Mini-Court Projection**
The system transforms player and ball positions from video pixel space to a standardized top-down mini-court coordinate system.

**Acceptance Criteria:**
- Creates 250x500 pixel mini-court representation with correct aspect ratio (1:2)
- Projects player foot positions onto mini-court using perspective transformation
- Projects ball positions onto mini-court using same transformation matrix
- Maintains proportional distances (e.g., player at baseline appears at mini-court baseline)
- Handles edge cases: players outside court boundaries (clips to court edges)

**FR-014: Player Height Reference Calibration**
The system uses known player heights to improve coordinate transformation accuracy.

**Acceptance Criteria:**
- Accepts player heights as input parameters (default: Player1=1.88m, Player2=1.91m)
- Calculates player height in pixels from bounding box dimensions
- Uses maximum observed bounding box height over 50-frame window (handles crouching)
- Applies player height as reference for pixel-to-meter conversion
- Recalculates conversion ratio if player height changes >10% (perspective shift)

### 5.6 Analytics Engine

**FR-015: Shot Speed Calculation**
The system calculates ball speed for each shot in kilometers per hour.

**Acceptance Criteria:**
- Measures distance traveled by ball between consecutive shot frames (using mini-court coordinates)
- Calculates time elapsed between shots using frame count and video frame rate
- Computes speed as: distance (meters) / time (seconds) × 3.6 (converts m/s to km/h)
- Filters unrealistic speeds (>200 km/h flagged as measurement errors)
- Stores speed for each shot with associated player ID and frame number

**FR-016: Player Movement Speed Calculation**
The system calculates player running speed during rallies in kilometers per hour.

**Acceptance Criteria:**
- Measures distance traveled by player between shot frames (opponent's shot to next shot)
- Uses mini-court coordinates to calculate Euclidean distance
- Calculates time elapsed between shots using frame count and video frame rate
- Computes speed as: distance (meters) / time (seconds) × 3.6
- Stores average speed, maximum speed, and total distance per rally

**FR-017: Distance Covered Tracking**
The system tracks total distance covered by each player throughout the match.

**Acceptance Criteria:**
- Accumulates frame-to-frame distance for each player across entire video
- Uses mini-court coordinates for accurate metric measurements
- Updates running total every frame (displayed in real-time during processing)
- Stores cumulative distance per player in meters
- Handles video cuts or scene changes gracefully (resets distance counter)

**FR-018: Shot Count Statistics**
The system counts total shots per player and tracks shot distribution over time.

**Acceptance Criteria:**
- Increments shot counter for identified player each time shot is detected
- Stores shot count separately for Player 1 and Player 2
- Calculates average shot speed per player (total speed / shot count)
- Tracks shots per rally (resets counter when ball goes out of play)
- Outputs final shot statistics in structured format (JSON/CSV)

### 5.7 Heatmap Generation

**FR-019: Ball Bounce Heatmap**
The system generates a 2D heatmap visualization showing spatial distribution of ball bounce locations on the court.

**Acceptance Criteria:**
- Creates heatmap using mini-court coordinate system (250x500 pixels)
- Uses Gaussian kernel density estimation with bandwidth=0.5 meters
- Color scheme: blue (low density) → green → yellow → red (high density)
- Overlays heatmap on mini-court outline with 50% transparency
- Normalizes heatmap intensity to match video duration (adjusts for short vs. long videos)
- Exports heatmap as PNG image (1000x2000 pixels for high resolution)

**FR-020: Player Position Heatmap**
The system generates separate heatmaps for each player showing where they spent time on court.

**Acceptance Criteria:**
- Creates one heatmap per player using mini-court coordinate system
- Samples player position every 5 frames to balance resolution and performance
- Uses Gaussian kernel density estimation with bandwidth=1.0 meters
- Color scheme: distinct colors per player (Player 1=red gradient, Player 2=blue gradient)
- Overlays both heatmaps on same mini-court with 40% transparency (shows overlap)
- Exports combined heatmap and individual player heatmaps as PNG images

**FR-021: Heatmap Statistical Summary**
The system generates quantitative metrics from heatmap data to complement visualizations.

**Acceptance Criteria:**
- Calculates court coverage percentage (percentage of court area with >10 position samples)
- Identifies most frequented zone for each player (divides court into 6 zones: left/center/right × baseline/mid-court)
- Calculates time spent in each zone (percentage of total match time)
- Outputs statistics as JSON file alongside heatmap images
- Validates statistics sum to 100% (time distribution) and are non-negative

### 5.8 Visualization & Rendering

**FR-022: Output Video Generation**
The system renders an annotated video showing all detections, tracking, and analytics overlays.

**Acceptance Criteria:**
- Draws bounding boxes on players (red rectangle, 2px width) with player ID labels
- Draws bounding boxes on ball (yellow rectangle, 2px width) with "Ball" label
- Draws court keypoints (red circles, 5px radius) with keypoint numbers
- Renders mini-court in top-right corner (250x500px) with 50px margins
- Plots player positions on mini-court as colored dots (Player 1=red, Player 2=blue)
- Plots ball position on mini-court as yellow dot
- Displays frame number in top-left corner (white text, 24pt font)
- Exports annotated video in MP4 format (H.264 codec, same resolution as input)

**FR-023: Statistics Overlay**
The system overlays real-time statistics on the output video during processing.

**Acceptance Criteria:**
- Creates semi-transparent black box (350x230px, 50% opacity) in bottom-right corner
- Displays current statistics: shot speed (last shot), player speeds, average speeds, shot counts
- Updates statistics every frame after shot detection
- Uses white text (16pt font) for readability on dark background
- Formats numbers to 1 decimal place (e.g., "45.3 km/h")
- Maintains statistics box visibility throughout entire video

**FR-024: Mini-Court Visualization**
The system renders a scaled-down tennis court overlay showing real-time positions.

**Acceptance Criteria:**
- Draws white rectangle (250x500px) representing court boundaries
- Draws black lines representing service lines, center line, and net (1px width)
- Adds 20px padding inside rectangle for visual spacing
- Positions mini-court in top-right corner with 50px margin from edges
- Maintains correct court aspect ratio (1:2 width:height)
- Updates player/ball positions on mini-court every frame

## 6. Non-Functional Requirements

### 6.1 Performance Requirements

**NFR-001: Video Processing Speed**
- Process 1 minute of input video in <2 minutes on cloud GPU (RunPod/Modal with T4 GPU)
- Process 1 minute of input video in <5 minutes on local CPU (Intel i7 or equivalent)
- Support videos up to 90 minutes (full tennis match) without performance degradation

**NFR-002: Model Inference Speed**
- YOLO player detection: <100ms per frame on GPU
- YOLO ball detection: <150ms per frame on GPU
- Court keypoint detection: <200ms for first frame (runs once)
- Total per-frame processing: <300ms on GPU (enables 3-4 fps throughput)

**NFR-003: Memory Efficiency**
- Peak memory usage <8GB RAM for 1080p videos
- Peak memory usage <16GB RAM for 4K videos
- Release memory after processing each 5-minute video segment
- GPU memory usage <6GB (fits T4 GPU on RunPod/Modal)

**NFR-004: Heatmap Generation Speed**
- Generate ball bounce heatmap in <5 seconds for 60-minute video
- Generate player position heatmaps in <10 seconds for 60-minute video
- Export heatmap images in <2 seconds (PNG format)

### 6.2 Accuracy & Quality Requirements

**NFR-005: Detection Accuracy**
- Ball detection recall ≥95% (detects ball when present in frame)
- Ball detection precision ≥90% (avoids false positives)
- Player detection accuracy ≥98% (correct bounding boxes)
- Player tracking ID persistence ≥98% (no ID swaps)

**NFR-006: Measurement Precision**
- Shot speed calculation error <5% (compared to professional Hawk-Eye systems)
- Player speed calculation error <10% (compared to GPS tracking)
- Distance measurements accurate to ±0.5 meters
- Court keypoint detection error <10 pixels average

**NFR-007: Robustness**
- Handles minor occlusions (player hidden <3 seconds) without losing tracking
- Tolerates camera shake or slight movement (±5% frame shift)
- Works with various lighting conditions (indoor/outdoor, shadows)
- Supports 720p to 4K video resolutions

### 6.3 Usability Requirements

**NFR-008: Setup Simplicity**
- Complete installation process in <30 minutes for developers new to ML
- Requires ≤5 command-line steps to set up environment
- Provides clear error messages for missing dependencies
- Includes automated test script to verify installation

**NFR-009: Execution Simplicity**
- Single command to process a video: `python main.py --input video.mp4`
- Automatically detects and uses GPU if available
- Displays progress bar during processing
- Outputs results to clearly named directory (e.g., `output_videos/`)

**NFR-010: Documentation Quality**
- README includes step-by-step installation guide with screenshots
- Provides example videos for testing
- Explains key concepts in beginner-friendly language (What is YOLO? What is a heatmap?)
- Includes troubleshooting section for common issues

### 6.4 Scalability Requirements

**NFR-011: Cloud GPU Integration**
- Supports RunPod.io PyTorch template with zero code changes
- Supports Modal.com serverless GPU functions with minimal configuration
- Provides environment configuration files for both platforms
- Falls back gracefully to CPU if GPU unavailable

**NFR-012: Batch Processing**
- Supports processing multiple videos in sequence
- Saves intermediate results (model outputs) to disk for debugging
- Provides option to skip already-processed videos (resume capability)

### 6.5 Maintainability Requirements

**NFR-013: Code Organization**
- Modular architecture: separate modules for detection, tracking, analytics, visualization
- Clear separation between model inference and business logic
- Follows PEP 8 Python style guidelines
- Uses type hints for function parameters and returns

**NFR-014: Testing**
- Includes unit tests for coordinate transformation functions
- Includes integration test using 30-second sample video
- Achieves ≥80% code coverage for core analytics functions
- Provides validation dataset with ground truth annotations

### 6.6 Security & Privacy Requirements

**NFR-015: Data Privacy**
- Processes videos locally (no upload to external servers except chosen cloud GPU)
- Does not store personal data or player identities
- Provides option to anonymize player IDs in output (e.g., "Player A" vs. "Player 1")
- Allows deletion of all intermediate files after processing

## 7. Technical Specifications

### 7.1 Technology Stack

**Recommended Stack (Original Project):**
- **Language:** Python 3.9+
- **Deep Learning Framework:** PyTorch 2.0+
- **Computer Vision:** OpenCV 4.8+
- **Object Detection:** Ultralytics YOLO (YOLOv8 for players, YOLOv5 for balls)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, OpenCV drawing functions
- **Cloud GPU:** RunPod.io or Modal.com (T4/A4000 GPUs)

**Alternative Stack Recommendations:**

| Component | Alternative Option | Pros | Cons |
|-----------|-------------------|------|------|
| Object Detection | **MMDetection** (OpenMMLab) | More model options, better docs | Steeper learning curve |
| Object Detection | **Detectron2** (Meta) | Production-ready, fast | More complex setup |
| Tracking | **DeepSORT** | Industry standard | Requires separate setup |
| Tracking | **StrongSORT** | Better accuracy | Slower than ByteTrack |
| Framework | **TensorFlow/Keras** | Larger community | Less PyTorch-native models |
| Heatmap Library | **Seaborn** | Beautiful defaults | Less control over styling |
| Heatmap Library | **Plotly** | Interactive heatmaps | Overkill for static images |

**Recommendation:** Stick with original stack (PyTorch + Ultralytics YOLO + OpenCV) because:
1. Ultralytics provides easiest API for beginners
2. Pre-trained YOLO models available for both tasks
3. Excellent documentation and community support
4. Native GPU support without extra configuration
5. Proven to work for this exact use case

**Only consider alternatives if:**
- You need real-time processing (use TensorRT optimization)
- You want interactive web-based heatmaps (use Plotly)
- You have specific model requirements not met by YOLO

### 7.2 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT LAYER                        │
│  [Video File] → [Frame Extractor] → [Frame Buffer]          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   DETECTION LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Player YOLO  │  │  Ball YOLO   │  │ Court Keypoint  │   │
│  │   (YOLOv8)   │  │  (YOLOv5)    │  │     (ResNet)    │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘   │
└─────────┼──────────────────┼───────────────────┼────────────┘
          │                  │                   │
┌─────────┴──────────────────┴───────────────────┴────────────┐
│                   TRACKING LAYER                             │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │Player Tracker│  │ Ball Tracker │                         │
│  │  (ByteTrack) │  │(Interpolate) │                         │
│  └──────┬───────┘  └──────┬───────┘                         │
└─────────┼──────────────────┼──────────────────────────────── ┘
          │                  │
┌─────────┴──────────────────┴──────────────────────────────── ┐
│              COORDINATE TRANSFORMATION LAYER                  │
│  [Pixel Coords] → [Perspective Transform] → [Mini-Court]     │
└────────────────────────┬──────────────────────────────────── ┘
                         │
┌────────────────────────┴──────────────────────────────────── ┐
│                    ANALYTICS ENGINE                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │Shot Detector│  │Speed Calc    │  │Distance Tracker │    │
│  └─────┬───────┘  └──────┬───────┘  └────────┬────────┘    │
└────────┼──────────────────┼───────────────────┼──────────── ┘
         │                  │                   │
┌────────┴──────────────────┴───────────────────┴──────────── ┐
│                  VISUALIZATION LAYER                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐     │
│  │Heatmap Gen   │  │Video Annotator│  │Stats Overlay │     │
│  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘     │
└─────────┼───────────────────┼──────────────────┼──────────── ┘
          │                   │                  │
┌─────────┴───────────────────┴──────────────────┴──────────── ┐
│                      OUTPUT LAYER                             │
│  [Annotated Video] + [Heatmap Images] + [Stats JSON]         │
└───────────────────────────────────────────────────────────── ┘
```

### 7.3 Data Models

#### VideoMetadata
```python
{
    "filename": str,              # "match_2024_01_15.mp4"
    "resolution": (int, int),     # (1920, 1080)
    "frame_rate": float,          # 30.0
    "total_frames": int,          # 54000
    "duration_seconds": float     # 1800.0
}
```

#### PlayerDetection
```python
{
    "frame_number": int,          # 0-54000
    "player_id": int,             # 1 or 2
    "bbox": [float, float, float, float],  # [x_min, y_min, x_max, y_max]
    "confidence": float,          # 0.0-1.0
    "foot_position": (float, float)  # (x, y) in pixels
}
```

#### BallDetection
```python
{
    "frame_number": int,
    "bbox": [float, float, float, float],
    "confidence": float,
    "center_position": (float, float),
    "interpolated": bool          # True if position was filled in
}
```

#### CourtKeypoints
```python
{
    "frame_number": int,          # Always 0 (first frame only)
    "keypoints": [                # 14 keypoints, each with (x, y)
        {"id": 0, "name": "top_left_outer", "position": (float, float)},
        {"id": 1, "name": "top_right_outer", "position": (float, float)},
        # ... 12 more keypoints
    ],
    "court_dimensions": {
        "width_meters": 10.97,    # Doubles court
        "height_meters": 23.77,
        "pixel_to_meter_ratio": float
    }
}
```

#### ShotEvent
```python
{
    "frame_number": int,
    "player_id": int,             # Which player hit the ball
    "ball_speed_kmh": float,      # Shot speed
    "ball_position": (float, float),  # Where ball was hit (mini-court coords)
    "player_position": (float, float) # Where player was standing
}
```

#### MatchStatistics
```python
{
    "player_1": {
        "total_shots": int,
        "average_shot_speed": float,
        "max_shot_speed": float,
        "total_distance_covered": float,
        "average_movement_speed": float,
        "court_coverage_percentage": float,
        "most_frequented_zone": str  # "baseline_left", "midcourt_center", etc.
    },
    "player_2": {
        # Same structure as player_1
    },
    "match_duration_seconds": float,
    "total_rallies": int
}
```

#### Heatmap
```python
{
    "type": str,                  # "ball_bounce" or "player_position"
    "player_id": int,             # null for ball_bounce, 1/2 for player
    "data": np.ndarray,           # 2D array of density values (250x500)
    "coordinate_system": "mini_court",
    "normalization": "match_duration",
    "color_scheme": str           # "blue_red", "viridis", etc.
}
```

### 7.4 Pre-trained Models

#### Model Sources and Setup

**Player Detection Model:**
- **Model:** YOLOv8x (extra-large variant)
- **Source:** Ultralytics pre-trained on COCO dataset
- **Download:** Automatic via Ultralytics library
- **File:** `yolov8x.pt` (~131 MB)
- **Setup:** `model = YOLO('yolov8x.pt')`

**Ball Detection Model:**
- **Model:** YOLOv5l fine-tuned on tennis balls
- **Source:** Roboflow Tennis Ball Detection dataset
- **Download:** Manual download from Roboflow or use provided model
- **File:** `yolov5_tennis_ball.pt` (~48 MB)
- **Alternative:** Use YOLOv8n fine-tuned on sports balls (faster inference)
- **Setup:** `model = YOLO('yolov5_tennis_ball.pt')`

**Court Keypoint Model:**
- **Model:** ResNet50 with custom head (14 keypoints × 2 coordinates = 28 outputs)
- **Source:** GitHub repository: `yastrebksv/tennis-court-detection`
- **Download:** Manual download from GitHub releases
- **File:** `court_keypoint_model.pth` (~95 MB)
- **Setup:** Load PyTorch state dict into custom ResNet50 architecture

#### Model Download Instructions (README section)

```bash
# 1. Player detection (automatic)
# No action needed - downloads on first use

# 2. Ball detection
# Option A: Download from provided link
wget https://your-storage.com/models/yolov5_tennis_ball.pt -P models/

# Option B: Train your own (optional, see Training Guide)
python training/train_ball_detector.py

# 3. Court keypoint detection
wget https://github.com/yastrebksv/tennis-court-detection/releases/download/v1.0/court_keypoint_model.pth -P models/
```

**Total Model Size:** ~275 MB (fits on GitHub LFS, requires separate download step)

### 7.5 File Structure

```
tennis-analysis/
├── main.py                          # Entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # Setup and usage guide
├── .env.example                     # Cloud GPU configuration template
│
├── models/                          # Pre-trained model weights
│   ├── yolov8x.pt                   # Auto-downloaded
│   ├── yolov5_tennis_ball.pt        # Manual download
│   └── court_keypoint_model.pth     # Manual download
│
├── input_videos/                    # Place input videos here
│   ├── sample_match.mp4             # Example video (not in repo)
│   └── .gitkeep
│
├── output_videos/                   # Annotated videos
│   └── .gitkeep
│
├── output_heatmaps/                 # Generated heatmap images
│   └── .gitkeep
│
├── output_stats/                    # JSON statistics files
│   └── .gitkeep
│
├── tracker_stubs/                   # Cached detection results (optional)
│   └── .gitkeep
│
├── trackers/                        # Detection and tracking modules
│   ├── __init__.py
│   ├── player_tracker.py            # Player detection + tracking
│   └── ball_tracker.py              # Ball detection + interpolation
│
├── court_line_detector/             # Court keypoint detection
│   ├── __init__.py
│   └── court_line_detector.py
│
├── mini_court/                      # Coordinate transformation
│   ├── __init__.py
│   └── mini_court.py
│
├── analytics/                       # Speed, distance, shot detection
│   ├── __init__.py
│   ├── shot_detector.py
│   ├── speed_calculator.py
│   └── distance_tracker.py
│
├── heatmaps/                        # Heatmap generation
│   ├── __init__.py
│   ├── ball_bounce_heatmap.py
│   └── player_position_heatmap.py
│
├── utils/                           # Helper functions
│   ├── __init__.py
│   ├── video_utils.py               # Read/write videos
│   ├── bbox_utils.py                # Bounding box operations
│   ├── conversions.py               # Pixel ↔ meter conversions
│   └── visualization.py             # Drawing functions
│
├── constants/                       # Configuration values
│   └── __init__.py                  # Court dimensions, player heights
│
├── tests/                           # Unit and integration tests
│   ├── test_coordinate_transform.py
│   ├── test_shot_detection.py
│   └── integration_test.py
│
└── docs/                            # Additional documentation
    ├── SETUP_RUNPOD.md              # RunPod.io setup guide
    ├── SETUP_MODAL.md               # Modal.com setup guide
    ├── CONCEPTS.md                  # ML concepts explained
    └── TROUBLESHOOTING.md           # Common issues and solutions
```

### 7.6 Cloud GPU Integration

#### RunPod.io Configuration

**Template:** PyTorch 2.0 (CUDA 11.8)
**GPU:** NVIDIA T4 or A4000
**Estimated Cost:** $0.20-0.40 per hour

**Setup Steps:**
1. Create RunPod account
2. Deploy PyTorch template with 50GB storage
3. Upload code via Jupyter or SSH
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python main.py --input /workspace/videos/match.mp4`

**Environment Variables:**
```bash
CUDA_VISIBLE_DEVICES=0          # Use first GPU
TORCH_HOME=/workspace/models    # Cache models here
```

#### Modal.com Configuration

**Function:** Serverless GPU function
**GPU:** T4 (4GB VRAM)
**Estimated Cost:** $0.0006 per second (~$1.08 for 30-min video)

**Setup Steps:**
1. Install Modal: `pip install modal-client`
2. Authenticate: `modal token new`
3. Define function in `modal_inference.py`:

```python
import modal

stub = modal.Stub("tennis-analysis")

image = modal.Image.debian_slim().pip_install([
    "torch", "ultralytics", "opencv-python", "pandas", "numpy"
])

@stub.function(
    image=image,
    gpu="T4",
    timeout=3600,
    mounts=[modal.Mount.from_local_dir("./models", remote_path="/models")]
)
def process_video(video_path: str) -> dict:
    from main import analyze_video
    return analyze_video(video_path)
```

4. Run: `modal run modal_inference.py::process_video --video-path="match.mp4"`

**Advantages of Modal:**
- No server management (fully serverless)
- Pay only for execution time
- Automatic scaling for batch jobs
- Cold start <30 seconds

**Disadvantages:**
- Less interactive than RunPod
- Requires function-based code structure
- Debugging is more challenging

**Recommendation:** 
- Use **RunPod** for development (Jupyter notebooks, easy debugging)
- Use **Modal** for production batch processing (cost-efficient, scalable)

## 8. UI/UX Requirements

### 8.1 Command-Line Interface

**Primary Interface (Phase 1):**
The system operates via command-line with the following usage:

```bash
python main.py --input <video_path> [options]

Required Arguments:
  --input PATH              Path to input video file

Optional Arguments:
  --output PATH             Output directory (default: ./output_videos)
  --heatmaps PATH           Heatmap output directory (default: ./output_heatmaps)
  --stats PATH              Stats output directory (default: ./output_stats)
  --use-cache               Use cached detections if available
  --skip-video              Generate heatmaps only (no annotated video)
  --player1-height FLOAT    Player 1 height in meters (default: 1.88)
  --player2-height FLOAT    Player 2 height in meters (default: 1.91)
  --device {cpu,cuda}       Force CPU or GPU (default: auto-detect)
  --verbose                 Show detailed processing logs
```

**Example Commands:**
```bash
# Basic usage
python main.py --input input_videos/match.mp4

# With custom player heights
python main.py --input match.mp4 --player1-height 1.85 --player2-height 1.92

# Generate only heatmaps (fast)
python main.py --input match.mp4 --skip-video

# Use cached detections for fast re-processing
python main.py --input match.mp4 --use-cache
```

### 8.2 Progress Indicators

**Console Output Format:**
```
TennisVision Analytics v1.0
============================
Input: input_videos/match.mp4
Duration: 30:00 (1800 seconds, 43,200 frames)
Resolution: 1920x1080 @ 24 fps

[1/7] Loading models...
  ✓ Player detector (YOLOv8x) loaded
  ✓ Ball detector (YOLOv5l) loaded
  ✓ Court detector (ResNet50) loaded

[2/7] Detecting court keypoints...
  ✓ 14 keypoints detected in first frame
  ✓ Court dimensions: 10.97m × 23.77m

[3/7] Detecting players...
  Progress: [████████████████████████████] 43200/43200 (100%) | ETA: 00:00
  ✓ Player 1 detected in 42,987 frames (99.5%)
  ✓ Player 2 detected in 42,854 frames (99.2%)

[4/7] Detecting ball...
  Progress: [████████████████████████████] 43200/43200 (100%) | ETA: 00:00
  ✓ Ball detected in 38,450 frames (89.0%)
  ✓ Interpolated 3,120 missing frames
  ✓ Final ball coverage: 41,570 frames (96.2%)

[5/7] Analyzing match...
  ✓ Detected 156 shots
  ✓ Player 1: 78 shots, avg speed 42.3 km/h
  ✓ Player 2: 78 shots, avg speed 45.1 km/h
  ✓ Total distance: Player 1 = 2.4 km, Player 2 = 2.7 km

[6/7] Generating heatmaps...
  ✓ Ball bounce heatmap created
  ✓ Player 1 position heatmap created
  ✓ Player 2 position heatmap created

[7/7] Rendering output video...
  Progress: [████████████████████████████] 43200/43200 (100%) | ETA: 00:00
  ✓ Video saved: output_videos/match_annotated.mp4

Processing complete! (Total time: 8m 32s)
Output files:
  - output_videos/match_annotated.mp4
  - output_heatmaps/ball_bounce_heatmap.png
  - output_heatmaps/player1_position_heatmap.png
  - output_heatmaps/player2_position_heatmap.png
  - output_heatmaps/combined_player_heatmap.png
  - output_stats/match_statistics.json
```

### 8.3 Output File Specifications

#### Annotated Video Output

**Visual Elements:**
1. **Player Bounding Boxes:** Red rectangles (2px) with "Player 1" / "Player 2" labels
2. **Ball Bounding Box:** Yellow rectangle (2px) with "Ball" label
3. **Court Keypoints:** Red circles (5px radius) numbered 0-13
4. **Frame Number:** Top-left corner, white text (24pt)
5. **Mini-Court:** Top-right corner (250×500px), shows real-time positions
6. **Statistics Box:** Bottom-right corner (350×230px, semi-transparent black)

**Statistics Box Content:**
```
Player 1 | Player 2
Shot Speed:  42.3 km/h | 45.1 km/h
Player Speed:  5.2 km/h |  6.1 km/h
Avg Shot:    41.8 km/h | 44.5 km/h
Avg Speed:    4.9 km/h |  5.8 km/h
```

#### Heatmap Image Outputs

**Ball Bounce Heatmap (`ball_bounce_heatmap.png`):**
- Size: 1000×2000 pixels (high resolution for reports)
- Color scheme: Blue (low) → Green → Yellow → Red (high)
- Overlaid on white mini-court outline with labeled lines
- Title: "Ball Bounce Distribution"
- Colorbar showing density scale

**Player Position Heatmaps:**
- `player1_position_heatmap.png`: Red gradient heatmap
- `player2_position_heatmap.png`: Blue gradient heatmap
- `combined_player_heatmap.png`: Overlay of both players (red + blue = purple in overlap zones)
- Size: 1000×2000 pixels each
- Titles: "Player 1 Court Coverage", "Player 2 Court Coverage", "Combined Court Coverage"

**Statistics JSON (`match_statistics.json`):**
```json
{
  "video_metadata": {
    "filename": "match.mp4",
    "duration_seconds": 1800,
    "total_frames": 43200,
    "processed_date": "2025-11-11T13:34:29Z"
  },
  "player_1": {
    "total_shots": 78,
    "average_shot_speed_kmh": 41.8,
    "max_shot_speed_kmh": 78.3,
    "total_distance_covered_meters": 2435.7,
    "average_movement_speed_kmh": 4.9,
    "max_movement_speed_kmh": 12.4,
    "court_coverage_percentage": 68.2,
    "zone_distribution": {
      "baseline_left": 32.1,
      "baseline_center": 28.5,
      "baseline_right": 18.3,
      "midcourt_left": 8.7,
      "midcourt_center": 7.2,
      "midcourt_right": 5.2
    }
  },
  "player_2": {
    "total_shots": 78,
    "average_shot_speed_kmh": 44.5,
    "max_shot_speed_kmh": 82.1,
    "total_distance_covered_meters": 2687.3,
    "average_movement_speed_kmh": 5.8,
    "max_movement_speed_kmh": 14.2,
    "court_coverage_percentage": 71.5,
    "zone_distribution": {
      "baseline_left": 29.8,
      "baseline_center": 31.2,
      "baseline_right": 20.4,
      "midcourt_left": 7.1,
      "midcourt_center": 6.8,
      "midcourt_right": 4.7
    }
  },
  "match_summary": {
    "total_rallies": 156,
    "average_rally_duration_seconds": 11.5,
    "longest_rally_duration_seconds": 47.2,
    "total_ball_bounces": 892
  }
}
```

### 8.4 Error Handling & User Feedback

**Error Messages:**

```
ERROR: Video file not found
  → Path: input_videos/match.mp4
  → Solution: Check file path and ensure video exists

ERROR: Unsupported video format
  → Format: .wmv
  → Supported formats: .mp4, .avi, .mov
  → Solution: Convert video to MP4 using ffmpeg

ERROR: Video resolution too low
  → Resolution: 640×480
  → Minimum required: 1280×720
  → Solution: Use higher quality video source

ERROR: GPU out of memory
  → Attempted allocation: 8.2 GB
  → Available: 6.0 GB
  → Solution: Use --device cpu or process shorter video segments

WARNING: Low ball detection rate
  → Detected in 67% of frames (target: 95%)
  → Possible causes: Poor lighting, ball too small, camera angle
  → Impact: Speed calculations may be inaccurate
  → Suggestion: Improve video quality or adjust detection threshold

WARNING: Player ID switch detected
  → Frame 15,234: Player IDs swapped
  → Automatically corrected using position history
  → Impact: Minor discontinuity in tracking visualization
```

## 9. Out of Scope

The following features are explicitly **NOT included** in Version 1.0:

### 9.1 Features Deferred to Future Versions

**Web Application Interface (Phase 2):**
- Browser-based video upload
- Real-time processing progress in web UI
- Interactive heatmap exploration (zoom, filter)
- User account management and video library

**Advanced Analytics:**
- Serve detection and classification (first serve vs. second serve)
- Stroke type classification (forehand, backhand, volley, smash)
- Ball spin estimation
- In/out line call detection
- Net touch detection
- Fault detection (foot fault, double fault)

**Multi-Camera Support:**
- Synchronization of multiple camera angles
- 3D ball trajectory reconstruction
- Stereoscopic court reconstruction

**Machine Learning Improvements:**
- Automatic player identification by jersey color
- Facial recognition for player naming
- Automated highlight reel generation
- Predictive analytics (next shot prediction)

**Integrations:**
- Export to tennis coaching software (Dartfish, Coach's Eye)
- Integration with wearable device data (heart rate, GPS)
- Live match analysis (real-time processing)
- Mobile app (iOS/Android)

### 9.2 Non-Tennis Sports

Version 1.0 is **tennis-specific only**. The following sports are out of scope:
- Badminton, squash, racquetball (similar but different court dimensions)
- Pickleball, padel tennis
- Table tennis (requires different detection scale)
- Generic sports analysis framework

(Future versions may abstract the system for multi-sport support)

### 9.3 Professional-Grade Features

**Not Included (Requires Specialized Hardware):**
- Hawk-Eye level accuracy (<1mm ball position error)
- High-speed camera support (>240 fps)
- Multi-court simultaneous processing
- Broadcast integration (live TV graphics overlay)

### 9.4 Manual Annotation Tools

**Not Included:**
- GUI for correcting detection errors
- Manual labeling interface for training data
- Video annotation software (use external tools like CVAT or LabelImg)

### 9.5 Custom Model Training Pipeline

**Not Included in Main Product:**
- Automated data collection and annotation workflow
- Model architecture search (NAS)
- Model quantization and optimization
- Edge device deployment (mobile/embedded)

(Training scripts provided separately in `/training` directory as optional tools)

## 10. Open Questions

These questions require stakeholder input before implementation:

### 10.1 Technical Decisions

**Q1: Ball Detection Threshold Trade-off**
- **Question:** Should we prioritize recall (detect all balls, more false positives) or precision (fewer false positives, miss some balls)?
- **Current Setting:** Confidence threshold = 0.15 (low threshold for high recall)
- **Impact:** Low threshold causes occasional false detections (e.g., white lines, player clothing). High threshold misses fast-moving balls.
- **Options:**
  - A) Keep low threshold (0.15) + aggressive false positive filtering
  - B) Increase threshold (0.25) + accept lower recall
  - C) Make threshold configurable per video
- **Decision Needed By:** Week 3

**Q2: Heatmap Time Windowing**
- **Question:** Should heatmaps show the entire match or allow time-based filtering?
- **Use Case:** Coaches may want to see "court coverage in first set" vs. "court coverage in third set" to analyze fatigue.
- **Options:**
  - A) Full-match heatmaps only (simplest)
  - B) Generate heatmaps per set (requires set boundary detection)
  - C) Generate heatmaps in 5-minute windows
- **Impact:** Options B/C increase processing time by 3-5x
- **Decision Needed By:** Week 5

**Q3: Cloud GPU Platform Preference**
- **Question:** Should documentation prioritize RunPod or Modal?
- **Factors:**
  - RunPod: Better for interactive development, easier debugging
  - Modal: Better for production, more cost-efficient for batch jobs
- **Options:**
  - A) Document both equally
  - B) Primary guide for RunPod, secondary for Modal
  - C) Primary guide for Modal, secondary for RunPod
- **Decision Needed By:** Week 1

### 10.2 Product Scope

**Q4: Player Height Input Method**
- **Question:** How should users specify player heights?
- **Current:** Command-line arguments (--player1-height, --player2-height)
- **Alternatives:**
  - A) Configuration file (YAML/JSON)
  - B) Interactive prompt at runtime
  - C) Automatic estimation from bounding box (less accurate)
  - D) Database lookup by player name (requires player identification)
- **Decision Needed By:** Week 2

**Q5: Multiple Video Batch Processing**
- **Question:** Should the system support processing multiple videos in one command?
- **Use Case:** Process entire tournament (10-50 videos) overnight
- **Implementation:**
  - `python main.py --input-dir tournaments/wimbledon_2024/`
- **Complexity:** Medium (requires parallel processing logic, error isolation)
- **Priority:** Medium (nice-to-have for v1.0)
- **Decision Needed By:** Week 6

**Q6: In/Out Line Calls**
- **Question:** Should v1.0 include basic in/out detection?
- **Approach:** Check if ball bounce position is inside court boundaries
- **Accuracy:** ±20cm (sufficient for coaching, not for refereeing)
- **Complexity:** Low (5-10 hours development)
- **Value:** High (frequently requested feature)
- **Options:**
  - A) Include in v1.0 (stretches timeline by ~1 week)
  - B) Defer to v1.1
- **Decision Needed By:** Week 2

### 10.3 User Experience

**Q7: Output Video Length vs. File Size**
- **Question:** Should we compress output videos for smaller file sizes?
- **Current:** Same resolution as input (e.g., 1080p → 1080p output)
- **Trade-off:** 
  - High quality: 30-min video = 5-10 GB file size
  - Compressed: 30-min video = 500 MB - 1 GB (more sharing-friendly)
- **Options:**
  - A) Keep high quality (prioritize visual detail)
  - B) Compress to 1080p @ 5 Mbps (H.264)
  - C) Make configurable (--output-quality {low, medium, high})
- **Decision Needed By:** Week 6

**Q8: Heatmap Color Scheme**
- **Question:** Should heatmap colors be customizable?
- **Current:** Ball bounce = blue→red, Player 1 = red, Player 2 = blue
- **Alternatives:**
  - Viridis (colorblind-friendly)
  - Grayscale (for black-and-white printing)
  - Custom RGB values via config file
- **Options:**
  - A) Fixed color schemes (simplest)
  - B) Preset options (--heatmap-style {default, colorblind, grayscale})
  - C) Fully customizable (config file)
- **Decision Needed By:** Week 5

### 10.4 Documentation & Support

**Q9: Video Tutorial Content**
- **Question:** Should we create video tutorials beyond written documentation?
- **Content Ideas:**
  - Installation walkthrough (10 min)
  - Processing first video (5 min)
  - RunPod setup guide (15 min)
  - Interpreting results (10 min)
- **Effort:** ~2 days recording + editing
- **Value:** High for beginners, reduces support burden
- **Decision:** Create tutorials or rely on written docs + screenshots?
- **Decision Needed By:** Week 8

**Q10: Sample Data Distribution**
- **Question:** Should we include sample videos in the repository?
- **Problem:** Video files are large (50-500 MB), bloat repository
- **Options:**
  - A) No sample videos (users bring their own)
  - B) Link to external sample videos (YouTube, Google Drive)
  - C) Include 1 small sample video (10 seconds, 5 MB)
  - D) Separate sample-data repository
- **Decision Needed By:** Week 1

---

## 11. Success Criteria Summary

Version 1.0 is considered **successful** if:

✅ **Functional Completeness:**
- All functional requirements (FR-001 to FR-024) implemented and tested
- Processes 30-minute tennis match video end-to-end without crashes

✅ **Accuracy Targets Met:**
- Ball detection: ≥95% recall
- Player tracking: ≥98% accuracy (no ID swaps)
- Shot speed: <5% error vs. ground truth
- Heatmaps generated for all test videos

✅ **Performance Targets Met:**
- Processes 1 minute of video in <2 minutes on cloud GPU
- Total processing time <60 minutes for 30-minute match

✅ **Usability Validated:**
- 3 external developers (new to ML) complete setup in <4 hours
- 95% of users successfully process sample video on first attempt

✅ **Deliverables Complete:**
- GitHub repository with complete source code
- README with setup instructions and examples
- Sample video and expected outputs
- Documentation for cloud GPU platforms
- 5 heatmap images and 1 annotated video from test match

---

## Appendix A: Glossary

**For developers new to coding and machine learning:**

- **Bounding Box:** A rectangle drawn around an object in an image, defined by 4 coordinates: (x_min, y_min, x_max, y_max). Example: A player at position (100, 200) with width 50 and height 100 has box (100, 200, 150, 300).

- **CNN (Convolutional Neural Network):** A type of AI model specialized for analyzing images. It learns patterns by applying filters across the image, similar to how your eye scans a scene.

- **Confidence Score:** A number between 0 and 1 indicating how sure the model is about a detection. 0.95 = 95% confident. Higher = better.

- **Frame:** A single image from a video. A 30 fps (frames per second) video contains 30 images per second. A 1-minute video = 1,800 frames.

- **GPU (Graphics Processing Unit):** A specialized computer chip that processes images and AI models much faster than regular CPUs. Essential for video processing (20-50x speedup).

- **Heatmap:** A visualization showing density or frequency using colors. Hot areas (red) = high activity, cold areas (blue) = low activity. Example: A ball bounce heatmap shows where balls landed most often.

- **Inference:** Running a trained AI model on new data to make predictions. Example: Using a ball detection model on a new video.

- **Interpolation:** Filling in missing values between known values. If ball is at position (100, 200) in frame 1 and (120, 220) in frame 3, interpolation estimates frame 2 position as (110, 210).

- **Keypoint:** A specific landmark on an object. For a tennis court: corners, line intersections, net posts. Each keypoint has (x, y) coordinates.

- **Mini-Court:** A scaled-down, top-down view of the tennis court used for visualization. Transforms 3D court captured by camera into 2D bird's-eye view.

- **Object Detection:** AI task of finding objects in images and drawing bounding boxes around them. Example: Detecting all people in a photo.

- **Object Tracking:** Following an object across video frames with consistent identity. Example: Tracking "Player 1" through entire match, even when they move around.

- **Perspective Transformation:** Converting a tilted camera view into a top-down view. Like taking a photo of a painting at an angle, then digitally straightening it.

- **Pre-trained Model:** An AI model already trained on millions of images, ready to use immediately. Like buying a trained dog vs. training a puppy yourself.

- **PyTorch:** A popular Python library for building and running AI models. Created by Meta (Facebook).

- **Recall:** Percentage of objects correctly detected out of all objects present. 95% recall = detected 95 out of 100 balls.

- **ResNet50:** A specific CNN architecture with 50 layers, proven effective for image recognition tasks.

- **YOLO (You Only Look Once):** A fast object detection AI model that can detect multiple objects in real-time. Popular for sports analytics.

---

## Appendix B: Development Timeline

### Detailed Week-by-Week Plan

**Week 1: Project Setup & Foundation**
- Day 1-2: Repository setup, dependency installation, documentation structure
- Day 3-4: Video I/O pipeline (read frames, save videos)
- Day 5-6: Player detection with YOLOv8 (load model, run inference)
- Day 7: Testing player detection on sample videos

**Week 2: Court Analysis**
- Day 8-9: Court keypoint detection model integration
- Day 10-11: Coordinate transformation system (pixel → meter)
- Day 12-13: Mini-court visualization rendering
- Day 14: Documentation update (setup guides, CONCEPTS.md)

**Week 3: Player Tracking**
- Day 15-16: Player tracking implementation (ByteTrack)
- Day 17-18: Player ID assignment and persistence
- Day 19-20: Player position extraction and validation
- Day 21: Integration testing (end-to-end player pipeline)

**Week 4: Ball Detection**
- Day 22-23: Ball detection model fine-tuning (if needed)
- Day 24-25: Ball detection inference and validation
- Day 26-27: Ball trajectory interpolation
- Day 28: Performance optimization (inference speed)

**Week 5: Ball Analysis**
- Day 29-30: Ball bounce detection logic
- Day 31-32: Shot detection algorithm
- Day 33-34: Shot speed calculation
- Day 35: Testing ball analysis accuracy

**Week 6: Analytics & Heatmaps**
- Day 36-37: Player movement speed and distance tracking
- Day 38-39: Ball bounce heatmap generation
- Day 40-41: Player position heatmap generation
- Day 42: Statistics aggregation and JSON export

**Week 7: Visualization & Output**
- Day 43-44: Output video rendering with all overlays
- Day 45-46: Statistics overlay panel
- Day 47-48: Mini-court real-time updates
- Day 49: Visual polish and styling

**Week 8: Integration & Polish**
- Day 50-51: End-to-end testing with multiple videos
- Day 52-53: Cloud GPU setup guides (RunPod, Modal)
- Day 54-55: Documentation completion and video tutorials
- Day 56: Final testing, bug fixes, and release preparation

---

## Next Steps: Sub-PRD Creation

Now that I've created the comprehensive Master PRD, I'll create **separate focused sub-PRDs** for parallel development. Each sub-PRD will be standalone and can be fed directly to Claude Code.

**I'll create these 6 sub-PRDs:**

1. **Video Processing Pipeline** - I/O, frame management, caching
2. **Player Detection & Tracking** - YOLO, ByteTrack, ID persistence
3. **Ball Detection & Analysis** - Fine-tuned YOLO, interpolation, bounce/shot detection
4. **Court Analysis & Transformation** - Keypoint detection, coordinate mapping, mini-court
5. **Analytics Engine** - Speed, distance, statistics calculation
6. **Heatmap & Visualization** - Heatmap generation, video annotation, output rendering

**Would you like me to proceed with creating all 6 sub-PRDs now?** Each will be formatted for direct use with Claude Code and include:
- Complete requirements for that module
- API contracts (how it interfaces with other modules)
- Acceptance criteria
- Code structure guidance
- Test specifications

Let me know if you'd like any adjustments to the Master PRD first, or if I should proceed with the sub-PRDs! 🎾