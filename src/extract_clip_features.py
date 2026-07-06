import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    from .court_features import CalibrationRegistry, build_court_anchor_features
except ImportError:
    from court_features import CalibrationRegistry, build_court_anchor_features

# Constants
FRAMES_WINDOW_SIZE = 36  # Maximum number of frames to process per clip
POSE_NUM_KEYPOINTS = 33  # Number of keypoints in MediaPipe Pose
FRAME_WIDTH = 1280       # Width of the video frames
FRAME_HEIGHT = 720       # Height of the video frames

# Paths are relative to the checkout so the project works from any home directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSE_DIR = PROJECT_ROOT / "features" / "pose"
SHUTTLE_DIR = PROJECT_ROOT / "features" / "shuttle"
OUTPUT_DIR = PROJECT_ROOT / "clip_features"
CALIBRATION_REGISTRY = PROJECT_ROOT / "features" / "court" / "calibrations.json"

def load_shuttle_positions(folder_path, base_name, num_frames=FRAMES_WINDOW_SIZE, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT):
    """Load shuttle positions from CSV and calculate velocities and acceleration."""
    shuttle_data = np.zeros((num_frames, 7), dtype=np.float32)  # [x, y, visibility, vx, vy, ax, ay]

    csv_file = os.path.join(folder_path, f"{base_name}_ball.csv")
    if not os.path.exists(csv_file):
        print(f"Warning: Shuttle CSV file not found for {base_name}.")
        return shuttle_data

    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        frame = int(row['Frame'])
        if frame < num_frames:
            if int(row['Visibility']) == 1:
                normalized_x = row['X'] / frame_width
                normalized_y = row['Y'] / frame_height
                shuttle_data[frame, :3] = [normalized_x, normalized_y, 1.0]  # [x, y, visibility]
            else:
                shuttle_data[frame, :3] = [0.0, 0.0, 0.0]  # No visibility

    # Calculate velocity [vx, vy]
    for frame in range(1, num_frames):
        if shuttle_data[frame, 2] == 1.0 and shuttle_data[frame - 1, 2] == 1.0:  # Both frames visible
            shuttle_data[frame, 3] = (shuttle_data[frame, 0] - shuttle_data[frame - 1, 0]) * 100  # vx
            shuttle_data[frame, 4] = (shuttle_data[frame, 1] - shuttle_data[frame - 1, 1]) * 100  # vy
        else:
            shuttle_data[frame, 3:5] = [0.0, 0.0]  # No velocity if visibility is 0

    # Calculate acceleration [ax, ay]
    for frame in range(2, num_frames):
        if shuttle_data[frame, 2] == 1.0 and shuttle_data[frame - 1, 2] == 1.0 and shuttle_data[frame - 2, 2] == 1.0:  # All frames visible
            shuttle_data[frame, 5] = (shuttle_data[frame, 3] - shuttle_data[frame - 1, 3]) * 100  # ax
            shuttle_data[frame, 6] = (shuttle_data[frame, 4] - shuttle_data[frame - 1, 4]) * 100  # ay
        else:
            shuttle_data[frame, 5:7] = [0.0, 0.0]  # No acceleration if visibility is 0

    return shuttle_data

def load_pose_data(pose_path, num_frames, num_keypoints=POSE_NUM_KEYPOINTS):
    """Load pose data from JSON and handle missing keypoints."""
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)

    frames = pose_data[:num_frames]
    pose_array = np.zeros((num_frames, num_keypoints * 2))  # [T, num_keypoints * 2]

    for i, frame in enumerate(frames):
        keypoints = frame.get("keypoints", {})
        if not keypoints:
            print(f"Missing keypoints in frame {i}. Interpolating...")
            if i > 0 and i < num_frames - 1:
                pose_array[i] = (pose_array[i - 1] + pose_array[i + 1]) / 2  # Linear interpolation
            elif i > 0:
                pose_array[i] = pose_array[i - 1]  # Use previous frame
            elif i < num_frames - 1:
                pose_array[i] = pose_array[i + 1]  # Use next frame
            continue

        for j in range(num_keypoints):
            kp = keypoints.get(str(j), {})
            pose_array[i, j * 2] = kp.get("x", 0.0)
            pose_array[i, j * 2 + 1] = kp.get("y", 0.0)

    return pose_array

def process_clip_features(
    pose_dir,
    shuttle_dir,
    output_dir,
    calibration_registry=CALIBRATION_REGISTRY,
):
    """Main processing loop to extract clip features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = CalibrationRegistry(calibration_registry)
    for fname in tqdm(os.listdir(pose_dir), desc="Processing clips"):
        if not fname.endswith("_pose.json"):
            continue

        base_name = fname.replace("_pose.json", "")
        pose_path = os.path.join(pose_dir, fname)

        # Load pose data
        with open(pose_path, "r", encoding="utf-8") as pose_file:
            pose_frames = json.load(pose_file)
        pose_array = load_pose_data(pose_path, FRAMES_WINDOW_SIZE)
        calibration = registry.calibration_for_clip(base_name)
        court_array = build_court_anchor_features(
            pose_frames, calibration, FRAMES_WINDOW_SIZE
        )

        # Load shuttle positions
        shuttle_array = load_shuttle_positions(shuttle_dir, base_name, FRAMES_WINDOW_SIZE)

        # Combine pose and shuttle features
        feature_array = np.concatenate([pose_array, shuttle_array, court_array], axis=1)

        # Save features
        out_path = os.path.join(output_dir, f"{base_name}_features.npy")
        np.save(out_path, feature_array)

        print(f"Saved features for {base_name} → {out_path}")
        print(
            f"Shape: {feature_array.shape} with {POSE_NUM_KEYPOINTS} keypoints, "
            "shuttle positions, and court anchor"
        )

if __name__ == "__main__":
    process_clip_features(POSE_DIR, SHUTTLE_DIR, OUTPUT_DIR)
