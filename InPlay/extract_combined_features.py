import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Constants
POSE_NUM_KEYPOINTS = 33
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Paths – change these for each new video
INPLAY_DIR = Path(__file__).resolve().parent
POSE_PATH = INPLAY_DIR / "features" / "pose" / "img_3218_pose.json"
SHUTTLE_PATH = INPLAY_DIR / "features" / "shuttle" / "IMG_3218_ball.csv"
OUTPUT_PATH = INPLAY_DIR / "inference_features.npy"

# Frame sampling strategy
frame_start = 0
frame_end = 2885
frame_step = 5
frames_to_process = list(range(frame_start, frame_end + 1, frame_step))

def load_pose_data(pose_path, frame_indices, num_keypoints=POSE_NUM_KEYPOINTS):
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)

    pose_array = np.zeros((len(frame_indices), num_keypoints * 2), dtype=np.float32)
    valid_mask = np.zeros(len(frame_indices), dtype=bool)

    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(pose_data):
            continue
        frame = pose_data[frame_idx]
        keypoints = frame.get("keypoints", {})
        if not keypoints:
            continue
        valid_mask[i] = True
        for j in range(num_keypoints):
            kp = keypoints.get(str(j), {})
            pose_array[i, j * 2] = kp.get("x", 0.0)
            pose_array[i, j * 2 + 1] = kp.get("y", 0.0)

    for i in range(len(frame_indices)):
        if valid_mask[i]:
            continue
        prev_idx = next((j for j in range(i - 1, -1, -1) if valid_mask[j]), None)
        next_idx = next((j for j in range(i + 1, len(frame_indices)) if valid_mask[j]), None)

        if prev_idx is not None and next_idx is not None:
            pose_array[i] = (pose_array[prev_idx] + pose_array[next_idx]) / 2
        elif prev_idx is not None:
            pose_array[i] = pose_array[prev_idx]
        elif next_idx is not None:
            pose_array[i] = pose_array[next_idx]

    return pose_array

def load_shuttle_data(shuttle_path, frame_indices):
    df = pd.read_csv(shuttle_path).set_index("Frame")
    shuttle_array = np.zeros((len(frame_indices), 7), dtype=np.float32)

    for i, frame in enumerate(frame_indices):
        if frame not in df.index:
            continue
        row = df.loc[frame]
        if int(row['Visibility']) == 1:
            x = row['X'] / FRAME_WIDTH
            y = row['Y'] / FRAME_HEIGHT
            shuttle_array[i, :3] = [x, y, 1.0]

    for i in range(1, len(frame_indices)):
        if shuttle_array[i, 2] == 1.0 and shuttle_array[i-1, 2] == 1.0:
            shuttle_array[i, 3] = (shuttle_array[i, 0] - shuttle_array[i-1, 0]) * 100
            shuttle_array[i, 4] = (shuttle_array[i, 1] - shuttle_array[i-1, 1]) * 100

    for i in range(2, len(frame_indices)):
        if shuttle_array[i, 2] == 1.0 and shuttle_array[i-1, 2] == 1.0 and shuttle_array[i-2, 2] == 1.0:
            shuttle_array[i, 5] = (shuttle_array[i, 3] - shuttle_array[i-1, 3]) * 100
            shuttle_array[i, 6] = (shuttle_array[i, 4] - shuttle_array[i-1, 4]) * 100

    return shuttle_array

def extract_combined_features(pose_path, shuttle_path, frame_indices, label_path=None):
    pose_features = load_pose_data(pose_path, frame_indices)
    shuttle_features = load_shuttle_data(shuttle_path, frame_indices)

    if label_path:
        label_df = pd.read_csv(label_path).drop_duplicates(subset="frame").set_index("frame")
        label_sel = label_df.loc[frame_indices, "label"]
        if hasattr(label_sel, "values"):
            labels = label_sel.values.astype(np.float32).reshape(-1, 1)
        else:
            labels = np.array([label_sel], dtype=np.float32).reshape(-1, 1)
        combined = np.concatenate([pose_features, shuttle_features, labels], axis=1)
    else:
        combined = np.concatenate([pose_features, shuttle_features], axis=1)

    return combined

# Example usage (Training)
if __name__ == "__main__":
    # Training mode
    POSE_PATH = INPLAY_DIR / "features" / "pose" / "img_3218_pose.json"
    SHUTTLE_PATH = INPLAY_DIR / "features" / "shuttle" / "IMG_3218_ball.csv"
    LABELS_PATH = INPLAY_DIR / "in_play_frame_labels.csv"
    OUTPUT_PATH = INPLAY_DIR / "combined_features.npy"

    label_df = pd.read_csv(LABELS_PATH).drop_duplicates(subset="frame").set_index("frame")
    frames_to_process = sorted(label_df.index.tolist())

    combined = extract_combined_features(POSE_PATH, SHUTTLE_PATH, frames_to_process, LABELS_PATH)
    np.save(OUTPUT_PATH, combined)
    print(f"✅ Saved {combined.shape[0]} frames of combined features to {OUTPUT_PATH}")
