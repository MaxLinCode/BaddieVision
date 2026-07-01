import os
from extract_combined_features import extract_combined_features
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple

def load_features_per_frame(json_path: str) -> List[np.ndarray]:
    """Load frame-level features from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [np.array(frame['features']) for frame in data]

def generate_frame_labels(total_frames: int, rally_segments: List[Tuple[int, int]]) -> List[int]:
    """Label each frame: 1 if in a rally segment, 0 otherwise."""
    labels = [0] * total_frames
    for start, end in rally_segments:
        for i in range(start, end + 1):
            if 0 <= i < total_frames:
                labels[i] = 1
    return labels

def split_all_sequences(features: List[np.ndarray], labels: List[int], sequence_length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split into all overlapping sequences for full video, including 0-labeled segments."""
    sequences = []

    for i in range(0, len(features) - sequence_length + 1):
        x_seq = np.stack(features[i:i + sequence_length])
        y_seq = np.array(labels[i:i + sequence_length])
        sequences.append((x_seq, y_seq))
    return sequences

# === CONFIGURATION ===
rally_intervals = [
    (356, 427),
    (612, 982),
    (1463, 1814),
    (2032, 2167),
    (2398, 2697),
    (2941, 3128),
    (3384, 3541),
    (3376, 3814),
    (4035, 4155),
    (4339, 4948),
    (5172, 5382),
    (5621, 5799),
    (6036, 6202),
    (6411, 6604),
    (6884, 6999),
    (7228, 7338),
    (7645, 7918),
    (8210, 8263),
    (8469, 8743),
    (8971, 9128),
    (9346, 9451),
    (9722, 9844),
]
sequence_len = 36
max_frame = 9844
# Paths – change these for each new video
INPLAY_DIR = Path(__file__).resolve().parent
VIDEO_PATH = INPLAY_DIR / "videos" / "IMG_3218.mp4"
POSE_PATH = INPLAY_DIR / "features" / "pose" / "img_3218_pose.json"
SHUTTLE_PATH = INPLAY_DIR / "features" / "shuttle" / "IMG_3218_ball.csv"
OUTPUT_PATH = INPLAY_DIR / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# === PROCESSING ===
# Step 1: Read total frame count from video
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Enforce max frame count
total_frames = min(total_frames, max_frame)
cap.release()

# Step 2: Load precomputed features (assumes 1 per frame)
features = extract_combined_features(POSE_PATH, SHUTTLE_PATH, list(range(total_frames)), label_path=None)
#assert len(features) == total_frames, "Mismatch between video frame count and feature file."
features = features[:total_frames]
# Step 3: Generate per-frame labels and split into LSTM sequences
labels = generate_frame_labels(total_frames, rally_intervals)
sequences = split_all_sequences(features, labels, sequence_len)

# Step 4: Save result
X = np.array([x for x, _ in sequences])
Y = np.array([y for _, y in sequences])

np.save(OUTPUT_PATH / "img_3418_X.npy", X)
np.save(OUTPUT_PATH / "img_3418_Y.npy", Y)

print(f"✅ Saved {len(sequences)} LSTM sequences to img_3418_X.npy and img_3418_Y.npy")
