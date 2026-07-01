# Inference script for in-play shuttle classification
import os
import numpy as np
import torch
from pathlib import Path
from torch.nn.functional import sigmoid
from train_lstm_model import LSTMClassifier  # Assuming model definition is in train_lstm_model.py
from extract_combined_features import extract_combined_features  # Using the shared feature extractor
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# --- Config ---
SEQUENCE_LENGTH = 15
INPLAY_DIR = Path(__file__).resolve().parent
POSE_PATH = INPLAY_DIR / "features" / "pose" / "img_3218_pose.json"
SHUTTLE_PATH = INPLAY_DIR / "features" / "shuttle" / "IMG_3218_ball.csv"
MODEL_PATH = INPLAY_DIR / "models" / "best_model-6-12-0524.pt"
MIN_FRAMES_RALLY_LENGTH = 40  # Minimum number of frames to qualify as a rally
GRACE_FRAMES = 3
THRESHOLD = 0.31
DEBUG_WINDOW = 3  # Number of frames before and after rally start/end to debug
SOFT_START_WINDOW = 30  # must have 5 consecutive high predictions to start a rally

# --- Frame selection ---
frame_indices = list(range(12500, 15000))

def smooth_preds(preds: np.ndarray, window_size: int = 5):
    smoothed = uniform_filter1d(preds.astype(float), size=window_size)
    return (smoothed > 0.5).astype(int)

# --- Load Combined Features ---
combined = extract_combined_features(POSE_PATH, SHUTTLE_PATH, frame_indices, label_path=None)
X_raw = combined.astype(np.float32)

# --- Create sequence input ---
X_seq = []
for i in range(SEQUENCE_LENGTH, len(X_raw)):
    X_seq.append(X_raw[i-SEQUENCE_LENGTH:i])
X_seq = np.stack(X_seq)
X_seq_tensor = torch.tensor(X_seq)

# --- Load Model ---
input_dim = X_seq.shape[2]
model = LSTMClassifier(input_size=input_dim, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --- Predict ---
# Initialize arrays to accumulate predictions and counts
frame_preds_sum = np.zeros(len(frame_indices))
frame_preds_count = np.zeros(len(frame_indices))

with torch.no_grad():
    logits = model(X_seq_tensor)  # shape: (num_sequences, seq_len)
    probs = torch.sigmoid(logits).numpy()  # shape: (num_sequences, seq_len)

# Aggregate predictions across sequences
for i in range(len(probs)):
    for t in range(SEQUENCE_LENGTH):
        frame_idx = i + t
        if frame_idx < len(frame_preds_sum):
            frame_preds_sum[frame_idx] += probs[i, t]
            frame_preds_count[frame_idx] += 1

# Compute averaged prediction per frame
avg_probs = frame_preds_sum / np.maximum(frame_preds_count, 1)
avg_preds = (avg_probs > THRESHOLD).astype(int)

def segment_rallies(preds: np.ndarray, frame_indices: list, min_rally_len: int = 50, grace: int = 3):
    rally_segments = []
    rally_start = None
    grace_counter = 0

    for i, p in enumerate(preds):
        frame_id = frame_indices[i]
        if p == 1:
            if rally_start is None:
                # Apply soft-start heuristic
                if i + SOFT_START_WINDOW < len(avg_preds):
                    if np.all(avg_preds[i:i+SOFT_START_WINDOW] == 1):
                        rally_start = frame_id
            else:
                grace_counter = 0
        else:
            if rally_start is not None:
                grace_counter += 1
                if grace_counter >= grace:
                    # Look ahead up to 30 frames to see if in-play resumes
                    recovery_window = preds[i:i+30]
                    if np.sum(recovery_window) == 0:  # No recovery — it's a real end
                        rally_end = frame_id
                        if rally_end - rally_start >= min_rally_len:
                            rally_segments.append((rally_start, rally_end))
                        rally_start = None
                        grace_counter = 0
                    else:
                        # False end — treat it as noise and continue the rally
                        grace_counter = 0

    if rally_start is not None:
        rally_end = frame_indices[-1]
        if rally_end - rally_start >= min_rally_len:
            rally_segments.append((rally_start, rally_end))

    return rally_segments

# --- Output ---
# Optionally smooth
#binary_preds = smooth_preds(preds, window_size=5)

rally_segments = segment_rallies(avg_preds, frame_indices, min_rally_len=MIN_FRAMES_RALLY_LENGTH, grace=GRACE_FRAMES)

print("\nDetected Rallies:")
for start, end in rally_segments:
    print(f"Rally from frame {start} to {end}")
    # start_idx = frame_indices.index(start)
    # end_idx = frame_indices.index(end)

    # print("  Start Debug Window:")
    # for j in range(max(start_idx - DEBUG_WINDOW, 0), min(start_idx + DEBUG_WINDOW + 1, len(avg_preds))):
    #     print(f"    Frame {frame_indices[j]}: In Play = {avg_preds[j]} (conf = {avg_probs[j]:.2f})")

    # print("  End Debug Window:")
    # for j in range(max(end_idx - DEBUG_WINDOW, 0), min(end_idx + DEBUG_WINDOW + 1, len(avg_preds))):
    #     print(f"    Frame {frame_indices[j]}: In Play = {avg_preds[j]} (conf = {avg_probs[j]:.2f})")

# Simulate loading prediction data (replace these with your actual arrays)
# These would typically be produced by your model after inference
# Plotting
plt.figure(figsize=(12, 4))
plt.plot(frame_indices, avg_probs, label="Confidence (avg_probs)", color='blue')
plt.plot(frame_indices, avg_preds, label="Predicted In-Play", color='red', linestyle='--')
plt.axhline(0.31, color='gray', linestyle=':', label='Threshold = 0.31')
plt.xlabel("Frame Index")
plt.ylabel("Probability / Prediction")
plt.title("Model Confidence and Binary Predictions Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# some time rally is ended too late due to random spike after
