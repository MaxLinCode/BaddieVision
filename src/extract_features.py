import os
import sys
import gc
import torch
from pathlib import Path
from pose_extraction import extract_pose
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
tracknet_path = PROJECT_ROOT / "src" / "TrackNetV3"
if str(tracknet_path) not in sys.path:
    sys.path.insert(0, str(tracknet_path))

from TrackNetV3.predictArgs import run_prediction_batch, load_torchscript_model, run_prediction
from TrackNetV3.utils.general import *

# Paths
video_dir = PROJECT_ROOT / "videos"
pose_out_dir = PROJECT_ROOT / "features" / "pose"
shuttle_out_dir = PROJECT_ROOT / "features" / "shuttle"
annotated_out_dir = PROJECT_ROOT / "outputs"

# Create directories
pose_out_dir.mkdir(parents=True, exist_ok=True)
annotated_out_dir.mkdir(parents=True, exist_ok=True)
print(f"[Before model] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"[Before model] Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# TrackNet checkpoint path. InpaintNet remains available through the explicit
# ``inpaintnet_file``/``reuse_models`` opt-in in TrackNetV3.predictArgs.
tracknet_file = PROJECT_ROOT / "src" / "TrackNetV3" / "ckpts" / "TrackNet_best.pt"

# Load TrackNet checkpoint to CPU
tracknet_ckpt = torch.load(tracknet_file, map_location=torch.device('cpu'))
tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
bg_mode = tracknet_ckpt['param_dict']['bg_mode']

# Load TrackNet TorchScript model
tracknet = load_torchscript_model(str(PROJECT_ROOT / "models" / "TrackNet_torchscript.pt"))

# Clear GPU cache after loading models
torch.cuda.empty_cache()

print(f"[After model] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"[After model] Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Prepare reusable models dict
reuse_models = {
    'tracknet': tracknet,
    'tracknet_seq_len': tracknet_seq_len,
    'bg_mode': bg_mode
}

# Process all video clips with progress bar
video_files = list(video_dir.glob("*.mp4"))
print(f"🎬 Found {len(video_files)} videos to process.")

# Step 1: Extract pose for each video
for video_path in tqdm(video_files, desc="Extracting poses"):
    base = video_path.stem.lower()
    print(f"\n🎬 Extracting pose for: {video_path.name}")
    extract_pose(
        video_path=str(video_path),
        out_json_path=str(pose_out_dir / f"{base}_pose.json"),
        vis_out_path=None
    )

# # Step 2: Run TrackNet for shuttle detection in batch
# # Split video files into smaller chunks
# chunk_size = 100  # Number of videos to process in each chunk
# video_chunks = [video_files[i:i + chunk_size] for i in range(0, len(video_files), chunk_size)]

# print("🏸 Running TrackNet in batches...")
# for video_chunk in tqdm(video_chunks, desc="Processing video chunks for shuttle detection"):
#     run_prediction_batch(
#         video_files=[str(video_path) for video_path in video_chunk],
#         save_dir=str(shuttle_out_dir),
#         batch_size=12,
#         output_video=False,
#         reuse_models=reuse_models
#     )

print("✅ Finished processing all videos.")
