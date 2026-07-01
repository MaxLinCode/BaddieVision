import sys
from pathlib import Path
from tqdm import tqdm
from pose_extraction import extract_pose

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPLAY_DIR = PROJECT_ROOT / "InPlay"
video_dir = INPLAY_DIR / "videos"
pose_out_dir = INPLAY_DIR / "features" / "pose"
shuttle_out_dir = INPLAY_DIR / "features" / "shuttle"
annotated_out_dir = INPLAY_DIR / "outputs"

# Create directories
pose_out_dir.mkdir(parents=True, exist_ok=True)
annotated_out_dir.mkdir(parents=True, exist_ok=True)

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
        vis_out_path=str(annotated_out_dir / f"{base}_pose.mp4" if annotated_out_dir else None)
        # vis_out_path=None  # Disable visualization if not needed
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
