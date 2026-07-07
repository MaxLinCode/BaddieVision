import cv2
import json
import os
from tqdm import tqdm

from src.pose_estimator import create_pose_estimator, pose_connections


def _draw_pose_landmarks(frame, landmarks):
    points = []
    height, width = frame.shape[:2]
    for landmark in landmarks:
        points.append(
            (
                int(round(float(landmark.x) * width)),
                int(round(float(landmark.y) * height)),
                float(landmark.visibility),
            )
        )
    for start, end in pose_connections():
        if start >= len(points) or end >= len(points):
            continue
        ax, ay, av = points[start]
        bx, by, bv = points[end]
        if av >= 0.5 and bv >= 0.5:
            cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2, cv2.LINE_AA)
    for x, y, visibility in points:
        if visibility >= 0.5:
            cv2.circle(frame, (x, y), 3, (0, 220, 255), -1, cv2.LINE_AA)


def extract_pose(video_path, out_json_path, vis_out_path=None, pose_model_asset=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        # Print debug info
        print("💡 Check if file exists:", os.path.exists(video_path))
        print("💡 Absolute path:", os.path.abspath(video_path))
        return
    pose = create_pose_estimator(model_asset_path=pose_model_asset, running_mode="video")
    # Read FPS and frame size right away
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    keypoints_all = []
    frame_idx = 0

    if vis_out_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_vid = cv2.VideoWriter(vis_out_path, fourcc, fps, (width, height))
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # vis_out_path = vis_out_path.replace(".mp4", ".avi")
        # out_vid = cv2.VideoWriter(vis_out_path, fourcc, fps, (width, height))
        if not out_vid.isOpened():
            print("❌ Failed to open VideoWriter")
            pose.close()
            return

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(round((frame_idx / fps) * 1000)) if fps else frame_idx
        result = pose.estimate_pose(rgb, timestamp_ms=timestamp_ms)

        frame_kps = {"frame": frame_idx, "keypoints": {}}

        if result.detected:
            for i, lm in enumerate(result.landmarks):
                frame_kps["keypoints"][str(i)] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }

            if vis_out_path:
                _draw_pose_landmarks(frame, result.landmarks)

        keypoints_all.append(frame_kps)

        if vis_out_path:
            out_vid.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if vis_out_path:
        out_vid.release()
    pose.close()

    with open(out_json_path, "w") as f:
        json.dump(keypoints_all, f, indent=2)

    print(f"Saved pose data to {out_json_path}")
    if vis_out_path:
        print(f"Saved annotated video to {vis_out_path}")
