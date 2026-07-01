import cv2
import mediapipe as mp
import json
import os
from tqdm import tqdm

mp_pose = mp.solutions.pose

def extract_pose(video_path, out_json_path, vis_out_path=None):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        # Print debug info
        print("💡 Check if file exists:", os.path.exists(video_path))
        print("💡 Absolute path:", os.path.abspath(video_path))
        return
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
            return

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        frame_kps = {"frame": frame_idx, "keypoints": {}}

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_kps["keypoints"][str(i)] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }

            if vis_out_path:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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
