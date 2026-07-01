import cv2
import sys
import matplotlib.pyplot as plt
from pynput import keyboard
import threading

if len(sys.argv) < 2:
    print("❗ Usage: python script.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
output_label_path = "frame_labels.csv"

sample_rate = 5  # Label every 5 frames

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit(1)

frame_idx = 0
labels = []
key_pressed = None
stop_flag = False

print("🔢 Commands: i = in play, o = not in play, f = forward 1, j = forward 10, k = back 10, q = quit")

# Listener function for capturing key presses
def on_press(key):
    global key_pressed, stop_flag
    try:
        if hasattr(key, 'char'):
            key_pressed = key.char
        if key == keyboard.Key.esc or key.char == 'q':
            stop_flag = True
            return False  # stop listener
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Set up persistent matplotlib window
fig, ax = plt.subplots()
img_obj = None

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or stop_flag:
        break

    display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ax.clear()
    ax.imshow(display_rgb)
    ax.set_title(f"Frame {frame_idx}")
    ax.axis('off')
    fig.canvas.draw()
    plt.pause(0.001)

    key_pressed = None
    while key_pressed is None and not stop_flag:
        plt.pause(0.01)

    if key_pressed == 'i':
        labels.append((frame_idx, 1))
        frame_idx += sample_rate
    elif key_pressed == 'o':
        labels.append((frame_idx, 0))
        frame_idx += sample_rate
    elif key_pressed == 'f':
        frame_idx += 1
    elif key_pressed == 'j':
        frame_idx += 10
    elif key_pressed == 'k':
        frame_idx = max(0, frame_idx - 10)
    elif key_pressed == 'q':
        print("👋 Quitting early...")
        break
    else:
        print("❗ Invalid input. Use 'i', 'o', 'f', 'j', 'k', or 'q'.")

    key_pressed = None

cap.release()
plt.close()

with open(output_label_path, "w") as f:
    f.write("frame,label\n")
    for frame, label in labels:
        f.write(f"{frame},{label}\n")

print(f"✅ Saved {len(labels)} labeled frames to {output_label_path}")