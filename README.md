# Badminton AI

Computer-vision experiments for analyzing badminton video:

- player pose extraction with MediaPipe;
- shuttle tracking with TrackNetV3 and YOLO;
- badminton-shot classification from pose and shuttle features;
- in-play/rally detection with frame-level and LSTM classifiers.
- metric court projection for static-camera footage.

## Repository layout

```text
.
├── src/                         Main feature extraction and shot classifier
│   └── TrackNetV3/              Vendored TrackNetV3 implementation
├── InPlay/                      Rally/in-play labeling, training, and inference
├── yolov8/shuttle-yolo-dataset/ YOLO config plus the small curated dataset
├── notebooks/                   Research and data-preparation notebooks
├── requirements.txt             Runtime dependencies
└── requirements-dev.txt         Notebook and development dependencies
```

Large and generated assets stay on each development machine and are intentionally
ignored by Git. This includes videos, extracted clips/features, model weights,
checkpoints, inference output, and NumPy training arrays.

## Set up on WSL/Linux or macOS

Python 3.11 is recommended because it has reliable wheel availability across the
computer-vision stack.

```bash
git clone --recurse-submodules <your-github-repository-url>
cd badminton

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If `python3.11` is not available on macOS, install it first with Homebrew:

```bash
brew install python@3.11
```

PyTorch automatically uses CUDA where available. On Apple Silicon, training code
uses Apple's MPS backend when supported and otherwise falls back to CPU.

## Local data and models

After cloning onto another machine, copy or download your private working assets
into the same relative locations:

```text
videos/                         Source shot-classification videos
clips/                          Extracted/labeled shot clips
features/                       Generated pose and shuttle features
clip_features/                  Combined per-clip feature arrays
models/                         TorchScript models and classifier checkpoints
outputs/                        Rendered predictions and plots
InPlay/videos/                  Source rally videos
InPlay/features/                Generated rally pose/shuttle features
InPlay/data/                    LSTM training arrays
InPlay/models/                  In-play model checkpoints
src/TrackNetV3/ckpts/           TrackNetV3 checkpoints
```

The code now resolves these paths from the repository location, so the checkout
does not need to be named `badminton` or live directly under your home directory.

Portable ZIP bundles have been prepared under `data_packages/`. The ZIP files are
ignored by Git and can be uploaded to private cloud storage; their README and
SHA-256 files remain versioned. See
[`data_packages/README.md`](data_packages/README.md) for their contents and restore
instructions.

Do not commit the ZIP files to ordinary Git history. If you later want versioned
models or datasets, configure Git LFS or DVC before adding those files.

## Typical workflows

Extract pose features from videos:

```bash
python src/extract_features.py
python src/extract_clip_features.py
```

Train the shot classifier:

```bash
python src/train_shot_classifier.py
```

Extract and train in-play features:

```bash
python InPlay/extract_features.py
python InPlay/extract_combined_features.py
python InPlay/train_lstm_model.py
```

Train the YOLO shuttle detector:

```bash
python yolov8/shuttle-yolo-dataset/train_yolo.py
```

TrackNetV3 is included as a submodule pointing to the
[`MaxLinCode/TrackNetV3`](https://github.com/MaxLinCode/TrackNetV3) fork and its
`codex/badminton-integration` branch. The fork contains this project's custom
batch-prediction helpers and CUDA/MPS/CPU device support.

TrackNetV3 checkpoints are not included. See
[`src/TrackNetV3/README.md`](src/TrackNetV3/README.md) for the upstream checkpoint
download and its detailed training/inference instructions.

## Court projection

Calibrate a static camera with draggable court-line guides. The browser workflow
also works in headless WSL environments without OpenCV's Qt/X11 window support:

```bash
python src/calibrate_court.py videos/example.mp4 \
  features/court/example.json --frame 0 \
  --preview outputs/example_court_overlay.jpg
```

The command prints a localhost URL; paste it into a browser. Add
`--open-browser` only if WSL supports launching your host browser. The browser
has previous/next, ±10, and direct frame-number controls; changing frames clears
the calibration. Zoom with the mouse wheel or the Zoom ± buttons, pan with the
scrollbars, and use Fit to restore the full-frame view. A crosshair magnifier
follows the pointer.

For each cyan guide named in the header, click two points along the corresponding
painted court line. After four guides are placed, the full projected court
appears in green. Drag the orange handles until the green model aligns with the
video. Guide intersections may lie outside the frame, so cropped outer corners
are supported.

The defaults use the left and right doubles sidelines plus the near and far
short-service lines. Choose other visible floor markings with `--court-lines`:

```bash
python src/calibrate_court.py videos/example.mp4 features/court/example.json \
  --frame 120 --preview outputs/example_court_overlay.jpg \
  --court-lines left_singles_sideline right_singles_sideline \
                near_doubles_long_service far_short_service
```

At least two longitudinal and two cross-court lines are required. Extra visible
lines improve robustness. The older intersection-click workflow remains
available with `--mode points`. The saved homography maps image pixels to court
coordinates in metres, with the origin at court centre, x running left-to-right,
and y running near-to-far.

```python
from src.court_projection import CourtHomography

calibration = CourtHomography.load("features/court/example.json")
feet_xy_metres = calibration.project_to_court([[player_foot_x, player_foot_y]])
# MediaPipe coordinates are normalized, so also pass the source image size:
feet_xy_metres = calibration.project_normalized_to_court(
    [[landmark.x, landmark.y]], (frame_width, frame_height)
)
```

A planar homography is valid for points on the floor (player foot positions and
shuttle landing/contact points). Projecting an airborne shuttle gives only its
vertical image-ray intersection with the court plane, not its true 3D position.

## Publish to GitHub

Once you have created an empty repository on GitHub:

```bash
git add .
git status
git commit -m "Initial project cleanup"
git branch -M main
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

Review `git status` before committing. The ignored multi-gigabyte local assets
should not appear in the staged file list.
