# Portable data packages

These ZIP files are generated from the local, Git-ignored data and can be
uploaded to Google Drive or a private Kaggle Dataset. Extract them into the
repository root so their paths match the Python scripts.

## Packages

- `badminton-raw-v1-2026-07-01.zip`
  - `videos/`
  - `clips/`
  - `InPlay/videos/`
  - unique original recordings and clips from `scratch/`
- `badminton-features-v1-2026-07-01.zip`
  - `features/`
  - `clip_features/`
  - `InPlay/features/`
  - `InPlay/clip_features/`
  - `InPlay/data/`
  - `InPlay/combined_features.npy`
- `badminton-models-v1-2026-07-01.zip`
  - final/exported files under `models/` (not per-epoch checkpoints)
  - `InPlay/models/`
  - `src/TrackNetV3/ckpts/`
  - trained YOLO shuttle-detector weights
  - legacy root-level model files

Generated visualizations, prediction videos, OS metadata, zero-byte arrays,
downloadable baseline YOLO weights, duplicate source files, and per-epoch
classifier checkpoints are intentionally excluded.

## Colab

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone <repository-url> /content/badminton
!unzip -q "/content/drive/MyDrive/badminton-data/badminton-features-v1-2026-07-01.zip" -d /content/badminton
```

Use the accompanying `.sha256` files to verify an upload or download with
`sha256sum -c <filename>.sha256`.
