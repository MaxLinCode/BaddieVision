# Footage augmentation for InPlay

Test whether changes to the footage improve rally segmentation on unseen
cameras. Use `full_motion` on the camera-invariance branch as the baseline.
The experiments below are hypotheses for InPlay.

## First experiments

| Transform | What to test | What to verify |
| --- | --- | --- |
| Horizontal flip | Mirror every frame left to right. Mirror the image landmarks and swap their left/right court labels when building the variant calibration. | For matching points and trajectories, court x and lateral velocity reverse sign while court y and longitudinal velocity are preserved. Check that near/far assignments agree and spot-check left/right joints after rerunning pose estimation. |
| Lighting and color | Vary exposure, contrast, and white balance mildly to approximate different gyms and camera settings. Keep the chosen settings fixed across the clip. | Inspect side-by-side frames for plausible color and exposure, visible court lines and players, and consistent brightness over time. |
| H.264 compression | Re-encode at lower quality to approximate shared or streamed footage. Use a high-quality re-encode with the same encoder as the control for this comparison. | Inspect block artifacts around court lines and moving players. Compare the lower-quality arm with the high-quality control and report both relative to the original baseline. |

Horizontal flipping is common in video training recipes. Video research also
supports applying spatial transforms consistently across frames. Compression
research provides a precedent for varying H.264 quality in robustness trials.
[PyTorchVideo transforms](https://pytorchvideo.readthedocs.io/en/latest/transforms.html),
[CVRL paper](https://arxiv.org/abs/2008.03800),
[Low-quality video study](https://arxiv.org/abs/2204.09166).

Later candidates include downsampling and resizing back to the original frame
dimensions, mild defocus blur, and low-light sensor noise. Try a small fixed
zoom or framing shift after validating the flip calibration. Transform the
calibration with the framing change and keep the entire court visible.
[CVPR video robustness benchmark](https://openaccess.thecvf.com/content/CVPR2023/papers/Schiappa_A_Large-Scale_Robustness_Analysis_of_Video_Action_Recognition_Models_CVPR_2023_paper.pdf).

## Pilot setup and evaluation

1. Use the [clean camera-held-out folds](../rally-training.md#clean-train-and-validation-partitions).
   Build one variant per training recording or training-only clip per experiment
   arm. Keep the original training footage in each arm. Use original footage for
   validation, decoder calibration, and held-out evaluation. Choose severity by
   comparing against artifacts seen in real recordings before training.
2. Give each variant its own source ID and artifact root. Record the parent
   recording or clip, transform settings, and calibration. Preserve the original
   30 fps frame sequence, source-local rally labels, and audio sync. Check frame
   counts and timestamps and inspect frames around a few labeled rally starts
   and ends before extraction. Keep footage and artifacts in ignored local storage.
3. Rerun person tracking, player assignment and pose extraction, then regenerate
   the motion inputs. Inspect a few matched trajectories and pose frames before
   training, including near- and far-court players in flipped footage.
4. Hold folds, training settings, seeds, and optimizer steps fixed. Compare each
   arm independently against `full_motion` trained on original footage. Report
   interval F1, signed rally start and end errors, false splits, and false merges
   for each held-out camera and across cameras. Combine transforms only after
   reviewing the individual results.

## Implementation

The augmentation is implemented as an isolated source workflow, not as a
feature transform inside the trainer:

- `src/workflow/augmentation.py` defines the versioned transform specs. Each
  variant is rendered with FFmpeg with the parent frame count, FPS, dimensions,
  and audio stream preserved. `horizontal-flip` mirrors calibration landmark
  and line coordinates and swaps their left/right names, composing the
  homography with an image reflection and a court-x sign flip. Lighting/color
  and H.264 arms reuse the parent calibration unchanged.
- `tools/build_footage_augmentation.py` registers each variant under
  `<parent>__<slug>` in a dedicated catalog and artifact root, writes its
  provenance, reruns person tracking and RTMPose player extraction, and writes
  extended source-local rally annotations plus an arm experiment config.
- `experiments/temporal_selector/footage_augmentation_experiment.py` trains a
  paired baseline and augmented arm per held-out camera. Both arms share the
  chronological common train/validation split, class weights, original-source
  draw stream, and optimizer steps. Variant windows are restricted to their
  parent's train frame range. Validation, decoder calibration, and held-out
  evaluation use original footage only.
- `tools/verify_footage_augmentation.py` writes side-by-side frames, pose
  overlays, media-property checks, and court-invariant statistics.
  `tools/report_footage_augmentation.py` compares the arms and the current best
  model.

The parent video, calibration, tracks, poses, and annotations are only read.
All variant footage and observations live under an ignored experiment root.

### Reproduction

The commands use an ignored experiment root and the local clean-experiment
config; substitute the local paths and GPU host for your environment.

```bash
python tools/build_footage_augmentation.py \
  --base-config .local-data/footage-augmentation/base-config.json \
  --output-root .local-data/footage-augmentation/run \
  --transforms horizontal-flip lighting-color h264-low h264-control \
  --device cuda

python experiments/temporal_selector/footage_augmentation_experiment.py \
  --config .local-data/footage-augmentation/run/horizontal-flip/experiment.json \
  --output-dir .local-data/footage-augmentation/run/horizontal-flip/training/full_motion \
  --arm horizontal-flip --feature-view full_motion --device cuda

python tools/verify_footage_augmentation.py \
  --run-root .local-data/footage-augmentation/run \
  --parent-config .local-data/footage-augmentation/base-config.json \
  --source-id <source-id>

python tools/report_footage_augmentation.py \
  --run-root .local-data/footage-augmentation/run \
  --reference <current-best-metrics.json>
```

`tools/run_footage_augmentation_training.py` runs every built arm and feature
view, and `tools/package_footage_augmentation.py` copies the metrics and
reports into a retrievable job workspace.

## Pilot result

Setup: seven reviewed 30 fps sources in four camera groups (Malaysia 2, Max-vs-Nik
2, Bothell 2, Max-vs-Lucas 1); one camera held out per fold; seed 1729; 25
epochs; batch 16; AdamW at 3e-4; boundary-balanced sampling; 0.25 boundary
auxiliary weight; 0.1 temporal-smoothing weight. Each arm keeps the originals,
adds its variants only inside the parent's chronological train range, and shares
the common windows, class weights, source draw stream, and optimizer steps with
its baseline. Validation, decoder calibration, and held-out scoring use
originals only. The baseline macro interval F1 is identical across arms
(0.6673 with `full_motion`, 0.7816 with `full_audio_zscore_motion`).

Macro interval F1 on the original held-out cameras:

| Arm | View | Baseline | Augmented | Delta | Cameras improved |
|---|---|---:|---:|---:|---:|
| horizontal-flip | `full_motion` | 0.6673 | 0.7777 | +0.1104 | 3/4 |
| horizontal-flip | `full_audio_zscore_motion` | 0.7816 | 0.8090 | +0.0274 | 3/4 |
| h264-control (CRF 16) | `full_motion` | 0.6673 | 0.7559 | +0.0885 | 3/4 |
| h264-low (CRF 32) | `full_motion` | 0.6673 | 0.7129 | +0.0456 | 1/4 |
| lighting-color | `full_motion` | 0.6673 | 0.6946 | +0.0272 | 1/4 |
| lighting-color | `full_audio_zscore_motion` | 0.7816 | 0.7175 | -0.0641 | 1/4 |

Horizontal flip per camera:

| Camera | `full_motion` baseline | augmented | delta | `full_audio_zscore_motion` baseline | augmented | delta |
|---|---:|---:|---:|---:|---:|---:|
| Bothell | 0.5649 | 0.5873 | +0.0224 | 0.7167 | 0.6885 | -0.0281 |
| Malaysia | 0.4348 | 0.8182 | +0.3834 | 0.7556 | 0.7727 | +0.0172 |
| Max-vs-Lucas | 0.8125 | 0.7912 | -0.0213 | 0.8736 | 0.9176 | +0.0441 |
| Max-vs-Nik | 0.8571 | 0.9143 | +0.0571 | 0.7805 | 0.8571 | +0.0767 |

Horizontal-flip boundary behavior (`full_motion` macro): boundary MAE
18.84 -> 13.33 frames, signed start error 15.05 -> 4.13 frames, signed end
error -6.23 -> -4.66 frames, false splits 4.25 -> 3.50, false merges 0 -> 0.
Under `full_audio_zscore_motion` interval F1 improves while boundary MAE
15.51 -> 13.76 and false splits 3.00 -> 4.25, so the audio-recipe gain is
smaller and less boundary-stable.

Interpretation: horizontal flip is the only consistent winner. It improves
three of four held-out cameras under both feature views, most of all on the
weakest camera, and it reduces boundary error and false splits under
`full_motion`. Lighting/color is not reliable: it helps under `full_motion`
(mainly Malaysia) but loses under the audio recipe. H.264 quality is not a
useful robustness lever here: the low-quality arm improves fewer cameras than
the high-quality re-encode control, so the small low-quality macro gain is not
evidence that compression artifacts help. The current best recorded model
(`full_audio_zscore_motion`, macro interval F1 0.7705) is slightly below the
audio-recipe flip arm (0.8090), but that comparison is cross-runner; the
controlled delta (+0.0274) is the reliable result. Combine transforms only
after testing them individually; these results do not support combining
lighting or compression with flip yet.
