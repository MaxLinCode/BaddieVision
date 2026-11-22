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
