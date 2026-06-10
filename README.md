# CelebV-HQ Unconditional Face Generation

This folder contains the cleaned, git-friendly code used for the final challenge submission.
Large artifacts such as datasets, generated images, logs, and model checkpoints are intentionally
excluded from this folder.

## Method Summary

- Model: StyleGAN2-ADA, initialized from FFHQ-256 pretrained weights.
- Task: unconditional generation of 256x256 face images matching the CelebV-HQ frame distribution.
- Dataset: sampled CelebV-HQ frames, then quality-filtered to remove severely dark/empty frames.
- Final candidates:
  - Clean 2500 kimg: `outputs/stylegan2ada_faces256_clean/.../network-snapshot-002500.pkl`
  - Clean 3500 kimg equivalent: `outputs/stylegan2ada_faces256_clean_continue/.../network-snapshot-001000.pkl`
  - Clean 4500 kimg equivalent: `outputs/stylegan2ada_faces256_clean_continue_4500/.../network-snapshot-001000.pkl`

## Important Files

- `checkpoints/stylegan2ada_faces_clean_3500kimg_seed0.pkl`: exact 3500 kimg equivalent checkpoint used for reproducible sampling.
- `scripts/sample_hf_celebvhq_frames.py`: sample frames from CelebV-HQ-style video data.
- `scripts/filter_faces_by_quality.py`: luminance/contrast filter used to create `faces_200k_clean`.
- `scripts/train_stylegan2ada_clean_2500.sh`: train the clean 2500 kimg baseline.
- `scripts/train_stylegan2ada_continue_4500.sh`: continue from the 3500 kimg checkpoint to 4500 kimg equivalent.
- `scripts/generate_stylegan2ada_submission.py`: deterministic StyleGAN2-ADA image generation.
- `scripts/generate_stylegan2ada_submission.sh`: environment wrapper for deterministic inference.
- `reports/report_notes.md`: report-ready notes and failure analysis.

## Reproducible Inference

Generate 1000 images with the 3500 kimg equivalent model:

```bash
cd /home/irteam/git/generate_image

CUDA_VISIBLE_DEVICES=7 final_challenge_code/scripts/generate_stylegan2ada_submission.sh \
  --network final_challenge_code/checkpoints/stylegan2ada_faces_clean_3500kimg_seed0.pkl \
  --output_dir samples/stylegan2ada_clean_continue3500_seed0_trunc1_jpg \
  --num_images 1000 \
  --seed 0 \
  --trunc 1.0 \
  --image_ext jpg \
  --jpeg_quality 95 \
  --make_zip
```

Generate 1000 images with the 4500 kimg equivalent model after the extra run finishes:

```bash
cd /home/irteam/git/generate_image

CUDA_VISIBLE_DEVICES=7 final_challenge_code/scripts/generate_stylegan2ada_submission.sh \
  --network outputs/stylegan2ada_faces256_clean_continue_4500/00000-faces_200k_clean-mirror-paper256-kimg1000-batch64-ada-target0.7-resumecustom/network-snapshot-001000.pkl \
  --output_dir samples/stylegan2ada_clean_continue4500_seed0_trunc1_jpg \
  --num_images 1000 \
  --seed 0 \
  --trunc 1.0 \
  --image_ext jpg \
  --jpeg_quality 95 \
  --make_zip
```

Determinism settings are enabled in `generate_stylegan2ada_submission.py`:

- fixed Python, NumPy, and PyTorch seeds
- `cudnn.benchmark=False`
- `cudnn.deterministic=True`
- TF32 disabled
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `G.eval()` and `torch.inference_mode()`

## Training

The training wrappers set the CUDA extension cache to `outputs/torch_extensions`, avoiding the slow
fallback path observed when PyTorch tried to use `~/.cache/torch_extensions`.

```bash
cd /home/irteam/git/generate_image
CUDA_VISIBLE_DEVICES=7 final_challenge_code/scripts/train_stylegan2ada_clean_2500.sh
```

Continue from the 3500 kimg equivalent model to produce a 4500 kimg equivalent candidate:

```bash
cd /home/irteam/git/generate_image
CUDA_VISIBLE_DEVICES=7 final_challenge_code/scripts/train_stylegan2ada_continue_4500.sh
```

## Notes

This folder assumes the official `stylegan2-ada-pytorch` directory exists at:

```text
/home/irteam/git/generate_image/stylegan2-ada-pytorch
```

The original pretrained FFHQ-256 network is downloaded by StyleGAN2-ADA when using `--resume=ffhq256`.
## Checkpoint Integrity

The final 3500 kimg equivalent checkpoint has SHA256:

```text
f0c8991e761cda66f69f1dc2f6de33f1850ebdc78d536c1e870c8e7c681b31bd
```

Generating with this copied checkpoint using `seed=0` and `trunc=1.0` reproduces the existing
3500 kimg sample images byte-for-byte by SHA256.
