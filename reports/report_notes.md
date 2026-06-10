# Final Challenge Report Notes

## Core Claim

The main challenge was not only improving visual quality, but matching the CelebV-HQ video-frame
distribution under a strict model-size constraint. We found that data curation, snapshot selection,
and truncation control mattered more than increasing model scale.

## Model Choice

- FLUX/LoRA was initially considered, but the task is unconditional distribution matching rather
  than text-conditioned image generation.
- Large diffusion/FLUX checkpoints are difficult to fit under the 5GB final-model constraint if the
  pretrained model is counted.
- StyleGAN2-ADA is compact, fast at sampling, well validated for faces, and supports transfer
  learning from FFHQ-256.

## Dataset Construction

- Raw sampled dataset: `faces_200k`, 200,000 CelebV-HQ frames at 256x256.
- Failure observed: many generated images were very dark or partially invisible.
- Diagnosis: the raw sampled frames contained a large tail of dark/black frames.
- Fix: quality filtering based on luminance, center luminance, contrast, dark-pixel fraction,
  bright-pixel fraction, and edge strength.
- Filter settings:
  - `min_mean=0.18`
  - `max_mean=0.78`
  - `min_center_mean=0.16`
  - `min_contrast=0.08`
  - `max_dark_frac=0.40`
  - `max_bright_frac=0.08`
  - `min_edge_mean=0.012`
- Result: 133,064 images passed; 100,000 were selected with seed 0 as `faces_200k_clean`.

## Training

- Base: StyleGAN2-ADA FFHQ-256 pretrained checkpoint.
- Training set: `faces_200k_clean`.
- Mirror augmentation enabled, so StyleGAN reports 200,000 effective images.
- Batch size: 64.
- ADA target: 0.7.
- Seed: 0.
- Main run: 2500 kimg.
- Continue run: additional 1000 kimg, yielding a 3500 kimg equivalent candidate.
- Extra continue run: additional 1000 kimg from 3500, yielding a 4500 kimg equivalent candidate.

## Reproducibility

Initial inference with a fixed seed did not produce byte-identical images in all cases. We corrected
this by adding deterministic PyTorch/CUDA settings:

- fixed Python, NumPy, and PyTorch seeds
- `torch.backends.cudnn.benchmark=False`
- `torch.backends.cudnn.deterministic=True`
- TF32 disabled
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `G.eval()`
- `torch.inference_mode()`

After the fix, two independent 16-image runs with the same seed and truncation produced identical
SHA256 hashes.

## Failure Cases

- Dark-frame failure: caused by raw video frames with extremely low luminance or black borders.
- Over-cleaning risk: removing too many low-light frames can improve visual quality and IS, but may
  hurt FID if the evaluation set contains genuine low-light CelebV-HQ frames.
- Truncation tradeoff: lower truncation improves average visual fidelity but can reduce diversity,
  harming IS or TopPR.
- Continue-training risk: additional kimg can improve texture/detail, but may overfit or narrow the
  distribution.
- CUDA custom op issue: one continue run used a slow reference fallback for `upfirdn2d`; this should
  not change the intended math significantly, but it made training dramatically slower.

## Recommended Ablations

| Setting | Purpose |
|---|---|
| Raw 200k vs clean 100k | Measure data curation impact |
| 2500 vs 3500 vs 4500 kimg | Check underfitting vs overfitting |
| truncation 0.8, 0.9, 1.0, 1.1 | Tune fidelity/diversity tradeoff |
| snapshot sweep near the end | Avoid assuming the final snapshot is best |

## Conclusion

The strongest practical system was StyleGAN2-ADA transfer learning on filtered CelebV-HQ frames,
with deterministic seed-controlled inference. The most important empirical lesson was that matching
the CelebV-HQ frame distribution required careful data filtering and metric-driven snapshot/truncation
selection, not simply using the largest pretrained generator.
