#!/usr/bin/env python3
"""Filter sampled CelebV-HQ frames into a cleaner StyleGAN training set."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        gray = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)

    arr = np.asarray(gray, dtype=np.float32) / 255.0
    h, w = arr.shape
    center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    gy, gx = np.gradient(arr)

    return {
        "mean": float(arr.mean()),
        "center_mean": float(center.mean()),
        "contrast": float(arr.std()),
        "dark_frac": float((arr < 0.08).mean()),
        "bright_frac": float((arr > 0.95).mean()),
        "edge_mean": float(np.sqrt(gx * gx + gy * gy).mean()),
    }


def passes_quality(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        args.min_mean <= metrics["mean"] <= args.max_mean
        and metrics["center_mean"] >= args.min_center_mean
        and metrics["contrast"] >= args.min_contrast
        and metrics["dark_frac"] <= args.max_dark_frac
        and metrics["bright_frac"] <= args.max_bright_frac
        and metrics["edge_mean"] >= args.min_edge_mean
    )


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "quantiles_01_05_10_50_90_95_99": [
            float(x) for x in np.quantile(arr, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--min_mean", type=float, default=0.18)
    parser.add_argument("--max_mean", type=float, default=0.78)
    parser.add_argument("--min_center_mean", type=float, default=0.16)
    parser.add_argument("--min_contrast", type=float, default=0.08)
    parser.add_argument("--max_dark_frac", type=float, default=0.40)
    parser.add_argument("--max_bright_frac", type=float, default=0.08)
    parser.add_argument("--min_edge_mean", type=float, default=0.012)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    files = sorted(
        path
        for path in args.source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    rng.shuffle(files)

    selected: list[tuple[Path, dict[str, float]]] = []
    failed: list[str] = []
    metric_values: dict[str, list[float]] = {
        "mean": [],
        "center_mean": [],
        "contrast": [],
        "dark_frac": [],
        "bright_frac": [],
        "edge_mean": [],
    }

    for path in files:
        try:
            metrics = image_metrics(path)
        except Exception as exc:  # noqa: BLE001 - keep bad inputs out of training.
            failed.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        for key, value in metrics.items():
            metric_values[key].append(value)

        if passes_quality(metrics, args):
            selected.append((path, metrics))

    selected_before_cap = len(selected)
    if args.max_images > 0:
        selected = selected[: args.max_images]

    stats = {
        "source": str(args.source),
        "dest": str(args.dest),
        "seed": args.seed,
        "thresholds": {
            "min_mean": args.min_mean,
            "max_mean": args.max_mean,
            "min_center_mean": args.min_center_mean,
            "min_contrast": args.min_contrast,
            "max_dark_frac": args.max_dark_frac,
            "max_bright_frac": args.max_bright_frac,
            "min_edge_mean": args.min_edge_mean,
        },
        "total_files": len(files),
        "failed_files": len(failed),
        "selected_before_cap": selected_before_cap,
        "selected_after_cap": len(selected),
        "metric_summary": {key: summarize(value) for key, value in metric_values.items()},
        "first_failed_files": failed[:20],
    }

    if args.dry_run:
        print(json.dumps(stats, indent=2))
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    for index, (src, _) in enumerate(selected):
        dst = args.dest / f"img_{index:06d}{src.suffix.lower()}"
        if not dst.exists():
            link_or_copy(src, dst)

    stats_path = args.dest / "quality_filter_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
