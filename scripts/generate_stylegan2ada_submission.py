import argparse
import os
import pickle
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--output_dir", default="samples/stylegan2ada_submission")
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trunc", type=float, default=1.0)
    parser.add_argument("--noise_mode", choices=["const", "random", "none"], default="const")
    parser.add_argument("--image_ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--make_zip", action="store_true")
    parser.add_argument("--zip_name", default="submission.zip")
    return parser.parse_args()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_image(tensor, path, image_ext, jpeg_quality):
    tensor = (tensor * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = tensor[0].permute(1, 2, 0).cpu().numpy()
    pil_image = Image.fromarray(image, "RGB")
    if image_ext == "jpg":
        pil_image.save(path, quality=jpeg_quality, optimize=True)
    else:
        pil_image.save(path)


def main():
    args = parse_args()
    seed_everything(args.seed)

    repo_dir = Path(__file__).resolve().parents[2] / "stylegan2-ada-pytorch"
    sys.path.insert(0, str(repo_dir))
    import dnnlib
    import legacy

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device).eval()
    for param in G.parameters():
        param.requires_grad_(False)

    saved_paths = []
    label = torch.zeros([1, G.c_dim], device=device)
    with torch.inference_mode():
        for index in range(args.num_images):
            z = torch.from_numpy(np.random.RandomState(args.seed + index).randn(1, G.z_dim)).to(device)
            image = G(z, label, truncation_psi=args.trunc, noise_mode=args.noise_mode)
            path = output_dir / f"img_{index:04d}.{args.image_ext}"
            save_image(image, path, args.image_ext, args.jpeg_quality)
            saved_paths.append(path)

    if args.make_zip:
        zip_path = output_dir / args.zip_name
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in saved_paths:
                zf.write(path, arcname=path.name)
        print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
