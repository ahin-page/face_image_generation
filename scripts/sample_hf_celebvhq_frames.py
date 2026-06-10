import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
from datasets import Features, Value, load_dataset
from PIL import Image


DEFAULT_DATA_FILE = "hf://datasets/SwayStar123/CelebV-HQ@871eb4efcf330e92d65becae18652dd1f75b084b/videos.tar"
DEFAULT_PROMPT = (
    "a high quality portrait photo of a person, centered face, natural skin texture, "
    "soft studio lighting, sharp focus, realistic facial details"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--celebvhq_info", default="/home1/irteam/git/CelebV-HQ/celebvhq_info.json")
    parser.add_argument("--output_dir", default="data/faces")
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_buffer", type=int, default=2048)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=100000)
    parser.add_argument("--frames_per_clip", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tmp_dir", default=None)
    parser.add_argument("--ffmpeg_bin", default=None)
    parser.add_argument("--ffprobe_bin", default=None)
    parser.add_argument("--black_border_threshold", type=int, default=10)
    parser.add_argument("--min_crop_area_ratio", type=float, default=0.45)
    return parser.parse_args()


def resolve_tool(name, explicit=None):
    if explicit:
        return explicit
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/conda/envs/generate_image/bin", "/opt/conda/bin", "/usr/bin", "/usr/local/bin"):
        candidate = Path(prefix) / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Could not find required tool: {name}")


def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True)


def load_caption_map(info_path):
    info_path = Path(info_path)
    if not info_path.exists():
        return {}

    with info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    appearance_mapping = data.get("meta_info", {}).get("appearance_mapping", [])
    caption_map = {}
    for clip_name, row in data.get("clips", {}).items():
        attrs = row.get("attributes", {})
        appearance = attrs.get("appearance", [])
        labels = [
            appearance_mapping[i].lower().replace("_", " ")
            for i, flag in enumerate(appearance)
            if flag and i < len(appearance_mapping)
        ]
        useful = []
        for label in labels:
            if label in {
                "male",
                "young",
                "eyeglasses",
                "smiling",
                "bald",
                "bangs",
                "black hair",
                "blond hair",
                "brown hair",
                "gray hair",
                "mustache",
                "beard",
            }:
                useful.append(label)
        emotion = attrs.get("emotion", {}).get("labels")
        if isinstance(emotion, str) and emotion and emotion != "unknown":
            useful.append(emotion.lower())
        if useful:
            details = ", ".join(dict.fromkeys(useful))
            caption = f"a high quality portrait photo of a person, {details}, centered face, natural skin texture, sharp focus"
        else:
            caption = DEFAULT_PROMPT
        caption_map[clip_name] = caption
    return caption_map


def probe_duration(video_path, ffprobe_bin):
    result = run([
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])
    if result.returncode != 0:
        return None
    try:
        return max(float(result.stdout.strip()), 0.0)
    except ValueError:
        return None


def postprocess_frame(image_path, resolution, black_border_threshold, min_crop_area_ratio):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    gray = image.convert("L")
    mask = gray.point(lambda value: 255 if value > black_border_threshold else 0)
    bbox = mask.getbbox()
    if bbox is not None:
        crop_w = bbox[2] - bbox[0]
        crop_h = bbox[3] - bbox[1]
        crop_area_ratio = (crop_w * crop_h) / max(width * height, 1)
        if crop_area_ratio >= min_crop_area_ratio:
            image = image.crop(bbox)

    width, height = image.size
    side = min(width, height)
    left = max((width - side) // 2, 0)
    top = max((height - side) // 2, 0)
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    image.save(image_path, quality=95)


def extract_frame(video_path, output_path, timestamp, resolution, ffmpeg_bin, black_border_threshold, min_crop_area_ratio):
    result = run([
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-loglevel",
        "error",
        str(output_path),
    ])
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        return False, result.stderr

    try:
        postprocess_frame(output_path, resolution, black_border_threshold, min_crop_area_ratio)
    except Exception as exc:
        return False, str(exc)

    return output_path.exists() and output_path.stat().st_size > 0, result.stderr


def extract_frames_cv2(video_path, frame_jobs, resolution, black_border_threshold, min_crop_area_ratio):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [(output_path, False, "failed to open video") for output_path, _ in frame_jobs]

    results = []
    for output_path, timestamp in frame_jobs:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            results.append((output_path, False, "failed to read frame"))
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(output_path, quality=95)
        try:
            postprocess_frame(output_path, resolution, black_border_threshold, min_crop_area_ratio)
        except Exception as exc:
            results.append((output_path, False, str(exc)))
            continue

        results.append((output_path, output_path.exists() and output_path.stat().st_size > 0, ""))

    cap.release()
    return results


def sample_timestamps(duration, frames_per_clip):
    if duration is None or duration <= 0:
        duration = 1.0
    frame_count = max(frames_per_clip, 1)
    if frame_count == 1:
        return [duration * 0.5]
    return [duration * (index + 1) / (frame_count + 1) for index in range(frame_count)]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captions_path = output_dir / "captions.jsonl"
    mode = "w" if args.overwrite or args.start_index == 0 else "a"

    ffmpeg_bin = resolve_tool("ffmpeg", args.ffmpeg_bin)
    ffprobe_bin = resolve_tool("ffprobe", args.ffprobe_bin)
    caption_map = load_caption_map(args.celebvhq_info)

    features = Features({"__key__": Value("string"), "__url__": Value("string"), "mp4": Value("binary")})
    dataset = load_dataset("webdataset", data_files=args.data_file, split="train", streaming=True, features=features)
    if args.shuffle_buffer > 1:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    saved = 0
    seen = 0
    output_index = args.start_index
    tmp_parent = Path(args.tmp_dir) if args.tmp_dir else None
    if tmp_parent:
        tmp_parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming CelebV-HQ videos from {args.data_file}")
    print(f"Saving sampled frames to {output_dir}")
    with captions_path.open(mode, encoding="utf-8") as captions_file:
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmp_name:
            tmp_dir = Path(tmp_name)
            for row in dataset:
                if saved >= args.num_images or seen >= args.max_rows:
                    break
                seen += 1
                key = row["__key__"].split("/")[-1]
                video_bytes = row.get("mp4")
                if not video_bytes:
                    continue

                tmp_video = tmp_dir / f"{key}.mp4"
                tmp_video.write_bytes(video_bytes)

                duration = probe_duration(tmp_video, ffprobe_bin)
                frame_jobs = []
                frame_names = []
                for timestamp in sample_timestamps(duration, args.frames_per_clip):
                    if saved + len(frame_jobs) >= args.num_images:
                        break

                    file_name = f"celebvhq_{output_index + len(frame_jobs):06d}.jpg"
                    output_path = output_dir / file_name
                    if output_path.exists() and not args.overwrite:
                        output_index += 1
                        continue
                    frame_jobs.append((output_path, timestamp))
                    frame_names.append(file_name)

                frame_results = extract_frames_cv2(
                    tmp_video,
                    frame_jobs,
                    args.resolution,
                    args.black_border_threshold,
                    args.min_crop_area_ratio,
                )

                for frame_index, (output_path, ok, message) in enumerate(frame_results):
                    file_name = frame_names[frame_index]
                    output_index += 1
                    if not ok:
                        print("skip extract failed {} frame {}: {}".format(key, frame_index, message.strip()))
                        continue

                    captions_file.write(json.dumps({"file_name": file_name, "text": caption_map.get(key, DEFAULT_PROMPT)}, ensure_ascii=False) + "\n")
                    saved += 1
                    if saved == 1 or saved % max(args.log_every, 1) == 0 or saved >= args.num_images:
                        captions_file.flush()
                        print("saved {}/{}: {} from {}".format(saved, args.num_images, file_name, key), flush=True)

                tmp_video.unlink(missing_ok=True)

    print(f"Done. saved={saved}, seen={seen}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
