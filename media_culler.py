#!/usr/bin/env python3
"""
Media Culler — one-pass sorter for mixed photo/video documentary footage.

WHAT IT DOES
  1. Walks a single source folder (SOURCE_FOLDER).
  2. Classifies each file as image or video.
  3. Scores images for blur (Laplacian variance) and videos for blur
     (Laplacian variance on a sampled middle frame) + flags very short clips.
  4. Groups visually near-identical images (burst shots) using a perceptual hash.
  5. NON-DESTRUCTIVE: copies (never moves/deletes) files into:
        _Keep/        - passed all checks
        _Review/      - borderline blur score, needs your eyes
        _Reject/      - clearly blurry / clearly too short
        _Duplicates/  - near-identical to something already kept
     Your original SOURCE_FOLDER is never modified.
  6. Writes manifest.csv logging every file, its scores, and why it landed
     where it did — so any decision is auditable and reversible.

TWO-STEP WORKFLOW (recommended)
  Step A — analyze only, no copying, just look at the numbers:
      python media_culler.py --source "/path/to/footage" --mode analyze

  This prints score distributions and a suggested blur threshold based on
  YOUR footage (camera/compression varies, so don't trust a hardcoded number).

  Step B — actually sort, using a threshold you choose from Step A:
      python media_culler.py --source "/path/to/footage" --mode sort --blur-threshold 65

REQUIREMENTS
  pip install opencv-python Pillow imagehash
  ffmpeg + ffprobe must be installed and on your PATH (for video frame
  extraction). On Mac: brew install ffmpeg. On Windows: download from
  ffmpeg.org and add to PATH. On Linux: sudo apt install ffmpeg.
"""

import os
import sys
import csv
import shutil
import argparse
import subprocess
import statistics
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency: run `pip install opencv-python`")

try:
    from PIL import Image, ExifTags
except ImportError:
    sys.exit("Missing dependency: run `pip install Pillow`")

try:
    import imagehash
except ImportError:
    sys.exit("Missing dependency: run `pip install imagehash`")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts"}

SHORT_CLIP_SECONDS = 2.0     # clips shorter than this are flagged as accidental triggers
DUPLICATE_HASH_DISTANCE = 5  # lower = stricter "these are duplicates" match


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def get_video_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def extract_middle_frame(video_path, tmp_frame_path):
    duration = get_video_duration(video_path)
    if duration is None:
        return None, None
    midpoint = max(duration / 2.0, 0.1)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(midpoint), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(tmp_frame_path)],
            capture_output=True, timeout=30
        )
        if os.path.exists(tmp_frame_path):
            return tmp_frame_path, duration
    except Exception:
        pass
    return None, duration


def blur_score(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_exif_datetime(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "DateTimeOriginal":
                return value
    except Exception:
        return None
    return None


def perceptual_hash(image_path):
    try:
        return imagehash.phash(Image.open(image_path))
    except Exception:
        return None


def analyze_folder(source_folder):
    """Pass 1: score everything, print stats, no copying."""
    records = []
    tmp_dir = Path(source_folder) / "_tmp_frames"
    tmp_dir.mkdir(exist_ok=True)

    files = [f for f in Path(source_folder).iterdir() if f.is_file()]
    print(f"Found {len(files)} files. Analyzing...\n")

    for f in files:
        ext = f.suffix.lower()
        record = {"file": f.name, "type": None, "blur_score": None,
                  "duration": None, "exif_datetime": None, "phash": None}

        if ext in IMAGE_EXTS:
            record["type"] = "image"
            record["blur_score"] = blur_score(f)
            record["exif_datetime"] = get_exif_datetime(f)
            h = perceptual_hash(f)
            record["phash"] = str(h) if h else None

        elif ext in VIDEO_EXTS:
            record["type"] = "video"
            if ffmpeg_available():
                tmp_frame = tmp_dir / f"{f.stem}_frame.jpg"
                frame_path, duration = extract_middle_frame(f, tmp_frame)
                record["duration"] = duration
                if frame_path:
                    record["blur_score"] = blur_score(frame_path)
                    os.remove(frame_path)
            else:
                record["duration"] = None
        else:
            continue

        records.append(record)
        print(f"  [{record['type'] or '?'}] {f.name} "
              f"blur={record['blur_score']} duration={record['duration']}")

    tmp_dir.rmdir()
    return records


def print_summary_and_suggest_threshold(records):
    img_scores = [r["blur_score"] for r in records if r["type"] == "image" and r["blur_score"] is not None]
    vid_scores = [r["blur_score"] for r in records if r["type"] == "video" and r["blur_score"] is not None]

    print("\n--- SUMMARY ---")
    if img_scores:
        img_scores_sorted = sorted(img_scores)
        p25 = img_scores_sorted[len(img_scores_sorted) // 4]
        print(f"Images: {len(img_scores)} scored. "
              f"min={min(img_scores):.1f} max={max(img_scores):.1f} "
              f"median={statistics.median(img_scores):.1f}")
        print(f"  Suggested blur-threshold (25th percentile): {p25:.1f}")
        print("  -> Anything scoring below this is probably genuinely blurry for THIS shoot.")
    else:
        print("No images found or scored.")

    if vid_scores:
        print(f"Videos: {len(vid_scores)} scored (on a sampled middle frame). "
              f"min={min(vid_scores):.1f} max={max(vid_scores):.1f} "
              f"median={statistics.median(vid_scores):.1f}")
    else:
        print("No videos scored (check ffmpeg/ffprobe is installed and on PATH).")

    short_clips = [r for r in records if r["type"] == "video" and r["duration"] and r["duration"] < SHORT_CLIP_SECONDS]
    if short_clips:
        print(f"\n{len(short_clips)} video(s) under {SHORT_CLIP_SECONDS}s — likely accidental triggers:")
        for r in short_clips:
            print(f"  {r['file']} ({r['duration']:.1f}s)")


def find_duplicate_groups(records):
    """Cluster images whose perceptual hashes are close together (burst shots)."""
    hashed = [(r["file"], imagehash.hex_to_hash(r["phash"])) for r in records
              if r["type"] == "image" and r["phash"]]
    groups = []
    used = set()
    for i, (name_a, hash_a) in enumerate(hashed):
        if name_a in used:
            continue
        group = [name_a]
        for name_b, hash_b in hashed[i + 1:]:
            if name_b in used:
                continue
            if hash_a - hash_b <= DUPLICATE_HASH_DISTANCE:
                group.append(name_b)
                used.add(name_b)
        if len(group) > 1:
            groups.append(group)
            used.add(name_a)
    return groups


def sort_folder(source_folder, blur_threshold, review_margin=15):
    """Pass 2: copy files into Keep/Review/Reject/Duplicates + write manifest.csv."""
    source = Path(source_folder)
    keep_dir = source / "_Keep"
    review_dir = source / "_Review"
    reject_dir = source / "_Reject"
    dup_dir = source / "_Duplicates"
    for d in (keep_dir, review_dir, reject_dir, dup_dir):
        d.mkdir(exist_ok=True)

    records = analyze_folder(source_folder)
    dup_groups = find_duplicate_groups(records)

    # Map filename -> "keep this one" (first in each dup group) vs duplicate
    duplicate_of = {}
    for group in dup_groups:
        keeper = group[0]
        for other in group[1:]:
            duplicate_of[other] = keeper

    manifest_path = source / "manifest.csv"
    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "type", "blur_score", "duration_sec",
                          "exif_datetime", "decision", "reason"])

        for r in records:
            fname = r["file"]
            src_path = source / fname
            decision, reason = "keep", "passed checks"

            if fname in duplicate_of:
                decision, reason = "duplicate", f"near-identical to {duplicate_of[fname]}"
            elif r["type"] == "video" and r["duration"] and r["duration"] < SHORT_CLIP_SECONDS:
                decision, reason = "reject", f"clip under {SHORT_CLIP_SECONDS}s"
            elif r["blur_score"] is not None:
                if r["blur_score"] < blur_threshold:
                    decision, reason = "reject", f"blur score {r['blur_score']:.1f} below threshold {blur_threshold}"
                elif r["blur_score"] < blur_threshold + review_margin:
                    decision, reason = "review", f"blur score {r['blur_score']:.1f} near threshold"

            dest_dir = {"keep": keep_dir, "review": review_dir,
                        "reject": reject_dir, "duplicate": dup_dir}[decision]
            shutil.copy2(src_path, dest_dir / fname)

            writer.writerow([fname, r["type"], r["blur_score"], r["duration"],
                              r["exif_datetime"], decision, reason])

    print(f"\nDone. Originals untouched in {source}.")
    print(f"Sorted copies in _Keep / _Review / _Reject / _Duplicates.")
    print(f"Full log: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Cull mixed photo/video documentary footage.")
    parser.add_argument("--source", required=True, help="Folder containing your photos and videos")
    parser.add_argument("--mode", choices=["analyze", "sort"], default="analyze",
                         help="analyze = just show scores, sort = copy files into decision folders")
    parser.add_argument("--blur-threshold", type=float, default=None,
                         help="Required for --mode sort. Use the value suggested by --mode analyze.")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        sys.exit(f"Folder not found: {args.source}")

    if not ffmpeg_available():
        print("WARNING: ffmpeg/ffprobe not found on PATH. Video scoring will be skipped, "
              "only durations/blur for images will work. Install ffmpeg to enable video analysis.\n")

    if args.mode == "analyze":
        records = analyze_folder(args.source)
        print_summary_and_suggest_threshold(records)
        dup_groups = find_duplicate_groups(records)
        if dup_groups:
            print(f"\n{len(dup_groups)} duplicate/burst group(s) found:")
            for g in dup_groups:
                print(f"  {g}")
    else:
        if args.blur_threshold is None:
            sys.exit("Run --mode analyze first to get a threshold, then pass it with --blur-threshold")
        sort_folder(args.source, args.blur_threshold)


if __name__ == "__main__":
    main()