#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master Documentary Slideshow Video Generator

Author: Abderrahmane Bouamama
Project: Scan2Speech

Creates a Full HD 1080p documentary video by synchronizing an audiobook narration
with a sequence of background images distributed evenly across the total audio runtime.
- Automatically scales and centers images to 1920x1080 Full HD
- Calculates precise per-slide duration from audio length
- Broadcast-grade H.264 / AAC video output compatible with YouTube and all platforms
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from PIL import Image


def get_ffmpeg_binary() -> str:
    """Find ffmpeg binary dynamically on Windows, Linux, or macOS."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            matches = list(Path(local_app_data).glob("Microsoft/WinGet/Packages/**/bin/ffmpeg.exe"))
            if matches:
                return str(matches[0])
            matches = list(Path(local_app_data).glob("Programs/**/ffmpeg.exe"))
            if matches:
                return str(matches[0])

        for fallback in [r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
            if Path(fallback).exists():
                return fallback

    return "ffmpeg"


def get_ffprobe_binary() -> str:
    """Find ffprobe binary dynamically on Windows, Linux, or macOS."""
    found = shutil.which("ffprobe")
    if found:
        return found

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            matches = list(Path(local_app_data).glob("Microsoft/WinGet/Packages/**/bin/ffprobe.exe"))
            if matches:
                return str(matches[0])
            matches = list(Path(local_app_data).glob("Programs/**/ffprobe.exe"))
            if matches:
                return str(matches[0])

        for fallback in [r"C:\ffmpeg\bin\ffprobe.exe", r"C:\Program Files\ffmpeg\bin\ffprobe.exe"]:
            if Path(fallback).exists():
                return fallback

    return "ffprobe"


def get_audio_duration(audio_file: Path) -> float:
    """Get accurate audio duration in seconds using ffprobe."""
    ffprobe = get_ffprobe_binary()
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def normalize_images(raw_images: list[Path], out_dir: Path) -> list[Path]:
    """Ensure all images are exact 1920x1080 Full HD with aspect-ratio preserving centering."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_w, target_h = 1920, 1080

    normalized_paths = []
    print(f"[PREP] Normalizing {len(raw_images)} images to 1920x1080 Full HD...")

    for idx, img_path in enumerate(raw_images, start=1):
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            # Calculate resize preserving aspect ratio
            im_rgb.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

            # Center onto 1920x1080 canvas
            canvas = Image.new("RGB", (target_w, target_h), (12, 12, 15))
            offset_x = (target_w - im_rgb.width) // 2
            offset_y = (target_h - im_rgb.height) // 2
            canvas.paste(im_rgb, (offset_x, offset_y))

            norm_file = out_dir / f"norm_slide_{idx:03d}.png"
            canvas.save(norm_file, "PNG", optimize=True)
            normalized_paths.append(norm_file)

    return normalized_paths


def build_slideshow_video(
    audio_path: Path,
    images_dir: Path,
    output_video_path: Path,
    fps: int = 25,
) -> bool:
    """Build a Full HD slideshow video timed perfectly with the audio."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Discover images
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    raw_images = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts],
        key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0,
    )

    if not raw_images:
        raise FileNotFoundError(f"No valid image files found in {images_dir}")

    print("=" * 60)
    print("  Master Documentary Slideshow Video Generator")
    print("  Creator: Abderrahmane Bouamama")
    print(f"  Audio : {audio_path.name}")
    print(f"  Images: {len(raw_images)} images in {images_dir}")
    print("=" * 60)

    duration_sec = get_audio_duration(audio_path)
    mins = int(duration_sec // 60)
    secs = int(duration_sec % 60)
    print(f"[AUDIO] Total Duration: {duration_sec:.2f}s ({mins}m {secs}s)")

    # Prepare normalized images
    prep_dir = output_video_path.parent / ".cache_slides"
    normalized_images = normalize_images(raw_images, prep_dir)

    num_images = len(normalized_images)
    slide_duration = duration_sec / num_images
    print(f"[TIMING] {num_images} slides: ~{slide_duration:.2f}s per image ({slide_duration / 60:.2f} min)")

    # Build concat demuxer file for FFmpeg
    concat_file = prep_dir / "slideshow_concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for idx, img_path in enumerate(normalized_images):
            clean_path = str(img_path.resolve()).replace("\\", "/")
            f.write(f"file '{clean_path}'\n")
            f.write(f"duration {slide_duration:.4f}\n")
        # Final image repeat needed for FFmpeg concat demuxer
        last_clean = str(normalized_images[-1].resolve()).replace("\\", "/")
        f.write(f"file '{last_clean}'\n")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = get_ffmpeg_binary()

    print(f"[RUN] Encoding 1080p Video to: {output_video_path.name}...")
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "256k",
        "-shortest",
        str(output_video_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(f"[ERROR] FFmpeg failed:\n{res.stderr[-500:]}")
        return False

    size_mb = output_video_path.stat().st_size / (1024 * 1024)
    print(f"\n[DONE] Documentary video generated successfully: {output_video_path}")
    print(f"       File size: {size_mb:.2f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate Documentary Slideshow Video from Audio & Images. Created by Abderrahmane Bouamama."
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        required=True,
        help="Path to audiobook audio file (.mp3 / .wav)",
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing slide/background images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to output MP4 video file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output video framerate (default: 25 fps)",
    )

    args = parser.parse_args()

    out_file = args.output or args.audio.with_name(f"{args.audio.stem}_documentary.mp4")
    try:
        build_slideshow_video(
            audio_path=args.audio,
            images_dir=args.images_dir,
            output_video_path=out_file,
            fps=args.fps,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
