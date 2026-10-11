#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audiobook Video Generator with Cover Art

Author: Abderrahmane Bouamama
Project: Scan2Speech

Combines audiobook audio files with cover art or artwork into crisp Full HD 1080p
MP4 videos ready for YouTube, podcasts, and video sharing platforms.
"""

from __future__ import annotations

import argparse
import os
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


def create_video_from_cover(
    audio_path: Path,
    cover_image_path: Path,
    output_mp4_path: Path,
) -> bool:
    """Combine static cover art with audio to produce a Full HD MP4 video."""
    print(f"\n[RUN] Encoding Video: {output_mp4_path.name} from {audio_path.name}...")

    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return False

    if not cover_image_path.exists():
        print(f"[ERROR] Cover image not found: {cover_image_path}")
        return False

    ffmpeg_bin = get_ffmpeg_binary()
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin, "-y",
        "-loop", "1",
        "-i", str(cover_image_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
        "-shortest",
        str(output_mp4_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg failed: {result.stderr[-300:]}")
        return False

    size_mb = output_mp4_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Video created successfully: {output_mp4_path.name} ({size_mb:.2f} MB)")
    return True


def find_default_cover(search_dirs: list[Path]) -> Path | None:
    """Look for standard cover images (cover.jpg, cover.png, etc.)."""
    candidates = [
        "cover.jpg", "cover.png", "cover.jpeg",
        "cover_background.jpg", "cover_background.png",
        "artwork.jpg", "artwork.png",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for c in candidates:
            p = d / c
            if p.exists():
                return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate MP4 Videos from Audiobook Audio and Cover Art. Created by Abderrahmane Bouamama."
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        default=None,
        help="Path to audio file (.mp3 / .wav)",
    )
    parser.add_argument(
        "--cover",
        "-c",
        type=Path,
        default=None,
        help="Path to cover art image (.jpg / .png)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to output MP4 video file",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory of chapter MP3 files to batch process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for batch-processed videos",
    )

    args = parser.parse_args()

    cover_path = args.cover
    if not cover_path:
        search_dirs = [Path("output/audiobook"), Path("output"), Path("input"), Path(".")]
        if args.audio_dir:
            search_dirs.insert(0, args.audio_dir)
        if args.audio:
            search_dirs.insert(0, args.audio.parent)
        cover_path = find_default_cover(search_dirs)

    if not cover_path or not cover_path.exists():
        print("[ERROR] Cover image not found. Please provide --cover path/to/cover.jpg")
        sys.exit(1)

    print("=" * 60)
    print("  Audiobook Video Generator — Scan2Speech")
    print("  Creator: Abderrahmane Bouamama")
    print(f"  Cover Art: {cover_path}")
    print("=" * 60)

    if args.audio:
        out_path = args.output or args.audio.with_suffix(".mp4")
        create_video_from_cover(args.audio, cover_path, out_path)

    elif args.audio_dir:
        out_dir = args.output_dir or args.audio_dir / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_files = sorted(args.audio_dir.glob("*.mp3"))
        if not audio_files:
            audio_files = sorted(args.audio_dir.glob("*.wav"))

        if not audio_files:
            print(f"[ERROR] No audio files found in {args.audio_dir}")
            sys.exit(1)

        print(f"Found {len(audio_files)} audio files to convert...")
        for af in audio_files:
            out_video = out_dir / f"{af.stem}.mp4"
            create_video_from_cover(af, cover_path, out_video)

    else:
        print("[ERROR] Please provide either --audio path/to/file.mp3 or --audio-dir path/to/chapters/")
        sys.exit(1)


if __name__ == "__main__":
    main()
