#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Studio Neural Audiobook Generator with Voice Cloning

Author: Abderrahmane Bouamama
Project: Scan2Speech

Generates professional, broadcast-quality audiobooks using Coqui XTTS-v2:
- Voice cloning from a reference audio file (.wav / .mp3)
- CUDA GPU acceleration (with automatic CPU fallback)
- Natural flowing speech without stutter, edge-clicks, or awkward silence gaps
- 120-character intelligent sentence chunking
- Smooth 5ms cosine fade ramps and digital silence trimming
- Broadcast-grade mastering with EBU R128 loudness normalization via FFmpeg
- Automated chapter concatenation and embedded ID3 metadata tags
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

# Force UTF-8 on Windows stdout/stderr
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Accept Coqui TOS non-interactively
os.environ["COQUI_TOS_AGREED"] = "1"

import numpy as np
import torch
from TTS.api import TTS


def get_ffmpeg_binary() -> str:
    """Find ffmpeg binary dynamically on Windows, Linux, or macOS."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            # WinGet packages or User Local Programs
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


def clean_narration_text(text: str) -> str:
    """Clean markdown and formatting artifacts while preserving language diacritics."""
    # Remove markdown headers (#, ##, etc.)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers (*, _, `)
    text = re.sub(r"[\*_`]", "", text)
    # Remove decorative lines (---, ===)
    text = re.sub(r"[-=]{3,}", "", text)
    # Replace repetitive dots (.. or ...) with comma or single period
    text = re.sub(r"\.{2,}", "،", text)
    # Normalize multiple whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_long_phrase(phrase: str, limit: int = 120) -> list[str]:
    """Split a phrase into pieces strictly not exceeding limit at word boundaries."""
    phrase = phrase.strip()
    if len(phrase) <= limit:
        return [phrase] if phrase else []

    words = phrase.split()
    pieces = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 <= limit:
            current = (current + " " + w).strip() if current else w
        else:
            if current:
                pieces.append(current)
            current = w
    if current:
        pieces.append(current)
    return pieces


def split_text_into_flowing_chunks(text: str, max_chars: int = 120) -> list[tuple[str, bool]]:
    """
    Split narration text into natural, complete sentences strictly <= max_chars.
    Returns list of (chunk_text, is_paragraph_end).
    """
    cleaned = clean_narration_text(text)
    paragraphs = [p.strip() for p in cleaned.splitlines() if p.strip()]

    chunks = []
    for p in paragraphs:
        # Split paragraph by full-stop sentence terminators (. or ؟ or ! or ?)
        sentences = re.split(r"(?<=[.؟!؟?])\s+", p)

        for s_idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue

            is_para_end = s_idx == len(sentences) - 1

            if len(sent) <= max_chars:
                chunks.append((sent, is_para_end))
            else:
                # Split at major punctuation marks: comma, semicolon, dash
                comma_parts = re.split(r"(?<=[،,؛;—\-])\s+", sent)
                for c_idx, part in enumerate(comma_parts):
                    part = part.strip()
                    if not part:
                        continue
                    pieces = split_long_phrase(part, limit=max_chars)
                    for piece_idx, piece in enumerate(pieces):
                        is_final = is_para_end and (c_idx == len(comma_parts) - 1) and (piece_idx == len(pieces) - 1)
                        chunks.append((piece, is_final))

    return chunks


def apply_edge_fades(samples: np.ndarray, fade_len: int = 120) -> np.ndarray:
    """Apply smooth 5ms cosine fade-in and fade-out to prevent digital edge clicks."""
    if len(samples) < fade_len * 2:
        return samples

    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
    fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))

    out = samples.copy()
    out[:fade_len] = (out[:fade_len] * fade_in).astype(np.int16)
    out[-fade_len:] = (out[-fade_len:] * fade_out).astype(np.int16)
    return out


def trim_silence_edges(samples: np.ndarray, threshold: int = 250, min_keep: int = 1200) -> np.ndarray:
    """Trim excessive digital silence padding produced by model at end of chunk."""
    if len(samples) <= min_keep:
        return samples

    abs_samples = np.abs(samples)
    non_silent = np.where(abs_samples > threshold)[0]
    if len(non_silent) > 0:
        last_sound = min(len(samples), non_silent[-1] + 600)  # keep ~25ms tail
        first_sound = max(0, non_silent[0] - 200)             # keep ~8ms lead
        return samples[first_sound:last_sound]
    return samples


def create_silence_wav(duration_ms: int, sample_rate: int = 24000) -> np.ndarray:
    """Generate silence buffer."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    return np.zeros(num_samples, dtype=np.int16)


def read_wav_samples(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read 16-bit PCM WAV samples."""
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        return samples, sample_rate


def write_wav_samples(wav_path: Path, samples: np.ndarray, sample_rate: int = 24000) -> None:
    """Write 16-bit PCM WAV samples."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def load_tts_model() -> TTS:
    """Initialize XTTS-v2 on GPU with fallback to CPU."""
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    print("=" * 60)
    print("  XTTS-v2 Studio Voice Cloning Pipeline")
    print(f"  PyTorch: {torch.__version__} | Device: {device.upper()}")
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    print("[OK] Model successfully loaded.\n")
    return tts


def synthesize_chapter(
    tts: TTS,
    chapter_num: int,
    script_path: Path,
    chapter_title: str,
    voice_path: Path,
    output_mp3_path: Path,
    cache_dir: Path,
    album_title: str = "Audiobook",
    author: str = "Abderrahmane Bouamama",
    language: str = "ar",
    speed: float = 1.0,
    test_mode: bool = False,
) -> bool:
    """Synthesize a chapter text file into a mastered MP3 with broadcast loudness."""
    if output_mp3_path.exists() and output_mp3_path.stat().st_size > 100000 and not test_mode:
        print(f"\n[OK] Chapter {chapter_num} already exists ({output_mp3_path.stat().st_size / (1024 * 1024):.2f} MB): {chapter_title}")
        print("  Skipping regeneration.")
        return True

    print(f"\n[RUN] Synthesizing Chapter {chapter_num}: {chapter_title}")

    if not script_path.exists():
        print(f"[ERROR] Script file not found: {script_path}")
        return False

    text = script_path.read_text(encoding="utf-8")
    chunks = split_text_into_flowing_chunks(text)

    if test_mode:
        chunks = chunks[:4]
        print(f"  [TEST MODE] Synthesizing first {len(chunks)} speech units.")

    print(f"  Total speech units: {len(chunks)}")

    chap_cache_dir = cache_dir / f"chapter_{chapter_num:02d}"
    chap_cache_dir.mkdir(parents=True, exist_ok=True)

    concatenated_samples = []
    target_sample_rate = 24000

    for idx, (chunk_text, is_para_end) in enumerate(chunks, start=1):
        chunk_wav_path = chap_cache_dir / f"chunk_{idx:04d}.wav"

        if chunk_wav_path.exists() and chunk_wav_path.stat().st_size > 1000:
            print(f"  [{idx}/{len(chunks)}] (Cached) {chunk_text[:45]}...")
        else:
            print(f"  [{idx}/{len(chunks)}] Synthesizing: {chunk_text[:50]}...")
            tts.tts_to_file(
                text=chunk_text,
                speaker_wav=str(voice_path),
                language=language,
                speed=speed,
                file_path=str(chunk_wav_path),
            )

        samples, sr = read_wav_samples(chunk_wav_path)
        target_sample_rate = sr

        trimmed = trim_silence_edges(samples)
        faded = apply_edge_fades(trimmed)
        concatenated_samples.append(faded)

        # 280ms pause at paragraph end, 60ms natural breath pause between sentences
        if is_para_end:
            concatenated_samples.append(create_silence_wav(280, target_sample_rate))
        else:
            concatenated_samples.append(create_silence_wav(60, target_sample_rate))

    merged_wav_path = chap_cache_dir / f"chapter_{chapter_num:02d}_merged.wav"
    all_samples = np.concatenate(concatenated_samples)
    write_wav_samples(merged_wav_path, all_samples, target_sample_rate)

    duration_sec = len(all_samples) / target_sample_rate
    print(f"  Raw speech duration: {duration_sec / 60:.2f} minutes.")

    # Master audio with EBU R128 loudness normalization
    print(f"  Mastering audio with EBU R128 normalization: {output_mp3_path.name}...")
    output_mp3_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = get_ffmpeg_binary()
    ffmpeg_cmd = [
        ffmpeg_bin, "-y",
        "-i", str(merged_wav_path),
        "-af", "highpass=f=60,lowpass=f=12000,loudnorm=I=-16:TP=-1.5:LRA=10",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-metadata", f"title={chapter_title}",
        "-metadata", f"artist={author}",
        "-metadata", f"album={album_title}",
        "-metadata", f"track={chapter_num}",
        "-metadata", "genre=Audiobook",
        str(output_mp3_path),
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        fallback_cmd = [
            ffmpeg_bin, "-y",
            "-i", str(merged_wav_path),
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            str(output_mp3_path),
        ]
        subprocess.run(fallback_cmd, check=True)

    print(f"[OK] Chapter {chapter_num} mastered successfully: {output_mp3_path}")
    return True


def merge_all_chapters(
    chapter_mp3s: list[Path],
    output_master_mp3: Path,
    title: str = "Audiobook",
    author: str = "Abderrahmane Bouamama",
) -> None:
    """Concatenate generated chapter MP3s into the master audiobook."""
    print("\n" + "=" * 60)
    print("  Merging all chapters into Full Master Audiobook...")
    print("=" * 60)

    ffmpeg_bin = get_ffmpeg_binary()
    concat_list_file = output_master_mp3.parent / "chapters_list.txt"

    with open(concat_list_file, "w", encoding="utf-8") as f:
        for mp3 in chapter_mp3s:
            clean_path = str(mp3.resolve()).replace("\\", "/")
            f.write(f"file '{clean_path}'\n")

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        "-metadata", f"title={title} (Complete Audiobook)",
        "-metadata", f"artist={author}",
        "-metadata", f"album={title}",
        "-metadata", "genre=Audiobook",
        str(output_master_mp3),
    ]

    subprocess.run(cmd, check=True)
    size_mb = output_master_mp3.stat().st_size / (1024 * 1024)
    print(f"\n[DONE] Master Audiobook completed: {output_master_mp3} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="XTTS-v2 Neural Voice Cloning Audiobook Generator. Created by Abderrahmane Bouamama."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=None,
        help="Directory containing chapter .txt files",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Path to a single narration text file",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=Path,
        required=True,
        help="Path to reference speaker audio file (.wav or .mp3)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("output/audiobook"),
        help="Destination directory for audiobook MP3s (default: output/audiobook)",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="Audiobook",
        help="Audiobook title (for metadata tags and file naming)",
    )
    parser.add_argument(
        "--author",
        "-a",
        type=str,
        default="Abderrahmane Bouamama",
        help="Author / narrator name (default: Abderrahmane Bouamama)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="ar",
        help="Target language code (e.g. ar, en, fr, es, de, it, ja)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Synthesize a specific chapter number (1-based index)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Synthesize all chapters and merge into master audiobook",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Concatenate existing chapter MP3s without re-synthesizing",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Synthesize quick audition sample (first 4 sentences)",
    )

    args = parser.parse_args()

    if not args.voice.exists():
        print(f"[ERROR] Voice reference file not found: {args.voice}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chapters_dir = args.output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sanitized_title = re.sub(r"[^a-zA-Z0-9_\-]+", "_", args.title)
    master_mp3 = args.output_dir / f"{sanitized_title}_master.mp3"

    if args.merge_only:
        existing_mp3s = sorted(chapters_dir.glob("chapter_*.mp3"))
        if not existing_mp3s:
            print(f"[ERROR] No chapter MP3s found in {chapters_dir}")
            sys.exit(1)
        merge_all_chapters(existing_mp3s, master_mp3, title=args.title, author=args.author)
        return

    # Discover chapter files
    chapter_files: list[Path] = []
    if args.input_file:
        chapter_files = [args.input_file]
    elif args.input_dir:
        chapter_files = sorted(
            [f for f in args.input_dir.glob("*.txt") if not f.name.startswith("master_")],
            key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0,
        )

    if not chapter_files:
        print(f"[ERROR] No text files found in specified input ({args.input_dir or args.input_file})")
        sys.exit(1)

    print("=" * 60)
    print("  Studio Audiobook Synthesis — Scan2Speech")
    print(f"  Creator: {args.author}")
    print(f"  Title  : {args.title}")
    print(f"  Voice  : {args.voice.name}")
    print(f"  Chapters: {len(chapter_files)} files")
    print("=" * 60)

    tts = load_tts_model()

    if args.test:
        test_out = args.output_dir / "test_audition_sample.mp3"
        synthesize_chapter(
            tts=tts,
            chapter_num=1,
            script_path=chapter_files[0],
            chapter_title="Test Audition",
            voice_path=args.voice,
            output_mp3_path=test_out,
            cache_dir=cache_dir,
            album_title=args.title,
            author=args.author,
            language=args.language,
            speed=args.speed,
            test_mode=True,
        )
        print(f"\n[DONE] Test audition sample created: {test_out}")
        return

    generated_mp3s = []

    if args.chapter:
        c_num = args.chapter
        if c_num < 1 or c_num > len(chapter_files):
            print(f"[ERROR] Chapter {c_num} out of range (1 to {len(chapter_files)})")
            sys.exit(1)
        c_file = chapter_files[c_num - 1]
        out_mp3 = chapters_dir / f"chapter_{c_num:02d}.mp3"
        synthesize_chapter(
            tts=tts,
            chapter_num=c_num,
            script_path=c_file,
            chapter_title=f"Chapter {c_num}: {c_file.stem}",
            voice_path=args.voice,
            output_mp3_path=out_mp3,
            cache_dir=cache_dir,
            album_title=args.title,
            author=args.author,
            language=args.language,
            speed=args.speed,
        )
    elif args.all:
        for c_num, c_file in enumerate(chapter_files, start=1):
            out_mp3 = chapters_dir / f"chapter_{c_num:02d}.mp3"
            success = synthesize_chapter(
                tts=tts,
                chapter_num=c_num,
                script_path=c_file,
                chapter_title=f"Chapter {c_num}: {c_file.stem}",
                voice_path=args.voice,
                output_mp3_path=out_mp3,
                cache_dir=cache_dir,
                album_title=args.title,
                author=args.author,
                language=args.language,
                speed=args.speed,
            )
            if success:
                generated_mp3s.append(out_mp3)

        if generated_mp3s:
            merge_all_chapters(generated_mp3s, master_mp3, title=args.title, author=args.author)
    else:
        # Default to synthesizing chapter 1 or audition
        out_mp3 = chapters_dir / "chapter_01.mp3"
        synthesize_chapter(
            tts=tts,
            chapter_num=1,
            script_path=chapter_files[0],
            chapter_title=f"Chapter 1: {chapter_files[0].stem}",
            voice_path=args.voice,
            output_mp3_path=out_mp3,
            cache_dir=cache_dir,
            album_title=args.title,
            author=args.author,
            language=args.language,
            speed=args.speed,
        )


if __name__ == "__main__":
    main()
