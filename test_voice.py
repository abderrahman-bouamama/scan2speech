#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audition Cloned Voice Sample with XTTS-v2

Author: Abderrahmane Bouamama
Project: Scan2Speech

Quickly generates a short audition sample using Coqui XTTS-v2 to evaluate
cloned voice fidelity and naturalness before launching full audiobook generation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["COQUI_TOS_AGREED"] = "1"

import torch
from TTS.api import TTS

DEFAULT_ARABIC_SAMPLE = (
    "مَرْحَبًا بِكَ فِي هَذَا التَّسْجِيلِ الصَّوْتِيِّ التَّجْرِيبِيِّ، "
    "حَيْثُ نَخْتَبِرُ نَقَاءَ وَفَصَاحَةَ الصَّوْتِ الْمُسْتَنْسَخِ لِإِنْتَاجِ الْكُتُبِ الصَّوْتِيَّةِ."
)

DEFAULT_ENGLISH_SAMPLE = (
    "Welcome to this audio test sample. We are auditioning the neural voice clone "
    "for high-fidelity audiobook narration and automated documentary production."
)


def main():
    parser = argparse.ArgumentParser(
        description="Audition cloned voice using XTTS-v2. Created by Abderrahmane Bouamama."
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=Path,
        required=True,
        help="Path to clean reference voice audio (.wav / .mp3)",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Custom text to synthesize for audition",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="ar",
        help="Language code (ar, en, fr, es, de, it, ja, etc. Default: ar)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/audition_sample.wav"),
        help="Output audio path (default: output/audition_sample.wav)",
    )

    args = parser.parse_args()

    if not args.voice.exists():
        print(f"[ERROR] Voice reference file not found: {args.voice}")
        sys.exit(1)

    text_to_speak = args.text
    if not text_to_speak:
        text_to_speak = DEFAULT_ARABIC_SAMPLE if args.language == "ar" else DEFAULT_ENGLISH_SAMPLE

    args.output.parent.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"

    print("=" * 60)
    print("  XTTS-v2 Voice Audition Tool — Scan2Speech")
    print("  Creator: Abderrahmane Bouamama")
    print(f"  Device : {device.upper()} (CUDA: {has_cuda})")
    print(f"  Voice  : {args.voice.name}")
    print(f"  Lang   : {args.language}")
    print("=" * 60)

    print("[RUN] Loading XTTS-v2 model...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print(f"[RUN] Synthesizing audition sample:\n  \"{text_to_speak}\"...")
    tts.tts_to_file(
        text=text_to_speak,
        speaker_wav=str(args.voice),
        language=args.language,
        file_path=str(args.output),
        split_sentences=False,
    )

    print(f"\n[OK] Audition sample created: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
