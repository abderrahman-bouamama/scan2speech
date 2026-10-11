#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split Long Document / Transcript into Numbered Audio Chapters

Author: Abderrahmane Bouamama
Project: Scan2Speech

Takes a full text file (from Scan2Speech or any document) and splits it into
numbered chapter files (`01_chapter.txt`, `02_chapter.txt`, ...) suitable
for modular Text-to-Speech synthesis.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def split_by_headings(text: str, custom_pattern: str | None = None) -> list[tuple[str, str]]:
    """
    Split text by Markdown headers (# Header), Chapter headings (Chapter 1 / الفصل الأول),
    or custom regex pattern.
    Returns list of (chapter_title, chapter_text).
    """
    default_pattern = r"(?:^|\n)(?=#{1,3}\s+|(?i:chapter|section|part)\s+\d+|الفصل\s+[\w\d]+|المبحث\s+[\w\d]+)"
    pattern = custom_pattern if custom_pattern else default_pattern

    raw_splits = re.split(pattern, text)
    chapters = []

    for idx, part in enumerate(raw_splits):
        content = part.strip()
        if not content:
            continue

        # Extract title from first line
        first_line = content.splitlines()[0].strip()
        title = re.sub(r"^[#=\-\*\s]+", "", first_line).strip()
        if len(title) > 60:
            title = title[:60] + "..."

        chapters.append((title or f"Chapter {len(chapters) + 1}", content))

    return chapters


def split_by_word_count(text: str, target_words: int = 1000) -> list[tuple[str, str]]:
    """Split text into uniform chapter blocks based on word count."""
    words = text.split()
    if not words:
        return []

    chapters = []
    chap_idx = 1
    for i in range(0, len(words), target_words):
        chunk_words = words[i:i + target_words]
        chunk_text = " ".join(chunk_words)
        chapters.append((f"Section {chap_idx:02d}", chunk_text))
        chap_idx += 1

    return chapters


def process_chapters(
    input_file: Path,
    output_dir: Path,
    mode: str = "auto",
    target_words: int = 1000,
    heading_pattern: str | None = None,
) -> None:
    """Split document into numbered chapter files and compute audio statistics."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    text = input_file.read_text(encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "words":
        chapters = split_by_word_count(text, target_words=target_words)
    elif mode == "headings":
        chapters = split_by_headings(text, custom_pattern=heading_pattern)
    else:  # auto
        heading_splits = split_by_headings(text, custom_pattern=heading_pattern)
        if len(heading_splits) > 1:
            chapters = heading_splits
        else:
            chapters = split_by_word_count(text, target_words=target_words)

    print("=" * 65)
    print(f"  Splitting Document into Chapters: {input_file.name}")
    print(f"  Target Output Directory: {output_dir}")
    print("=" * 65)

    total_words = 0
    total_chars = 0
    master_lines = []

    for idx, (title, content) in enumerate(chapters, start=1):
        sanitized_title = re.sub(r"[^a-zA-Z0-9_\-\u0600-\u06FF]+", "_", title)[:35].strip("_")
        filename = f"{idx:02d}_{sanitized_title}.txt"
        file_path = output_dir / filename

        file_path.write_text(content.strip() + "\n", encoding="utf-8")

        words = len(content.split())
        chars = len(content)
        total_words += words
        total_chars += chars
        est_mins = words / 130.0

        print(f"  ✓ {filename:<40} : {words:5d} words | {chars:6d} chars | ~{est_mins:4.1f} min audio")
        master_lines.append(f"=== {filename} ===\n{content}")

    master_file = output_dir / "master_audiobook_script.txt"
    master_file.write_text("\n\n".join(master_lines) + "\n", encoding="utf-8")

    est_total_mins = total_words / 130.0
    print("=" * 65)
    print(f"  Chapters Generated : {len(chapters)}")
    print(f"  Total Words        : {total_words}")
    print(f"  Total Characters   : {total_chars}")
    print(f"  Est. Audio Duration: ~{est_total_mins:.1f} minutes ({int(est_total_mins)}m {int((est_total_mins%1)*60)}s)")
    print(f"  Master Script Saved: {master_file.name}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Split full text into chapter files for audiobook narration. Created by Abderrahmane Bouamama."
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=Path,
        required=True,
        help="Path to full text or transcript file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to save numbered chapter files",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "headings", "words"],
        default="auto",
        help="Splitting strategy: 'auto' (headings if found, else words), 'headings', or 'words'",
    )
    parser.add_argument(
        "--words-per-chapter",
        type=int,
        default=1000,
        help="Target word count when splitting by words (default: 1000 = ~7.5 mins audio)",
    )
    parser.add_argument(
        "--heading-pattern",
        type=str,
        default=None,
        help="Custom regex delimiter to identify chapter headings",
    )

    args = parser.parse_args()
    try:
        process_chapters(
            input_file=args.input_file,
            output_dir=args.output_dir,
            mode=args.mode,
            target_words=args.words_per_chapter,
            heading_pattern=args.heading_pattern,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
