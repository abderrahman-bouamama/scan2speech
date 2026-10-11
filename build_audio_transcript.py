#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Curated Audiobook / TTS Transcript

Author: Abderrahmane Bouamama
Project: Scan2Speech

Stitches per-page OCR text files into a clean, continuous, TTS-ready narration script:
- Strips running headers, repeated titles, and standalone page numbers
- Removes footnote separator lines and bottom footnotes
- Cleans inline footnote reference markers: (1), [2], (*), (١), etc.
- Rejoins hyphenated words and sentences broken across page boundaries
- Generates structured narration with clear paragraph breaks
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def is_running_header(line: str, repeated_patterns: set[str] | None = None) -> bool:
    """Check if a line matches running header, divider, or page number patterns."""
    s = line.strip()
    if not s:
        return False

    # 1. Pure dashes, underscores, dots, or decoration rules
    if re.match(r"^[-_ـ\.\s=~*]{3,}$", s):
        return True

    # 2. Standalone page number (Western or Arabic-Indic numerals)
    if re.match(r"^[0-9٠-٩]+$", s):
        return True

    # 3. Contains long dash sequence (header rule)
    if re.search(r"[-_ـ]{4,}", s):
        return True

    # 4. Ends with 3+ dashes and optional number
    if re.search(r"[-_ـ]{3,}\s*[0-9٠-٩]*$", s):
        return True

    # 5. Starts with page number followed by common header pattern
    if re.match(r"^[0-9٠-٩]+\s*[-/|•·]\s*", s):
        return True

    # 6. Matches user-specified repeated header pattern
    if repeated_patterns and s in repeated_patterns:
        return True

    return False


def clean_page_text(
    raw_text: str,
    strip_headers: bool = True,
    strip_footnotes: bool = True,
    strip_inline_markers: bool = True,
    repeated_headers: set[str] | None = None,
) -> tuple[str, bool]:
    """
    Clean an individual page transcript:
    - Strips running headers and page numbers from the top
    - Strips footnotes and footnote separators from the bottom
    - Strips inline footnote reference numbers
    - Preserves body sentences and paragraph boundaries
    - Returns (cleaned_body, ends_mid_sentence)
    """
    if not raw_text or raw_text.strip() == "[BLANK PAGE]":
        return "", False

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return "", False

    # 1. Strip running headers from top of page
    if strip_headers:
        while lines and is_running_header(lines[0], repeated_headers):
            lines.pop(0)

    if not lines:
        return "", False

    # 2. Strip footnotes from bottom of page
    if strip_footnotes:
        body_lines = []
        fn_re = re.compile(r"^[\(\[\{][0-9٠-٩\*]+[\)\]\}]")
        sep_re = re.compile(r"^[-_ـ\.]{3,}$")
        in_footnotes = False

        for idx, line in enumerate(lines):
            # Divider line indicates start of footnote block
            if not in_footnotes and idx >= 1 and sep_re.match(line):
                in_footnotes = True
                continue

            # Numbered footnote entry at bottom
            if not in_footnotes and idx >= 1 and fn_re.match(line):
                in_footnotes = True
                continue

            if not in_footnotes:
                body_lines.append(line)
        lines = body_lines

    if not lines:
        return "", False

    raw_joined = "\n".join(lines)

    # 3. Strip inline footnote markers: (1), [2], (*), (١), etc.
    if strip_inline_markers:
        raw_joined = re.sub(r"[\(\[]\s*[0-9٠-٩\*]+\s*[\)\]]", "", raw_joined)

    # 4. Normalize spacing
    raw_joined = re.sub(r"[ ]{2,}", " ", raw_joined)
    cleaned_lines = [l.strip() for l in raw_joined.splitlines() if l.strip()]
    body = "\n".join(cleaned_lines).strip()

    # 5. Check if page ends mid-sentence (to seamlessly rejoin with next page)
    last_char = body[-1] if body else ""
    terminal_punct = {".", "!", "؟", "?", ":", "؛", ";", "»", '"', "…", "—", "ـ"}
    ends_mid_sentence = bool(body and last_char not in terminal_punct)

    return body, ends_mid_sentence


def build_transcript(
    input_dir: Path,
    output_file: Path | None = None,
    title: str | None = None,
    author: str = "Abderrahmane Bouamama",
    start_page: int | None = None,
    end_page: int | None = None,
    rejoin_boundaries: bool = True,
) -> dict:
    """Build a curated audiobook transcript from a directory of page text files."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Discover and sort page text files naturally (page_0001.txt, etc.)
    page_files = sorted(
        input_dir.glob("page_*.txt"),
        key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0,
    )

    if not page_files:
        # Fallback to any .txt files if page_*.txt not found
        page_files = sorted(input_dir.glob("*.txt"))

    if not page_files:
        raise FileNotFoundError(f"No text files found in {input_dir}")

    doc_title = title or input_dir.name
    if not output_file:
        output_file = input_dir / f"{input_dir.name}_tts_transcript.txt"

    full_output = []

    # Header banner
    full_output.append(
        "=" * 70 + "\n"
        f"  Title  : {doc_title}\n"
        f"  Author : {author}\n"
        "=" * 70 + "\n\n"
        "Curated Audio Transcript — Optimized for Text-to-Speech (TTS) Narration.\n"
        "Headers, footers, footnote references, and pagination rules removed.\n\n"
    )

    processed_count = 0
    body_parts = []
    prev_mid_sentence = False

    for p_file in page_files:
        match = re.search(r"\d+", p_file.stem)
        page_num = int(match.group()) if match else None

        if page_num is not None:
            if start_page is not None and page_num < start_page:
                continue
            if end_page is not None and page_num > end_page:
                continue

        raw = p_file.read_text(encoding="utf-8")
        cleaned, ends_mid = clean_page_text(raw)
        if not cleaned:
            continue

        processed_count += 1

        if body_parts and prev_mid_sentence and rejoin_boundaries:
            # Rejoin across page boundary
            last_part = body_parts[-1]
            if last_part.endswith("-") or last_part.endswith("ـ"):
                body_parts[-1] = last_part[:-1] + cleaned
            else:
                body_parts[-1] = last_part + " " + cleaned
        else:
            body_parts.append(cleaned)

        prev_mid_sentence = ends_mid

    full_body = "\n\n".join(body_parts)
    full_body = re.sub(r"\n{3,}", "\n\n", full_body)
    full_output.append(full_body)

    final_text = "".join(full_output).strip() + "\n"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(final_text, encoding="utf-8")

    words = final_text.split()
    total_words = len(words)
    total_chars = len(final_text)
    est_duration_mins = total_words / 130.0

    stats = {
        "title": doc_title,
        "author": author,
        "total_pages_processed": processed_count,
        "total_words": total_words,
        "total_chars": total_chars,
        "estimated_duration_minutes": round(est_duration_mins, 1),
        "output_file": str(output_file),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build curated TTS transcript from scanned page text files. Created by Abderrahmane Bouamama."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing page text files (e.g., output/my_book/)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=None,
        help="Destination path for curated transcript (default: <input-dir>/<name>_tts_transcript.txt)",
    )
    parser.add_argument(
        "--title",
        "-t",
        type=str,
        default=None,
        help="Title of the document/audiobook",
    )
    parser.add_argument(
        "--author",
        "-a",
        type=str,
        default="Abderrahmane Bouamama",
        help="Author or creator name (default: Abderrahmane Bouamama)",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="First page number to include",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Last page number to include",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Curated Audio Transcript Builder — Scan2Speech")
    print(f"  Creator: {args.author}")
    print("=" * 60)

    try:
        stats = build_transcript(
            input_dir=args.input_dir,
            output_file=args.output_file,
            title=args.title,
            author=args.author,
            start_page=args.start_page,
            end_page=args.end_page,
        )
        print("\n[SUCCESS] Transcript built successfully:")
        for k, v in stats.items():
            print(f"  • {k}: {v}")
    except Exception as e:
        print(f"\n[ERROR] Failed to build transcript: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
