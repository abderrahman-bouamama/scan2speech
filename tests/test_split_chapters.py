"""Tests for split_chapters utility."""

from pathlib import Path
from split_chapters import split_by_headings, split_by_word_count, process_chapters


def test_split_by_headings():
    text = (
        "# Chapter 1: Beginnings\n"
        "Here is the story beginning.\n\n"
        "# Chapter 2: The Journey\n"
        "Continuing the journey forward."
    )
    chapters = split_by_headings(text)
    assert len(chapters) == 2
    assert "Chapter 1" in chapters[0][0]
    assert "Chapter 2" in chapters[1][0]


def test_split_by_word_count():
    text = "word " * 300
    chapters = split_by_word_count(text, target_words=100)
    assert len(chapters) == 3


def test_process_chapters_flow(tmp_path: Path):
    in_file = tmp_path / "full_book.txt"
    in_file.write_text(
        "# Section Alpha\nContent for alpha.\n\n# Section Beta\nContent for beta.",
        encoding="utf-8",
    )
    out_dir = tmp_path / "chapters_out"
    process_chapters(in_file, out_dir, mode="headings")

    files = list(out_dir.glob("*.txt"))
    # Includes chapter files plus master_audiobook_script.txt
    assert len(files) >= 3
    assert (out_dir / "master_audiobook_script.txt").exists()
