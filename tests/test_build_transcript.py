"""Tests for build_audio_transcript utility."""

import pytest
from pathlib import Path
from build_audio_transcript import clean_page_text, is_running_header, build_transcript


def test_is_running_header():
    assert is_running_header("---") is True
    assert is_running_header("===") is True
    assert is_running_header("42") is True
    assert is_running_header("٤٢") is True
    assert is_running_header("Normal paragraph sentence.") is False


def test_clean_page_text():
    raw = (
        "--- Header Rule ---\n"
        "This is paragraph one.\n\n"
        "This is paragraph two (1).\n"
        "---\n"
        "(1) Footnote explanation"
    )
    cleaned, ends_mid = clean_page_text(raw)
    assert "This is paragraph one." in cleaned
    assert "This is paragraph two ." in cleaned or "This is paragraph two." in cleaned
    assert "Footnote explanation" not in cleaned
    assert ends_mid is False


def test_build_transcript_flow(tmp_path: Path):
    in_dir = tmp_path / "book_pages"
    in_dir.mkdir()

    (in_dir / "page_0001.txt").write_text("Introduction to the first chapter.", encoding="utf-8")
    (in_dir / "page_0002.txt").write_text("Second page with more details.", encoding="utf-8")

    out_file = tmp_path / "curated_transcript.txt"
    stats = build_transcript(
        input_dir=in_dir,
        output_file=out_file,
        title="Test Book",
        author="Abderrahmane Bouamama",
    )

    assert stats["total_pages_processed"] == 2
    assert stats["author"] == "Abderrahmane Bouamama"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Test Book" in content
    assert "Abderrahmane Bouamama" in content
    assert "Introduction to the first chapter." in content
