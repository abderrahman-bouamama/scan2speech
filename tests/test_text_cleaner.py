"""Tests for the TextCleaner module."""

from unittest.mock import MagicMock

from scan2speech.text_cleaner import TextCleaner
from scan2speech.config import PipelineConfig


def _make_cleaner() -> TextCleaner:
    """Create a TextCleaner with a minimal mock config."""
    config = MagicMock(spec=PipelineConfig)
    return TextCleaner(config)


# ======================================================================
# Per-page cleaning (clean_page)
# ======================================================================

class TestCleanPageHyphens:
    """Verify that hyphenated words split across lines are rejoined."""

    def test_simple_hyphen(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("This is a compre-\nhensive test.")
        assert "comprehensive" in result

    def test_multiple_hyphens(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("The under-\nstanding of multi-\ncolumn layouts.")
        assert "understanding" in result
        assert "multicolumn" in result

    def test_no_hyphens(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("No hyphens here at all.")
        assert result == "No hyphens here at all."


class TestCleanPageNumbers:
    """Verify standalone page numbers are stripped."""

    def test_removes_page_number(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("Some text\n42\nMore text")
        assert "42" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_preserves_numbers_in_text(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("There are 42 items in the list.")
        assert "42" in result


class TestCleanPagePunctuation:
    """Verify typographic characters are normalized to ASCII."""

    def test_smart_quotes(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("\u201cHello,\u201d she said, \u2018world.\u2019")
        assert result == '"Hello," she said, \'world.\''

    def test_em_dash(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("word\u2014another")
        assert result == "word - another"

    def test_ellipsis(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("wait\u2026")
        assert result == "wait..."


class TestCleanPageWhitespace:
    """Verify excessive whitespace is collapsed."""

    def test_excess_newlines(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("Para one.\n\n\n\n\nPara two.")
        assert result == "Para one.\n\nPara two."

    def test_excess_spaces(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("Too   many    spaces.")
        assert result == "Too many spaces."


class TestCleanPageBlank:
    """Verify blank page sentinel is preserved as-is."""

    def test_blank_page(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("[BLANK PAGE]")
        assert result == "[BLANK PAGE]"

    def test_empty_string(self):
        cleaner = _make_cleaner()
        result = cleaner.clean_page("")
        assert result == ""


# ======================================================================
# Stitching (clean_and_stitch — combines per-page clean + stitch)
# ======================================================================

class TestStitchHeaderFooterDetection:
    """Verify that repeated lines across pages are detected."""

    def test_detects_repeated_header(self):
        cleaner = _make_cleaner()
        transcripts = [
            (1, "Chapter Title\nSome body text on page one."),
            (2, "Chapter Title\nBody text on page two."),
            (3, "Chapter Title\nBody text on page three."),
            (4, "Chapter Title\nBody text on page four."),
        ]
        result = cleaner.clean_and_stitch(transcripts)
        assert "Chapter Title" not in result

    def test_ignores_unique_lines(self):
        cleaner = _make_cleaner()
        transcripts = [
            (1, "Unique line one.\nBody text."),
            (2, "Unique line two.\nMore text."),
            (3, "Unique line three.\nEven more text."),
        ]
        result = cleaner.clean_and_stitch(transcripts)
        assert "Unique line one" in result


class TestStitchBlankPages:
    """Verify blank pages are filtered out during stitching."""

    def test_blank_pages_excluded(self):
        cleaner = _make_cleaner()
        transcripts = [
            (1, "Real content here."),
            (2, "[BLANK PAGE]"),
            (3, "More real content."),
        ]
        result = cleaner.clean_and_stitch(transcripts)
        assert "[BLANK PAGE]" not in result
        assert "Real content" in result
        assert "More real content" in result

    def test_all_blank(self):
        cleaner = _make_cleaner()
        transcripts = [
            (1, "[BLANK PAGE]"),
            (2, "[BLANK PAGE]"),
        ]
        result = cleaner.clean_and_stitch(transcripts)
        assert result == ""


class TestStitchingOrder:
    """Verify pages are stitched in correct order regardless of input order."""

    def test_out_of_order_input(self):
        cleaner = _make_cleaner()
        transcripts = [
            (3, "Third."),
            (1, "First."),
            (2, "Second."),
        ]
        result = cleaner.clean_and_stitch(transcripts)
        first_pos = result.index("First")
        second_pos = result.index("Second")
        third_pos = result.index("Third")
        assert first_pos < second_pos < third_pos
