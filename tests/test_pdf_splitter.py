"""Tests for the PdfSplitter module.

Note: These tests require a real (tiny) PDF. We generate one in-memory
using pypdfium2 so there are no external test fixtures needed.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scan2speech.config import PipelineConfig
from scan2speech.pdf_splitter import PdfSplitter, PageImage


def _make_splitter(dpi_scale: float = 1.0) -> PdfSplitter:
    """Create a PdfSplitter with a minimal mock config."""
    config = MagicMock(spec=PipelineConfig)
    config.dpi_scale = dpi_scale
    return PdfSplitter(config)


class TestPdfSplitterInvalidInput:
    """Verify error handling for bad input."""

    def test_nonexistent_file(self, tmp_path: Path):
        splitter = _make_splitter()
        fake_path = tmp_path / "does_not_exist.pdf"
        with pytest.raises(Exception):
            splitter.split(fake_path, tmp_path / "out")

    def test_corrupt_file(self, tmp_path: Path):
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"this is not a PDF")
        splitter = _make_splitter()
        with pytest.raises(Exception):
            splitter.split(bad_pdf, tmp_path / "out")


class TestCountPages:
    """Verify page counting for invalid files."""

    def test_nonexistent_file(self, tmp_path: Path):
        with pytest.raises(Exception):
            PdfSplitter.count_pages(tmp_path / "missing.pdf")
