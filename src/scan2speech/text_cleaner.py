"""
Text Cleaner — two-phase cleaning:
  1. Per-page cleaning  (called immediately after each API transcription)
  2. Final stitching     (joins all per-page .txt files into one TTS-ready doc)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from scan2speech.config import PipelineConfig

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans raw Gemini transcripts and stitches them into a TTS-ready document.

    Per-page cleaning (``clean_page``):
    1. Strip standalone page numbers.
    2. Rejoin hyphenated words split across lines.
    3. Normalize punctuation (smart quotes → ASCII, etc.).
    4. Collapse excessive whitespace.

    Final stitching (``stitch_from_files``):
    1. Detect and remove repeated running headers/footers across pages.
    2. Filter out blank pages.
    3. Join pages with double newlines (TTS pause cues).
    4. Final whitespace normalization.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    """

    # Matches a line containing only digits (with optional surrounding space)
    _PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

    # Matches word-ending hyphen followed by a newline and the continuation
    _HYPHEN_RE = re.compile(r"(\w)-\n(\w)")

    # Three or more consecutive newlines → collapsed to two
    _EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

    # Two or more consecutive spaces → collapsed to one
    _EXCESS_SPACES_RE = re.compile(r" {2,}")

    # Smart-quote and typographic character replacements
    _SMART_QUOTE_MAP: dict[str, str] = {
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": " - ", # em dash → spaced hyphen
        "\u2026": "...", # ellipsis
    }

    # Blank-page sentinel emitted by the model
    _BLANK_PAGE = "[BLANK PAGE]"

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Phase 1: Per-page cleaning (called after each image transcription)
    # ------------------------------------------------------------------

    def clean_page(self, raw_text: str) -> str:
        """
        Clean a single page's raw transcript from the API.

        Applied in order:
        1. Strip standalone page numbers.
        2. Rejoin hyphenated words.
        3. Normalize smart quotes / typographic chars.
        4. Collapse excess whitespace.

        Parameters
        ----------
        raw_text : str
            Raw transcript text from the Gemini API.

        Returns
        -------
        str
            Cleaned per-page text.
        """
        if not raw_text or raw_text.strip() == self._BLANK_PAGE:
            return raw_text.strip()

        text = raw_text

        # 1. Remove standalone page numbers
        text = self._PAGE_NUMBER_RE.sub("", text)

        # 2. Rejoin hyphenated words split across lines
        text = self._rejoin_hyphens(text)

        # 3. Normalize punctuation
        text = self._normalize_punctuation(text)

        # 4. Collapse whitespace
        text = self._normalize_whitespace(text)

        return text

    # ------------------------------------------------------------------
    # Phase 2: Final stitching (all per-page .txt → one output)
    # ------------------------------------------------------------------

    def stitch_from_files(
        self, transcript_dir: Path, page_nums: list[int]
    ) -> str:
        """
        Load per-page transcript files and stitch them into a single
        TTS-ready document.

        Parameters
        ----------
        transcript_dir : Path
            Directory containing ``page_NNNN.txt`` files.
        page_nums : list[int]
            Page numbers to include.

        Returns
        -------
        str
            Final stitched and cleaned text.
        """
        transcripts: list[tuple[int, str]] = []
        for pn in sorted(page_nums):
            path = transcript_dir / f"page_{pn:04d}.txt"
            if path.exists():
                text = path.read_text(encoding="utf-8")
                transcripts.append((pn, text))
            else:
                logger.warning("Transcript file missing: %s", path)

        return self._stitch(transcripts)

    def _stitch(self, transcripts: list[tuple[int, str]]) -> str:
        """
        Join per-page transcripts into a single coherent text.

        Parameters
        ----------
        transcripts : list[tuple[int, str]]
            List of (page_number, cleaned_text) pairs.

        Returns
        -------
        str
            The final stitched text ready for TTS.
        """
        # Sort by page number to guarantee correct order
        sorted_transcripts = sorted(transcripts, key=lambda t: t[0])

        # Filter out blank pages
        texts = [
            text for _, text in sorted_transcripts
            if text.strip() and text.strip() != self._BLANK_PAGE
        ]

        if not texts:
            logger.warning("All pages were blank — output will be empty")
            return ""

        # Detect repeated headers/footers across pages
        header_footer_lines = self._detect_repeated_lines(texts)

        # Remove detected headers/footers from each page
        if header_footer_lines:
            texts = [
                self._remove_lines(text, header_footer_lines)
                for text in texts
            ]

        # Join pages with double newlines (paragraph separator / TTS pause)
        joined = "\n\n".join(page for page in texts if page.strip())

        # Final global pass
        joined = self._normalize_whitespace(joined)

        logger.info(
            "Stitched %d pages into %d characters", len(texts), len(joined)
        )
        return joined

    # ------------------------------------------------------------------
    # Also keep clean_and_stitch for backward compatibility / direct use
    # ------------------------------------------------------------------

    def clean_and_stitch(self, transcripts: list[tuple[int, str]]) -> str:
        """
        Clean and stitch raw transcripts (combines per-page cleaning + stitch).

        Parameters
        ----------
        transcripts : list[tuple[int, str]]
            List of (page_number, raw_text) pairs.

        Returns
        -------
        str
            Cleaned, stitched text.
        """
        cleaned = [(pn, self.clean_page(text)) for pn, text in transcripts]
        return self._stitch(cleaned)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_repeated_lines(
        self, page_texts: list[str], threshold: float = 0.3
    ) -> set[str]:
        """
        Detect lines that appear across a significant fraction of pages —
        these are likely running headers or footers.

        Parameters
        ----------
        page_texts : list[str]
            Cleaned text for each page.
        threshold : float
            Fraction of pages a line must appear in to be flagged.

        Returns
        -------
        set[str]
            Set of line contents identified as headers/footers.
        """
        if len(page_texts) < 3:
            return set()

        # Count how many pages each unique trimmed line appears in
        line_counter: Counter[str] = Counter()
        for text in page_texts:
            unique_lines = {ln.strip() for ln in text.split("\n") if ln.strip()}
            for ln in unique_lines:
                line_counter[ln] += 1

        min_occurrences = max(3, int(len(page_texts) * threshold))
        repeated = {
            ln for ln, count in line_counter.items()
            if count >= min_occurrences and len(ln) > 1
        }

        if repeated:
            logger.info("Detected %d repeated header/footer lines", len(repeated))
            logger.debug("Repeated lines: %s", repeated)

        return repeated

    @staticmethod
    def _remove_lines(text: str, lines_to_remove: set[str]) -> str:
        """Remove lines whose stripped content matches the given set."""
        return "\n".join(
            ln for ln in text.split("\n")
            if ln.strip() not in lines_to_remove
        ).strip()

    @classmethod
    def _rejoin_hyphens(cls, text: str) -> str:
        """Rejoin words that were hyphenated across line breaks."""
        return cls._HYPHEN_RE.sub(r"\1\2", text)

    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        """Collapse excessive blank lines and spaces."""
        text = cls._EXCESS_NEWLINES_RE.sub("\n\n", text)
        text = cls._EXCESS_SPACES_RE.sub(" ", text)
        return text.strip()

    @classmethod
    def _normalize_punctuation(cls, text: str) -> str:
        """Replace smart quotes and typographic characters with ASCII."""
        for orig, replacement in cls._SMART_QUOTE_MAP.items():
            text = text.replace(orig, replacement)
        return text
