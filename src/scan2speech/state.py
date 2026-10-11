"""
Job State — persists pipeline progress to a JSON manifest for resume-on-failure.

The state file (``.scan2speech_state.json``) is written inside the work
directory and tracks which pages have been split, transcribed, and cleaned.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from scan2speech import StateCorruptedError

logger = logging.getLogger(__name__)

STATE_FILENAME = ".scan2speech_state.json"
STATE_VERSION = 1


class JobState:
    """
    Tracks and persists pipeline progress for a single PDF processing job.

    Parameters
    ----------
    work_dir : Path
        Directory where the state file lives.
    """

    def __init__(self, work_dir: Path) -> None:
        self._path = work_dir / STATE_FILENAME
        self._data: dict = {}

    # ------------------------------------------------------------------
    # Load / create
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Check whether a state file already exists on disk."""
        return self._path.is_file()

    def load(self) -> None:
        """
        Load an existing state file from disk.

        Raises
        ------
        StateCorruptedError
            If the file cannot be parsed or has an incompatible version.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
            self._data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            raise StateCorruptedError(
                f"Cannot read state file: {self._path} — {exc}"
            ) from exc

        if self._data.get("version") != STATE_VERSION:
            raise StateCorruptedError(
                f"State version mismatch: expected {STATE_VERSION}, "
                f"got {self._data.get('version')}"
            )

    def create(
        self,
        input_pdf: Path,
        total_pages: int,
        config_snapshot: dict,
    ) -> None:
        """
        Create a fresh state for a new job.

        Parameters
        ----------
        input_pdf : Path
            Path to the source PDF (used for hash validation).
        total_pages : int
            Number of pages in the PDF.
        config_snapshot : dict
            Key config values to detect prompt/model changes on resume.
        """
        now = datetime.now(timezone.utc).isoformat()
        self._data = {
            "version": STATE_VERSION,
            "input_pdf": str(input_pdf),
            "input_pdf_hash": self._hash_file(input_pdf),
            "total_pages": total_pages,
            "config_snapshot": config_snapshot,
            "stages": {
                "split": {"status": "pending", "completed_pages": []},
                "transcribe": {
                    "status": "pending",
                    "completed_pages": [],
                    "failed_pages": {},
                },
                "clean": {"status": "pending"},
            },
            "created_at": now,
            "updated_at": now,
        }
        self._flush()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def total_pages(self) -> int:
        return self._data.get("total_pages", 0)

    @property
    def input_pdf_hash(self) -> str:
        return self._data.get("input_pdf_hash", "")

    @property
    def config_snapshot(self) -> dict:
        return self._data.get("config_snapshot", {})

    def split_completed_pages(self) -> set[int]:
        """Pages whose images have been rendered."""
        return set(self._data["stages"]["split"].get("completed_pages", []))

    def is_split_complete(self) -> bool:
        return self._data["stages"]["split"]["status"] == "complete"

    def transcribe_completed_pages(self) -> set[int]:
        """Pages whose transcripts are saved."""
        return set(
            self._data["stages"]["transcribe"].get("completed_pages", [])
        )

    def is_transcribe_complete(self) -> bool:
        return (
            self._data["stages"]["transcribe"]["status"] == "complete"
            and len(self.transcribe_completed_pages()) >= self.total_pages
        )

    def is_clean_complete(self) -> bool:
        return (
            self._data["stages"]["clean"]["status"] == "complete"
            and self.is_transcribe_complete()
        )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def mark_page_split(self, page_num: int) -> None:
        """Record that a page image has been rendered."""
        pages = self._data["stages"]["split"]["completed_pages"]
        if page_num not in pages:
            pages.append(page_num)
        self._touch()

    def mark_split_complete(self) -> None:
        self._data["stages"]["split"]["status"] = "complete"
        self._flush()

    def mark_page_transcribed(self, page_num: int) -> None:
        """Record that a page transcript has been saved."""
        pages = self._data["stages"]["transcribe"]["completed_pages"]
        if page_num not in pages:
            pages.append(page_num)
        # Remove from failed if it was there
        failed = self._data["stages"]["transcribe"].get("failed_pages", {})
        failed.pop(str(page_num), None)
        self._flush()

    def mark_page_failed(self, page_num: int, error: str) -> None:
        """Record a transcription failure for a page."""
        failed = self._data["stages"]["transcribe"].setdefault(
            "failed_pages", {}
        )
        failed[str(page_num)] = {"error": error}
        self._flush()

    def mark_transcribe_complete(self) -> None:
        self._data["stages"]["transcribe"]["status"] = "complete"
        self._flush()

    def mark_transcribe_in_progress(self) -> None:
        self._data["stages"]["transcribe"]["status"] = "in_progress"
        self._flush()

    def mark_clean_complete(self) -> None:
        self._data["stages"]["clean"]["status"] = "complete"
        self._flush()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _touch(self) -> None:
        """Update the timestamp without flushing (batch operations)."""
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()

    def _flush(self) -> None:
        """Write the current state to disk atomically."""
        self._touch()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(self._path)

    def flush(self) -> None:
        """Public flush — used by the pipeline on Ctrl+C."""
        self._flush()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path: Path, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha.update(chunk)
        return f"sha256:{sha.hexdigest()}"

    @staticmethod
    def hash_file(path: Path) -> str:
        """Public API for hashing a file (used for validation)."""
        return JobState._hash_file(path)
