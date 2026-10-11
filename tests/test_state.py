"""Tests for the JobState module."""

import json
from pathlib import Path

import pytest

from scan2speech.state import JobState, STATE_FILENAME


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Provide a temporary work directory."""
    d = tmp_path / "test_work"
    d.mkdir()
    return d


@pytest.fixture
def dummy_pdf(tmp_path: Path) -> Path:
    """Create a tiny dummy file to use as a fake PDF for hashing."""
    f = tmp_path / "dummy.pdf"
    f.write_bytes(b"%PDF-1.4 fake content for hashing")
    return f


class TestJobStateCreation:
    """Verify state creation and basic queries."""

    def test_create_and_exists(self, work_dir: Path, dummy_pdf: Path):
        state = JobState(work_dir)
        assert not state.exists()

        state.create(input_pdf=dummy_pdf, total_pages=10, config_snapshot={"model": "test"})
        assert state.exists()

    def test_total_pages(self, work_dir: Path, dummy_pdf: Path):
        state = JobState(work_dir)
        state.create(input_pdf=dummy_pdf, total_pages=42, config_snapshot={})
        assert state.total_pages == 42

    def test_initial_stages_pending(self, work_dir: Path, dummy_pdf: Path):
        state = JobState(work_dir)
        state.create(input_pdf=dummy_pdf, total_pages=5, config_snapshot={})
        assert not state.is_split_complete()
        assert not state.is_transcribe_complete()
        assert not state.is_clean_complete()


class TestJobStateResume:
    """Verify state can be saved and reloaded."""

    def test_save_and_reload(self, work_dir: Path, dummy_pdf: Path):
        state1 = JobState(work_dir)
        state1.create(input_pdf=dummy_pdf, total_pages=5, config_snapshot={"model": "test"})
        state1.mark_page_split(1)
        state1.mark_page_split(2)
        state1.mark_split_complete()

        state2 = JobState(work_dir)
        state2.load()
        assert state2.is_split_complete()
        assert state2.split_completed_pages() == {1, 2}

    def test_pdf_hash_persists(self, work_dir: Path, dummy_pdf: Path):
        state = JobState(work_dir)
        state.create(input_pdf=dummy_pdf, total_pages=1, config_snapshot={})

        expected_hash = JobState.hash_file(dummy_pdf)

        state2 = JobState(work_dir)
        state2.load()
        assert state2.input_pdf_hash == expected_hash


class TestJobStateTranscription:
    """Verify transcription tracking."""

    def test_mark_transcribed(self, work_dir: Path, dummy_pdf: Path):
        state = JobState(work_dir)
        state.create(input_pdf=dummy_pdf, total_pages=5, config_snapshot={})

        state.mark_page_transcribed(1)
        state.mark_page_transcribed(3)
        assert state.transcribe_completed_pages() == {1, 3}

    def test_mark_failed_then_succeed(self, work_dir: Path, dummy_pdf: Path):
        state = JobState(work_dir)
        state.create(input_pdf=dummy_pdf, total_pages=5, config_snapshot={})

        state.mark_page_failed(2, "429 Too Many Requests")
        state.mark_page_transcribed(2)

        # Page 2 should now be in completed, not failed
        assert 2 in state.transcribe_completed_pages()

        # Reload and verify the failed entry was removed
        state2 = JobState(work_dir)
        state2.load()
        raw = json.loads((work_dir / STATE_FILENAME).read_text())
        assert "2" not in raw["stages"]["transcribe"].get("failed_pages", {})


class TestJobStateCompletion:
    """Verify stage completion markers."""

    def test_full_lifecycle(self, work_dir: Path, dummy_pdf: Path):
        state = JobState(work_dir)
        state.create(input_pdf=dummy_pdf, total_pages=2, config_snapshot={})

        state.mark_page_split(1)
        state.mark_page_split(2)
        state.mark_split_complete()
        assert state.is_split_complete()

        state.mark_page_transcribed(1)
        state.mark_page_transcribed(2)
        state.mark_transcribe_complete()
        assert state.is_transcribe_complete()

        state.mark_clean_complete()
        assert state.is_clean_complete()
