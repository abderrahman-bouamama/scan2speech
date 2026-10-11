"""Tests for the Pipeline orchestrator.

These tests use mocks to verify the orchestration logic without
requiring a real PDF or API access.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scan2speech.config import PipelineConfig


class TestPipelinePhaseSkipping:
    """Verify that the pipeline skips already-completed phases."""

    def test_skips_split_when_complete(self):
        """If split is already done in state, _run_split should be a no-op."""
        from scan2speech.pipeline import Pipeline

        config = MagicMock(spec=PipelineConfig)
        config.input_pdf = Path("/fake/input.pdf")
        config.resolved_output = Path("/fake/output.txt")
        config.resolved_work_dir = Path("/fake/work")
        config.keep_artifacts = False
        config.force_restart = False
        config.dpi_scale = 2.0
        config.model = "test"
        config.prompt_hash = "abc"

        with patch.object(Pipeline, "__init__", lambda self, cfg: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline._config = config

            # Mock state that says split is complete
            mock_state = MagicMock()
            mock_state.is_split_complete.return_value = True
            pipeline._state = mock_state
            pipeline._splitter = MagicMock()
            pipeline._pages_dir = Path("/fake/pages")

            pipeline._run_split(Path("/fake/input.pdf"))

            # Splitter should NOT have been called
            pipeline._splitter.split.assert_not_called()

    def test_split_only_stops_after_split(self):
        """When split_only is True, pipeline runs split but not transcribe or clean."""
        from scan2speech.pipeline import Pipeline

        config = MagicMock(spec=PipelineConfig)
        config.input_pdf = Path("/fake/input.pdf")
        config.input_dir = Path("/fake/input_dir")
        config.output_dir = Path("/fake/output_dir")
        config.split_only = True
        config.force_restart = False

        with patch.object(Pipeline, "__init__", lambda self, cfg: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline._config = config
            pipeline._input_dir = config.input_dir
            pipeline._output_dir = config.output_dir
            pipeline._init_state = MagicMock()
            pipeline._run_split = MagicMock()
            pipeline._run_transcribe = MagicMock()
            pipeline._run_clean = MagicMock()
            mock_state = MagicMock()
            mock_state.total_pages = 10
            pipeline._state = mock_state

            result = pipeline.run()

            pipeline._init_state.assert_called_once_with(config.input_pdf)
            pipeline._run_split.assert_called_once_with(config.input_pdf)
            pipeline._run_transcribe.assert_not_called()
            pipeline._run_clean.assert_not_called()
            assert result == config.input_dir
