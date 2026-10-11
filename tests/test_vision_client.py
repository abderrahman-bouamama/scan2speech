"""Tests for the VisionClient module.

These tests mock the google-genai SDK to avoid real API calls.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from scan2speech import QuotaExhaustedError, TranscriptionError
from scan2speech.config import PipelineConfig
from scan2speech.vision_client import VisionClient, PageTranscript


def _make_config(**overrides) -> MagicMock:
    """Create a mock PipelineConfig."""
    config = MagicMock(spec=PipelineConfig)
    config.api_key = "fake-api-key"
    config.provider = "gemini"
    config.model = "gemini-3.8-flash"
    config.resolved_prompt = "Transcribe this page."
    config.requests_per_minute = 6000  # Effectively no rate limiting in tests
    config.max_retries = 3
    config.retry_base_delay = 0.01  # Very short delays for tests
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


class TestRateLimit:
    """Verify the self-imposed rate limiter."""

    def test_rate_limit_enforces_delay(self):
        config = _make_config(requests_per_minute=120)  # 0.5s interval
        client = VisionClient(config)

        # First call should not block
        start = time.monotonic()
        client._rate_limit()
        elapsed_1 = time.monotonic() - start
        assert elapsed_1 < 0.1

        # Second call immediately should block ~0.5s
        start = time.monotonic()
        client._rate_limit()
        elapsed_2 = time.monotonic() - start
        assert elapsed_2 >= 0.3  # Allow some tolerance


class TestBackoff:
    """Verify exponential backoff calculation."""

    def test_backoff_increases(self):
        config = _make_config()
        client = VisionClient(config)

        delays = [client._backoff_delay(i) for i in range(1, 5)]
        # Base delay doubles each attempt (before jitter)
        # So the mean of attempt N should be about 2x the mean of N-1
        # Just check that the sequence is generally increasing
        assert delays[2] > delays[0]


class TestStatusExtraction:
    """Verify HTTP status code extraction from exceptions."""

    def test_extract_from_code_attr(self):
        exc = Exception("API error")
        exc.code = 429
        assert VisionClient._extract_status(exc) == 429

    def test_extract_from_string(self):
        exc = Exception("Error: 503 Service Unavailable")
        assert VisionClient._extract_status(exc) == 503

    def test_extract_from_httpx_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        exc = Exception("Rate limit")
        exc.response = mock_resp
        assert VisionClient._extract_status(exc) == 429

    def test_unknown_returns_zero(self):
        exc = Exception("Something went wrong")
        assert VisionClient._extract_status(exc) == 0


class TestDeepSeekProvider:
    """Verify DeepSeek provider initialization and transcription."""

    def test_deepseek_init(self):
        config = _make_config(provider="deepseek", model="deepseek-flash")
        client = VisionClient(config)
        assert client._provider == "deepseek"
        assert client._model == "deepseek-flash"
        assert hasattr(client, "_http_client")

    def test_deepseek_transcribe_mock(self, tmp_path):
        dummy_img = tmp_path / "page_0001.png"
        dummy_img.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")

        config = _make_config(provider="deepseek", model="deepseek-flash")
        client = VisionClient(config)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Transcribed Arabic text"}}]
        }
        client._http_client.post = MagicMock(return_value=mock_resp)

        text = client.transcribe_page(dummy_img)
        assert text == "Transcribed Arabic text"
        client._http_client.post.assert_called_once()
