"""
Scan2Speech — Lightweight scanned PDF to TTS-ready text pipeline.
"""

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------

class Scan2SpeechError(Exception):
    """Base exception for the scan2speech package."""


class ConfigError(Scan2SpeechError):
    """Invalid or missing configuration."""


class SplitError(Scan2SpeechError):
    """PDF rendering failure."""


class TranscriptionError(Scan2SpeechError):
    """Non-retryable API error for a single page."""


class QuotaExhaustedError(Scan2SpeechError):
    """All retries exhausted due to rate limiting / quota."""


class StateCorruptedError(Scan2SpeechError):
    """State file is unreadable or inconsistent."""
