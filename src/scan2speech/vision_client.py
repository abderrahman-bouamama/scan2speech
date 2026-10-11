"""
Vision Client — wraps the Google GenAI SDK to transcribe page images via
Gemini Flash with retry logic and rate limiting.
"""

from __future__ import annotations

import base64
import logging
import random
import time
from pathlib import Path
from typing import Callable, NamedTuple

import httpx
from google import genai
from google.genai import types as genai_types
from PIL import Image

from scan2speech import ConfigError, QuotaExhaustedError, TranscriptionError
from scan2speech.config import PipelineConfig

logger = logging.getLogger(__name__)


class PageTranscript(NamedTuple):
    """Holds the transcript result for a single page."""
    page_num: int
    text: str


class VisionClient:
    """
    Sends page images to the Gemini multimodal vision model for transcription.

    Implements:
    - Self-imposed rate limiting (requests per minute).
    - Exponential backoff with jitter on 429 / 5xx errors.
    - Graceful quota-exhaustion detection.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration (uses ``api_key``, ``model``,
        ``requests_per_minute``, ``max_retries``, ``retry_base_delay``,
        ``resolved_prompt``).
    """

    # HTTP status codes that trigger a retry
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

    def __init__(self, config: PipelineConfig) -> None:
        self._provider = getattr(config, "provider", "gemini")
        self._api_key = config.api_key
        self._model = config.model
        self._prompt = config.resolved_prompt
        self._max_retries = config.max_retries
        self._base_delay = config.retry_base_delay
        self._min_interval = 60.0 / config.requests_per_minute
        self._last_request_time: float = 0.0

        if self._provider == "gemini":
            self._client = genai.Client(api_key=config.api_key)
        elif self._provider == "deepseek":
            self._http_client = httpx.Client(timeout=120.0)
        else:
            raise ConfigError(f"Unsupported provider: {self._provider}")

    def __del__(self) -> None:
        if hasattr(self, "_http_client"):
            try:
                self._http_client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_page(
        self,
        image_path: Path,
        prompt: str | None = None,
    ) -> str:
        """
        Transcribe a single page image via Gemini or DeepSeek vision model.

        Parameters
        ----------
        image_path : Path
            Path to the PNG page image.
        prompt : str | None
            Override the default transcription prompt for this page.

        Returns
        -------
        str
            The raw transcript text returned by the model.

        Raises
        ------
        QuotaExhaustedError
            If all retries are exhausted due to rate-limiting.
        TranscriptionError
            If the API returns an unexpected, non-retryable error.
        """
        effective_prompt = prompt or self._prompt

        for attempt in range(1, self._max_retries + 1):
            self._rate_limit()
            try:
                if self._provider == "gemini":
                    pil_image = Image.open(image_path)
                    response = self._client.models.generate_content(
                        model=self._model,
                        contents=[
                            effective_prompt,
                            pil_image,
                        ],
                        config=genai_types.GenerateContentConfig(
                            temperature=0.0,
                        ),
                    )
                    text = response.text or ""
                    return text.strip()
                elif self._provider == "deepseek":
                    text = self._transcribe_deepseek(image_path, effective_prompt)
                    return text.strip()
                else:
                    raise TranscriptionError(f"Unknown provider: {self._provider}")

            except Exception as exc:
                status = self._extract_status(exc)

                if status in self._RETRYABLE_STATUS_CODES:
                    delay = self._backoff_delay(attempt)
                    if status == 429:
                        delay = max(delay, 15.0 * attempt)
                    elif status == 503:
                        delay = max(delay, 5.0 * attempt)
                    logger.warning(
                        "API %d on attempt %d/%d — retrying in %.1fs",
                        status, attempt, self._max_retries, delay,
                    )
                    time.sleep(delay)
                    continue

                # Non-retryable error
                raise TranscriptionError(
                    f"{self._provider.capitalize()} API error for {image_path.name}: {exc}"
                ) from exc

        raise QuotaExhaustedError(
            f"All {self._max_retries} retries exhausted for {image_path.name}"
        )

    def _transcribe_deepseek(self, image_path: Path, prompt: str) -> str:
        """Send page image to DeepSeek vision API."""
        b64_img = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_img}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
        }
        resp = self._http_client.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
        )
        if resp.status_code != 200:
            resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def transcribe_pages(
        self,
        pages: list[tuple[int, Path]],
        on_progress: Callable[[int], None] | None = None,
    ) -> list[PageTranscript]:
        """
        Batch-transcribe multiple page images sequentially.

        Parameters
        ----------
        pages : list[tuple[int, Path]]
            List of (page_number, image_path) pairs to transcribe.
        on_progress : Callable[[int], None] | None
            Called with the page number after each successful transcription
            (used by the pipeline to update state).

        Returns
        -------
        list[PageTranscript]
            Successfully transcribed pages.

        Raises
        ------
        QuotaExhaustedError
            Propagated from ``transcribe_page`` if quota is exhausted.
        """
        results: list[PageTranscript] = []

        for page_num, image_path in pages:
            logger.info(
                "Transcribing page %d (%s) …", page_num, image_path.name
            )
            text = self.transcribe_page(image_path)
            results.append(PageTranscript(page_num=page_num, text=text))

            if on_progress:
                on_progress(page_num)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce the self-imposed request-per-minute ceiling."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            sleep_for = self._min_interval - elapsed
            logger.debug("Rate-limit: sleeping %.2fs", sleep_for)
            time.sleep(sleep_for)
        self._last_request_time = time.monotonic()

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with ±25 % jitter."""
        base = self._base_delay * (2 ** (attempt - 1))
        jitter = base * 0.25 * (2 * random.random() - 1)
        return base + jitter

    @staticmethod
    def _extract_status(exc: Exception) -> int:
        """Best-effort extraction of an HTTP status code from an SDK exception."""
        # httpx HTTPStatusError
        if hasattr(exc, "response") and getattr(exc.response, "status_code", None):
            return exc.response.status_code
        # google-genai wraps HTTP errors with a .code or .status attribute
        for attr in ("code", "status", "status_code"):
            val = getattr(exc, attr, None)
            if isinstance(val, int):
                return val
        # Fall back to parsing the string repr
        exc_str = str(exc)
        for code in (429, 500, 502, 503):
            if str(code) in exc_str:
                return code
        return 0
