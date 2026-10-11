"""
Configuration loading, validation, and the central PipelineConfig dataclass.

Priority (lowest → highest):
  Hardcoded defaults → .env file → Environment variables → CLI arguments

Folder layout:
  <project_root>/input/<pdf_stem>/   — extracted page images
  <project_root>/output/<pdf_stem>/  — per-page transcripts + final stitched text
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from scan2speech import ConfigError

# ---------------------------------------------------------------------------
# Default transcription prompt
# ---------------------------------------------------------------------------

DEFAULT_TRANSCRIPTION_PROMPT = (
    "You are an expert document transcription assistant. Transcribe the text "
    "visible in this scanned page image accurately and completely.\n\n"
    "Rules:\n"
    "- Output ONLY the body text content. No page numbers, headers, footers, "
    "or footnotes.\n"
    "- Rejoin any hyphenated words that are split across lines.\n"
    "- Preserve paragraph structure with blank lines between paragraphs.\n"
    "- Do not add any commentary, markdown formatting, or annotations.\n"
    "- If the page is blank or contains only images/diagrams with no text, "
    "output exactly: [BLANK PAGE]\n"
    "- Transcribe in: {language}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Walk up from this file to find the project root (directory containing
    ``pyproject.toml``).  Falls back to cwd if not found.
    """
    current = Path(__file__).resolve().parent
    for ancestor in [current] + list(current.parents):
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    return Path.cwd()


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a single pipeline run."""

    # Required
    input_pdf: Path
    api_key: str

    # Provider ("gemini" or "deepseek")
    provider: str = "gemini"

    # Project root (auto-detected or overridden)
    project_root: Path = field(default_factory=_find_project_root)

    # PDF rendering
    dpi_scale: float = 2.0

    # API settings
    model: str = "gemini-3.8-flash"
    requests_per_minute: int = 10
    max_retries: int = 5
    retry_base_delay: float = 1.0

    # Transcription prompt
    prompt: str = DEFAULT_TRANSCRIPTION_PROMPT

    # Behavior
    split_only: bool = False
    force_restart: bool = False
    language: str = "en"
    verbose: bool = False

    # ----- derived / resolved paths (set in __post_init__) -----
    input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    resolved_output: Path = field(init=False)
    resolved_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        stem = self.input_pdf.stem

        # input/<pdf_stem>/  — page images live here
        object.__setattr__(
            self, "input_dir",
            self.project_root / "input" / stem,
        )

        # output/<pdf_stem>/ — transcripts + final text live here
        object.__setattr__(
            self, "output_dir",
            self.project_root / "output" / stem,
        )

        # Final stitched output file
        object.__setattr__(
            self, "resolved_output",
            self.project_root / "output" / stem / f"{stem}_tts_ready.txt",
        )

        # Resolve prompt with language placeholder
        object.__setattr__(
            self, "resolved_prompt",
            self.prompt.format(language=self.language),
        )

    @property
    def prompt_hash(self) -> str:
        """SHA-256 of the resolved prompt (for state validation)."""
        return hashlib.sha256(self.resolved_prompt.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Factory: build config from CLI namespace
# ---------------------------------------------------------------------------

def load_config(args) -> PipelineConfig:
    """
    Build a PipelineConfig by merging CLI args over env-var defaults.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    ConfigError
        If the API key is missing or the input file doesn't exist.
    """
    # Load .env (no-op if file absent)
    load_dotenv()

    input_pdf = Path(args.input_pdf).resolve()
    if not input_pdf.is_file():
        raise ConfigError(f"Input file not found: {input_pdf}")

    split_only = getattr(args, "split_only", False)
    provider_arg = getattr(args, "provider", None) or os.getenv("SCAN2SPEECH_PROVIDER")

    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip().strip("[]").strip("\"'")
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip().strip("[]").strip("\"'")

    if provider_arg:
        provider = provider_arg.lower()
        if provider not in ("gemini", "deepseek"):
            raise ConfigError(
                f"Unsupported provider '{provider}'. Supported: 'gemini', 'deepseek'."
            )
        api_key = deepseek_key if provider == "deepseek" else gemini_key
    elif deepseek_key:
        provider = "deepseek"
        api_key = deepseek_key
    elif gemini_key:
        provider = "gemini"
        api_key = gemini_key
    else:
        provider = "gemini"
        api_key = ""

    if not api_key and not split_only:
        raise ConfigError(
            "No API key configured. "
            "Set DEEPSEEK_API_KEY or GEMINI_API_KEY in your .env file or environment."
        )

    default_model = "deepseek-flash" if provider == "deepseek" else "gemini-3.8-flash"
    model = args.model or os.getenv("SCAN2SPEECH_MODEL") or default_model

    # Build config with env-var fallbacks where applicable
    return PipelineConfig(
        input_pdf=input_pdf,
        api_key=api_key,
        provider=provider,
        dpi_scale=args.dpi_scale or float(os.getenv("SCAN2SPEECH_DPI_SCALE", "2.0")),
        model=model,
        requests_per_minute=args.rpm or int(os.getenv("SCAN2SPEECH_RPM", "10")),
        max_retries=args.max_retries,
        prompt=args.prompt or DEFAULT_TRANSCRIPTION_PROMPT,
        split_only=split_only,
        force_restart=args.force_restart,
        language=args.language,
        verbose=args.verbose,
    )
