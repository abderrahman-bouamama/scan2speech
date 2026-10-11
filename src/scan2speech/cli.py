"""
CLI handler — command-line entry point for Scan2Speech.

Usage:
  scan2speech book.pdf
  scan2speech book.pdf --dpi-scale 3.0 --language fr --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

from scan2speech import Scan2SpeechError, __version__
from scan2speech.config import load_config
from scan2speech.pipeline import Pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and parse the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="scan2speech",
        description=(
            "Convert scanned PDFs into clean, TTS-ready text using "
            "Gemini Flash multimodal vision.\n\n"
            "Page images are saved to  input/<pdf_name>/\n"
            "Transcripts are saved to  output/<pdf_name>/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_pdf",
        help="Path to the scanned PDF file",
    )
    parser.add_argument(
        "--dpi-scale",
        type=float,
        default=None,
        help="DPI scale multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "deepseek"],
        default=None,
        help="Vision AI provider (gemini or deepseek, default: auto-detect)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: gemini-3.8-flash or deepseek-flash)",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Max API requests per minute (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per page on API errors (default: 5)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom transcription prompt",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Target language code (default: en)",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only extract PDF pages as images (skips API transcription)",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore saved state and start fresh",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args(argv)


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the scan2speech CLI."""
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger("scan2speech")

    try:
        config = load_config(args)
        pipeline = Pipeline(config)
        pipeline.run()
    except Scan2SpeechError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
