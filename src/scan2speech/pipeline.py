"""
Pipeline Orchestrator — coordinates the full split → transcribe → clean
lifecycle with state checkpointing and graceful error handling.

Folder layout:
  input/<pdf_stem>/               Page images (page_0001.png, ...)
  output/<pdf_stem>/              Per-page transcripts (page_0001.txt, ...)
  output/<pdf_stem>/<stem>_tts_ready.txt   Final stitched output
  output/<pdf_stem>/.scan2speech_state.json   Job state manifest
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from scan2speech import QuotaExhaustedError
from scan2speech.config import PipelineConfig
from scan2speech.pdf_splitter import PdfSplitter
from scan2speech.state import JobState
from scan2speech.text_cleaner import TextCleaner
from scan2speech.vision_client import VisionClient

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the end-to-end Scan2Speech processing pipeline.

    Lifecycle:
    1. Initialize/resume state
    2. Split PDF → page images into  input/<pdf_stem>/
    3. Transcribe each page image → per-page .txt into  output/<pdf_stem>/
    4. Clean & stitch all transcripts → final TTS-ready text
    5. Report completion

    Parameters
    ----------
    config : PipelineConfig
        Fully resolved pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._splitter = PdfSplitter(config)
        self._cleaner = TextCleaner(config)
        self._vision: VisionClient | None = None

        # Directory paths
        self._input_dir = config.input_dir       # input/<pdf_stem>/
        self._output_dir = config.output_dir     # output/<pdf_stem>/

        # State file lives inside the output folder
        self._state = JobState(config.output_dir)

    def run(self) -> Path:
        """
        Execute the full pipeline (or split-only if configured).

        Returns
        -------
        Path
            Path to the final output text file (or input images dir if split-only).

        Raises
        ------
        QuotaExhaustedError
            If API quota is exhausted (state is checkpointed first).
        SystemExit
            On unrecoverable errors.
        """
        config = self._config
        pdf_path = config.input_pdf

        logger.info("Starting Scan2Speech pipeline")
        logger.info("  Input PDF:   %s", pdf_path)
        logger.info("  Images dir:  %s", self._input_dir)
        logger.info("  Output dir:  %s", self._output_dir)
        if config.split_only:
            logger.info("  Mode:        Split-only (images extraction)")

        # ------ Phase 0: State setup ------
        self._init_state(pdf_path)

        try:
            # ------ Phase 1: Split PDF → images ------
            self._run_split(pdf_path)

            if config.split_only:
                logger.info(
                    "✓ Split complete! %d page images ready in: %s",
                    self._state.total_pages, self._input_dir,
                )
                return self._input_dir

            # ------ Phase 2: Transcribe images → text (page by page) ------
            self._run_transcribe()

            # ------ Phase 3: Clean & stitch → output ------
            self._run_clean()

        except QuotaExhaustedError:
            self._state.flush()
            logger.error(
                "API quota exhausted. Progress saved — re-run the same "
                "command to resume from where you left off."
            )
            sys.exit(2)

        except KeyboardInterrupt:
            self._state.flush()
            logger.info(
                "\nInterrupted. Progress saved — re-run to resume."
            )
            sys.exit(130)

        logger.info("✓ Done! Output written to: %s", config.resolved_output)
        return config.resolved_output

    # ------------------------------------------------------------------
    # Phase 0: State initialization
    # ------------------------------------------------------------------

    def _init_state(self, pdf_path: Path) -> None:
        """Load existing state or create a fresh one."""
        if self._config.force_restart:
            if self._input_dir.is_dir():
                shutil.rmtree(self._input_dir, ignore_errors=True)
                logger.info("--force-restart: cleared %s", self._input_dir)
            if self._output_dir.is_dir():
                shutil.rmtree(self._output_dir, ignore_errors=True)
                logger.info("--force-restart: cleared %s", self._output_dir)

        if self._state.exists():
            self._state.load()

            # Validate PDF hash
            current_hash = JobState.hash_file(pdf_path)
            if current_hash != self._state.input_pdf_hash:
                logger.error(
                    "Input PDF has changed since the last run. "
                    "Use --force-restart to start fresh."
                )
                sys.exit(1)

            # Validate config snapshot
            saved = self._state.config_snapshot
            if saved.get("model") != self._config.model:
                logger.warning(
                    "Model changed (%s → %s). Transcripts from the "
                    "previous model will be kept; re-run with "
                    "--force-restart to discard them.",
                    saved.get("model"), self._config.model,
                )

            logger.info("Resuming from saved state")
        else:
            total_pages = PdfSplitter.count_pages(pdf_path)
            self._state.create(
                input_pdf=pdf_path,
                total_pages=total_pages,
                config_snapshot={
                    "dpi_scale": self._config.dpi_scale,
                    "model": self._config.model,
                    "prompt_hash": self._config.prompt_hash,
                },
            )
            logger.info("Created fresh state (%d pages)", total_pages)

    # ------------------------------------------------------------------
    # Phase 1: Split PDF → page images
    # ------------------------------------------------------------------

    def _run_split(self, pdf_path: Path) -> None:
        """Render PDF pages to images inside input/<pdf_stem>/."""
        if self._state.is_split_complete():
            logger.info("Split stage: already complete — skipping")
            return

        skip = self._state.split_completed_pages()
        if skip:
            logger.info("Split stage: resuming (%d pages already done)", len(skip))

        results = self._splitter.split(
            pdf_path=pdf_path,
            output_dir=self._input_dir,
            skip_pages=skip,
        )

        # Update state for each rendered page
        for page_img in results:
            self._state.mark_page_split(page_img.page_num)

        self._state.mark_split_complete()
        logger.info(
            "Split stage: complete — %d page images in %s",
            self._state.total_pages, self._input_dir,
        )

    # ------------------------------------------------------------------
    # Phase 2: Transcribe (image by image)
    # ------------------------------------------------------------------

    def _run_transcribe(self) -> None:
        """
        Transcribe each page image via Gemini and save the cleaned
        per-page transcript into output/<pdf_stem>/page_NNNN.txt.
        """
        if self._state.is_transcribe_complete():
            logger.info("Transcribe stage: already complete — skipping")
            return

        done_pages = self._state.transcribe_completed_pages()
        total_pages = self._state.total_pages
        all_pages = set(range(1, total_pages + 1))
        pending = sorted(all_pages - done_pages)

        if not pending:
            self._state.mark_transcribe_complete()
            logger.info("Transcribe stage: all pages already transcribed")
            return

        logger.info(
            "Transcribe stage: %d/%d pages pending", len(pending), total_pages
        )
        self._state.mark_transcribe_in_progress()

        # Ensure output dir exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self._vision is None:
            self._vision = VisionClient(self._config)

        # Process page by page
        for page_num in pending:
            image_path = self._input_dir / f"page_{page_num:04d}.png"

            if not image_path.is_file():
                logger.error(
                    "Page image not found: %s — was the split stage skipped?",
                    image_path,
                )
                sys.exit(1)

            logger.info(
                "Transcribing page %d/%d (%s) …",
                page_num, total_pages, image_path.name,
            )

            try:
                raw_text = self._vision.transcribe_page(image_path)

                # Clean individual page text
                cleaned_text = self._cleaner.clean_page(raw_text)

                # Save per-page transcript
                transcript_path = self._output_dir / f"page_{page_num:04d}.txt"
                transcript_path.write_text(cleaned_text, encoding="utf-8")

                self._state.mark_page_transcribed(page_num)
                logger.info(
                    "  ✓ Page %d saved → %s (%d chars)",
                    page_num, transcript_path.name, len(cleaned_text),
                )

            except QuotaExhaustedError:
                # Let it bubble up to the pipeline-level handler
                raise

            except Exception as exc:
                self._state.mark_page_failed(page_num, str(exc))
                logger.warning(
                    "  ✗ Page %d failed: %s — skipping", page_num, exc
                )

        if len(self._state.transcribe_completed_pages()) >= total_pages:
            self._state.mark_transcribe_complete()
            logger.info("Transcribe stage: complete")
        else:
            remaining = total_pages - len(self._state.transcribe_completed_pages())
            logger.warning(
                "Transcribe stage: %d/%d pages remaining (some pages failed)",
                remaining, total_pages,
            )

    # ------------------------------------------------------------------
    # Phase 3: Clean & stitch
    # ------------------------------------------------------------------

    def _run_clean(self) -> None:
        """Stitch all per-page transcripts into the final output file."""
        if not self._state.is_transcribe_complete():
            logger.warning("Clean stage: skipping stitch until all pages are transcribed")
            return

        if self._state.is_clean_complete():
            logger.info("Clean stage: already complete — skipping")
            return

        total_pages = self._state.total_pages
        all_page_nums = list(range(1, total_pages + 1))

        cleaned = self._cleaner.stitch_from_files(
            transcript_dir=self._output_dir,
            page_nums=all_page_nums,
        )

        # Write final output
        output_path = self._config.resolved_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned, encoding="utf-8")

        self._state.mark_clean_complete()
        logger.info("Clean stage: complete (%d chars written)", len(cleaned))
