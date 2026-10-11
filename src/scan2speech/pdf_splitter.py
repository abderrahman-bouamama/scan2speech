"""
PDF Splitter — renders each page of a scanned PDF to a high-res PNG image.

Uses pypdfium2 for zero-heavy-dependency PDF rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import pypdfium2 as pdfium
from PIL import Image

from scan2speech import SplitError
from scan2speech.config import PipelineConfig

logger = logging.getLogger(__name__)


class PageImage(NamedTuple):
    """Represents a rendered page image on disk."""
    page_num: int
    image_path: Path


class PdfSplitter:
    """
    Renders PDF pages to PNG images at a configurable DPI scale.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration (uses ``dpi_scale``).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._scale = config.dpi_scale

    def split(
        self,
        pdf_path: Path,
        output_dir: Path,
        skip_pages: set[int] | None = None,
    ) -> list[PageImage]:
        """
        Render each page of *pdf_path* to a PNG inside *output_dir*.

        Parameters
        ----------
        pdf_path : Path
            Source PDF file.
        output_dir : Path
            Directory to write page images into.
        skip_pages : set[int] | None
            Page numbers (1-indexed) to skip (already rendered).

        Returns
        -------
        list[PageImage]
            List of (page_num, image_path) for every newly rendered page.

        Raises
        ------
        SplitError
            If the PDF cannot be opened or a page fails to render.
        """
        skip_pages = skip_pages or set()
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:
            raise SplitError(f"Cannot open PDF: {pdf_path} — {exc}") from exc

        total_pages = len(pdf)
        logger.info("PDF has %d pages (scale=%.1fx)", total_pages, self._scale)

        results: list[PageImage] = []

        for page_idx in range(total_pages):
            page_num = page_idx + 1

            if page_num in skip_pages:
                logger.debug("Skipping page %d (already rendered)", page_num)
                continue

            image_path = output_dir / f"page_{page_num:04d}.png"

            try:
                page = pdf[page_idx]
                bitmap = page.render(scale=self._scale)
                pil_image: Image.Image = bitmap.to_pil()
                pil_image.save(str(image_path), format="PNG")
                logger.debug("Rendered page %d → %s", page_num, image_path.name)
            except Exception as exc:
                raise SplitError(
                    f"Failed to render page {page_num}: {exc}"
                ) from exc

            results.append(PageImage(page_num=page_num, image_path=image_path))

        pdf.close()
        logger.info("Rendered %d new page images", len(results))
        return results

    @staticmethod
    def count_pages(pdf_path: Path) -> int:
        """Return the total number of pages in a PDF without rendering."""
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
            count = len(pdf)
            pdf.close()
            return count
        except Exception as exc:
            raise SplitError(f"Cannot open PDF: {pdf_path} — {exc}") from exc
