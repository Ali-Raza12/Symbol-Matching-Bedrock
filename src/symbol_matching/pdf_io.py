"""PDF rendering and page-metadata extraction.

We render at 300 DPI grayscale (standard for AEC pipelines) and pull
page-level metadata from the title-block text. CAD-produced PDFs almost
always embed real text in the title block — we just regex for the sheet
reference and page name and infer `pageType` from the sheet-letter prefix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

# Sheet-letter prefix → broad pageType. AEC convention; covers the CD set.
PAGE_TYPE_BY_PREFIX = {
    "A": "Architectural",
    "S": "Structural",
    "M": "Mechanical",
    "E": "Electrical",
    "P": "Plumbing",
    "C": "Civil",
    "L": "Landscape",
    "FP": "Fire Protection",
    "T": "Telecom",
}

# Matches things like "E201", "E-201", "P-101", "A-1.1", "FP201" in title blocks.
_SHEET_RE = re.compile(r"\b([A-Z]{1,2})[-\s]?(\d{1,3}(?:\.\d+)?[A-Z]?)\b")


@dataclass
class PageMeta:
    page_idx: int          # 0-based
    sheet_ref: str | None  # e.g. "E201"
    page_name: str | None  # e.g. "First Floor Power Plan"
    page_type: str | None  # e.g. "Electrical"
    width_px: int
    height_px: int
    dpi: int


def render_page(pdf_path: str | Path, page_idx: int, dpi: int = 300, grayscale: bool = True) -> np.ndarray:
    """Render a PDF page to a numpy array.

    Returns HxW (grayscale) or HxWx3 (BGR) uint8 array.
    """
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_idx]
        zoom = dpi / 72.0  # PDF user-space is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY if grayscale else fitz.csRGB, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if grayscale:
            return arr[:, :, 0]
        # PyMuPDF returns RGB; convert to BGR for cv2 interop.
        return arr[:, :, ::-1].copy()


def extract_metadata(pdf_path: str | Path, page_idx: int, dpi: int = 300) -> PageMeta:
    """Pull sheet ref, page name, and pageType from a page's text layer.

    Strategy:
      - Sheet ref = first letter-prefix + number match found in the bottom-right
        third of the page (where title blocks live in AEC drawings).
      - Page name = the longest line of text in that same region that isn't the
        sheet ref or a date.
      - pageType = lookup on the sheet-ref prefix.

    Any field that can't be confidently extracted is left None — callers should
    treat None as "unknown" and degrade gracefully.
    """
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_idx]
        zoom = dpi / 72.0
        w_pdf, h_pdf = page.rect.width, page.rect.height
        # Title-block region: bottom-right ~third of the page (for sheet ref).
        clip = fitz.Rect(w_pdf * 2 / 3, h_pdf * 2 / 3, w_pdf, h_pdf)
        title_text = page.get_text("text", clip=clip)
        full_text = page.get_text("text")

        # Sheet ref: prefer the title-block region.
        sheet_ref = _find_sheet_ref(title_text) or _find_sheet_ref(full_text)

        # Page name: scan the whole page — in drawings the plan title is often
        # printed below the drawing body, not inside the title block proper.
        page_name = _find_page_name(full_text, sheet_ref)

        page_type = None
        if sheet_ref:
            prefix = re.match(r"^([A-Z]{1,2})", sheet_ref)
            if prefix:
                page_type = PAGE_TYPE_BY_PREFIX.get(prefix.group(1))

        return PageMeta(
            page_idx=page_idx,
            sheet_ref=sheet_ref,
            page_name=page_name,
            page_type=page_type,
            width_px=int(round(w_pdf * zoom)),
            height_px=int(round(h_pdf * zoom)),
            dpi=dpi,
        )


def _find_sheet_ref(text: str) -> str | None:
    candidates = []
    for prefix, num in _SHEET_RE.findall(text):
        if prefix in PAGE_TYPE_BY_PREFIX:
            candidates.append(f"{prefix}{num}")
    if not candidates:
        return None
    # Prefer shorter prefixes (E201 over FP201) only when they appear more often
    # — in real drawings the sheet ref is repeated several times.
    from collections import Counter
    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def _find_page_name(text: str, sheet_ref: str | None) -> str | None:
    skip_words = {"date", "drawn", "scale", "project", "sheet", "no", "drawing"}
    best = None
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 8 or len(line) > 80:
            continue
        if sheet_ref and sheet_ref in line:
            continue
        low = line.lower()
        if any(w in low for w in skip_words):
            continue
        # Looks like a plan name: mostly letters, has "PLAN" or "ELEVATION" or similar.
        if re.search(r"\b(PLAN|ELEVATION|DETAIL|SECTION|SCHEDULE|LEGEND|NOTES|DIAGRAM)\b", line, re.I):
            if best is None or len(line) > len(best):
                best = line
    return best


def page_count(pdf_path: str | Path) -> int:
    with fitz.open(str(pdf_path)) as doc:
        return doc.page_count
