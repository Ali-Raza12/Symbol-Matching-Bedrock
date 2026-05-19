"""Shared benchmark config: reference symbols + approximate ground truth.

Used by every notebook so S1..S4 are compared on identical inputs. The GT
here is *approximate* — it's the set of hexagonal contours found at the
expected size on the rendered pages.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import pdf_io
from .records import BBox

CD_SET_PDF = str(Path(__file__).resolve().parents[2] / "data" / "cd_set.pdf")
DPI = 300

REFERENCE = {
    "page_idx": 0,
    "bbox": BBox(x=7528, y=5424, w=74, h=66),
    "name": "hex_callout_P11",
}


def load_reference_crop() -> np.ndarray:
    """Return the cropped reference symbol (grayscale uint8)."""
    page = pdf_io.render_page(CD_SET_PDF, REFERENCE["page_idx"], dpi=DPI)
    b = REFERENCE["bbox"]
    return page[b.y : b.y + b.h, b.x : b.x + b.w].copy()


@lru_cache(maxsize=None)
def detect_hexagons(page_idx: int, dpi: int = DPI) -> list[BBox]:
    """Find hexagonal-callout-like contours on a rendered page.

    Used as approximate ground truth across all strategy comparisons. The
    size filter matches the dominant cluster on the test set (~60x50 px at
    300 dpi) with generous slack.
    """
    img = pdf_io.render_page(CD_SET_PDF, page_idx, dpi=dpi)
    _, bin_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (1500 < area < 5000):
            continue
        peri = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, closed=True)
        if len(approx) != 6:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.8 < w / h < 1.4):
            continue
        # Dominant size band on the test set.
        if not (50 <= w <= 80 and 45 <= h <= 70):
            continue
        boxes.append(BBox(x=int(x), y=int(y), w=int(w), h=int(h)))
    return boxes


def ground_truth_per_page() -> dict[int, list[BBox]]:
    """Return {page_idx: [bboxes]} for all pages in the test PDF."""
    return {i: detect_hexagons(i) for i in range(pdf_io.page_count(CD_SET_PDF))}
