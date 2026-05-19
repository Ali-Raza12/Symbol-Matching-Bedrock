"""S1 — Multi-rotation NCC template matching.

Sweep the reference template across each rotation (and optional scale ladder),
run normalized cross-correlation, take peaks
above a threshold, dedupe with NMS across all rotations/scales.

Strengths: zero training, fast on CPU, interpretable score per match,
high precision when the symbol is distinctive linework.
Weaknesses: discrete rotation sweep is lossy; loses recall on stylistic
variation (different stroke weights, different inner labels).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..records import BBox, Capture, DrawingItem, MatchResult, new_id
from ..reference import DEFAULT_ROTATIONS, DEFAULT_SCALES, Template, build_template_bank


@dataclass
class S1Hit:
    bbox: BBox
    score: float
    rotation_deg: int
    scale: float


def _ncc_hits(
    page: np.ndarray,
    template: Template,
    threshold: float,
) -> list[S1Hit]:
    """Run cv2.matchTemplate with TM_CCOEFF_NORMED and return hits above threshold.

    No NMS at this stage — that happens after merging across rotations.
    Score is the raw correlation coefficient in [-1, 1].
    """
    if (
        template.image.shape[0] >= page.shape[0]
        or template.image.shape[1] >= page.shape[1]
    ):
        return []
    result = cv2.matchTemplate(page, template.image, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= threshold)
    th, tw = template.image.shape
    return [
        S1Hit(
            bbox=BBox(x=int(x), y=int(y), w=int(tw), h=int(th)),
            score=float(result[y, x]),
            rotation_deg=template.rotation_deg,
            scale=template.scale,
        )
        for x, y in zip(xs, ys)
    ]


def _nms(hits: list[S1Hit], iou_thresh: float = 0.3) -> list[S1Hit]:
    """Non-max suppression across all hits regardless of rotation/scale."""
    if not hits:
        return []
    hits = sorted(hits, key=lambda h: h.score, reverse=True)
    boxes = np.array(
        [[h.bbox.x, h.bbox.y, h.bbox.x + h.bbox.w, h.bbox.y + h.bbox.h] for h in hits]
    )
    kept_idx: list[int] = []
    suppressed = np.zeros(len(hits), dtype=bool)
    for i in range(len(hits)):
        if suppressed[i]:
            continue
        kept_idx.append(i)
        if i + 1 == len(hits):
            break
        # IoU between hits[i] and all remaining
        bi = boxes[i]
        rest = boxes[i + 1 :]
        ix1 = np.maximum(bi[0], rest[:, 0])
        iy1 = np.maximum(bi[1], rest[:, 1])
        ix2 = np.minimum(bi[2], rest[:, 2])
        iy2 = np.minimum(bi[3], rest[:, 3])
        iw = np.clip(ix2 - ix1, 0, None)
        ih = np.clip(iy2 - iy1, 0, None)
        inter = iw * ih
        a_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
        a_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / (a_i + a_r - inter + 1e-9)
        suppressed[i + 1 :] |= iou > iou_thresh
    return [hits[i] for i in kept_idx]


def find_matches_on_page(
    page: np.ndarray,
    reference: np.ndarray,
    *,
    threshold: float = 0.65,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    nms_iou: float = 0.3,
    page_idx: int = 0,
    page_meta: dict | None = None,
    reference_id: str = "ref",
) -> list[MatchResult]:
    """Find all matches of `reference` on `page`. Returns MatchResult records."""
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if page.ndim == 3:
        page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    bank = build_template_bank(reference, rotations=rotations, scales=scales)
    all_hits: list[S1Hit] = []
    for t in bank:
        all_hits.extend(_ncc_hits(page, t, threshold=threshold))
    kept = _nms(all_hits, iou_thresh=nms_iou)

    meta = page_meta or {}
    results: list[MatchResult] = []
    for h in kept:
        cap = Capture(
            id=new_id("cap"),
            source_page_id=meta.get("sheet_ref") or f"page-{page_idx}",
            page_idx=page_idx,
            page_name=meta.get("page_name"),
            sheet_ref=meta.get("sheet_ref"),
            page_type=meta.get("page_type"),
            bbox=h.bbox,
            rotation_deg=h.rotation_deg,
            scale=h.scale,
        )
        item = DrawingItem(
            id=new_id("item"),
            capture_id=cap.id,
            reference_id=reference_id,
            score=h.score,
            score_breakdown={"ncc": h.score},
            stage="template",
        )
        results.append(MatchResult(capture=cap, item=item))
    return results
