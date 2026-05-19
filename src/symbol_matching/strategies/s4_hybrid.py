"""S4 — Hybrid: classical template proposals + DINOv3 rerank + dense fallback.

Stage A (classical, fast, high-precision)
  Multi-rotation NCC template matching at a *loose* threshold to flood
  the page with candidates. Loose threshold trades precision for recall —
  by design.

Stage B (deep, semantic)
  DINOv3 cosine similarity between the reference embedding and each
  proposal crop. Catches stylistic variation that pixel correlation
  misses (different stroke weights, different inner labels).
  Final score = weighted sum of stage-A NCC and stage-B cosine.

Stage C (fallback, recall safety net)
  On pages where stage A returned almost nothing, fire S3's dense-grid
  DINOv3 scan. Catches symbols that template matching missed entirely.
  Conditional, not run on every page — keeps wall-clock tractable.

"""

from __future__ import annotations

import cv2
import numpy as np

from .. import embed
from ..records import MatchResult
from . import s1_template, s3_dinov3


def find_matches_on_page(
    page: np.ndarray,
    reference: np.ndarray,
    *,
    # Stage A — loose threshold favors recall.
    stage_a_threshold: float = 0.50,
    stage_a_rotations: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315),
    stage_a_scales: tuple[float, ...] = (0.9, 1.0, 1.1),
    # Stage B — final accept threshold on the combined score.
    stage_b_weight_ncc: float = 0.5,
    stage_b_weight_cosine: float = 0.5,
    stage_b_accept_threshold: float = 0.55,
    # Stage C — when to fire the dense-grid fallback.
    stage_c_min_proposals: int = 5,
    stage_c_threshold: float = 0.55,
    stage_c_stride_frac: float = 1.0,  # coarse — Stage C is a safety net, not a primary scan
    page_idx: int = 0,
    page_meta: dict | None = None,
    reference_id: str = "ref",
) -> list[MatchResult]:
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if page.ndim == 3:
        page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # ---- Stage A: template proposals at loose threshold.
    proposals = s1_template.find_matches_on_page(
        page,
        reference,
        threshold=stage_a_threshold,
        rotations=stage_a_rotations,
        scales=stage_a_scales,
        page_idx=page_idx,
        page_meta=page_meta,
        reference_id=reference_id,
    )

    # ---- Stage B: DINOv3 rerank of each proposal (batched).
    ref_emb = embed.encode_crop(reference)
    crops = []
    for m in proposals:
        b = m.capture.bbox
        pad = max(4, min(b.w, b.h) // 8)
        y0 = max(0, b.y - pad)
        x0 = max(0, b.x - pad)
        y1 = min(page.shape[0], b.y + b.h + pad)
        x1 = min(page.shape[1], b.x + b.w + pad)
        crops.append(page[y0:y1, x0:x1])
    crop_embs = embed.encode_crops(crops) if crops else np.zeros((0, ref_emb.shape[0]))
    accepted: list[MatchResult] = []
    for m, ce in zip(proposals, crop_embs):
        cos = float(np.dot(ref_emb, ce))
        ncc = m.item.score_breakdown.get("ncc", m.item.score)
        combined = stage_b_weight_ncc * ncc + stage_b_weight_cosine * cos
        if combined >= stage_b_accept_threshold:
            m.item.score = combined
            m.item.score_breakdown = {
                "ncc": float(ncc),
                "dinov3_cosine": cos,
                "combined": combined,
            }
            m.item.stage = "rerank"
            accepted.append(m)

    # ---- Stage C: dense fallback on under-served pages.
    if len(accepted) < stage_c_min_proposals:
        fallback = s3_dinov3.find_matches_on_page(
            page,
            reference,
            threshold=stage_c_threshold,
            stride_frac=stage_c_stride_frac,
            page_idx=page_idx,
            page_meta=page_meta,
            reference_id=reference_id,
        )

        for m in fallback:
            m.item.stage = "fallback"
        # Merge, prefer existing rerank matches by NMS-ish: drop fallback hits
        # that overlap an accepted match.
        accepted = _merge(accepted, fallback)

    return accepted


def _merge(
    primary: list[MatchResult], fallback: list[MatchResult], iou_thresh: float = 0.3
) -> list[MatchResult]:
    """Add fallback matches that don't overlap any primary match."""
    if not primary:
        return fallback
    p_boxes = [
        (
            m.capture.bbox.x,
            m.capture.bbox.y,
            m.capture.bbox.x + m.capture.bbox.w,
            m.capture.bbox.y + m.capture.bbox.h,
        )
        for m in primary
    ]
    out = list(primary)
    for f in fallback:
        b = f.capture.bbox
        fx1, fy1, fx2, fy2 = b.x, b.y, b.x + b.w, b.y + b.h
        keep = True
        for px1, py1, px2, py2 in p_boxes:
            ix1 = max(fx1, px1)
            iy1 = max(fy1, py1)
            ix2 = min(fx2, px2)
            iy2 = min(fy2, py2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                continue
            union = (fx2 - fx1) * (fy2 - fy1) + (px2 - px1) * (py2 - py1) - inter
            if inter / union > iou_thresh:
                keep = False
                break
        if keep:
            out.append(f)
    return out
