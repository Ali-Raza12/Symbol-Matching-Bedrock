"""S3 — Pure DINOv3 sliding-window matcher.

Slide a window of the reference symbol's pixel size across the page (at a
modest stride), upsample each window to the encoder's input size, batch-
encode, cosine vs the reference embedding, threshold, NMS.

"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import embed
from ..records import BBox, Capture, DrawingItem, MatchResult, new_id
from .s1_template import _nms, S1Hit


@dataclass
class S3Hit:
    bbox: BBox
    score: float
    rotation_deg: int


def _windows(page_shape: tuple[int, int], win_h: int, win_w: int, stride: int):
    H, W = page_shape
    ys = list(range(0, max(1, H - win_h + 1), stride))
    xs = list(range(0, max(1, W - win_w + 1), stride))
    if ys[-1] != H - win_h:
        ys.append(max(0, H - win_h))
    if xs[-1] != W - win_w:
        xs.append(max(0, W - win_w))
    return ys, xs


def find_matches_on_page(
    page: np.ndarray,
    reference: np.ndarray,
    *,
    stride_frac: float = 0.5,
    threshold: float = 0.80,
    rotations: tuple[int, ...] = (0, 90, 180, 270),
    nms_iou: float = 0.3,
    batch_size: int = 128,
    page_idx: int = 0,
    page_meta: dict | None = None,
    reference_id: str = "ref",
) -> list[MatchResult]:
    """Sliding-window DINOv3 matcher.

    Parameters
    ----------
    stride_frac : fraction of the reference size used as the window stride.
        0.5 ≈ half-symbol overlap between windows; smaller = denser, slower.
    threshold : minimum cosine similarity to keep as a match.
    rotations : reference rotations tried; we take the max cosine across them.
    """
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if page.ndim == 3:
        page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # Reference embedding per rotation.
    ref_imgs = []
    deg_for_idx = []
    for deg in rotations:
        if deg == 0:
            r = reference
        elif deg == 90:
            r = np.rot90(reference, 1)
        elif deg == 180:
            r = np.rot90(reference, 2)
        elif deg == 270:
            r = np.rot90(reference, 3)
        else:
            continue
        ref_imgs.append(r)
        deg_for_idx.append(deg)
    ref_embs = embed.encode_crops(ref_imgs)  # (R, D)

    rh, rw = reference.shape
    stride = max(8, int(round(min(rh, rw) * stride_frac)))
    ys, xs = _windows(page.shape, rh, rw, stride)

    # Batch windows.
    hits: list[S3Hit] = []
    batch: list[np.ndarray] = []
    batch_yx: list[tuple[int, int]] = []

    def flush():
        if not batch:
            return
        embs = embed.encode_crops(batch, batch_size=batch_size)  # (B, D)
        sims = embs @ ref_embs.T  # (B, R)
        best_sim = sims.max(axis=1)
        best_rot_idx = sims.argmax(axis=1)
        for k, (y, x) in enumerate(batch_yx):
            s = float(best_sim[k])
            if s >= threshold:
                hits.append(
                    S3Hit(
                        bbox=BBox(x=x, y=y, w=rw, h=rh),
                        score=s,
                        rotation_deg=int(deg_for_idx[best_rot_idx[k]]),
                    )
                )
        batch.clear()
        batch_yx.clear()

    for y in ys:
        for x in xs:
            crop = page[y : y + rh, x : x + rw]
            batch.append(crop)
            batch_yx.append((y, x))
            if len(batch) >= batch_size:
                flush()
    flush()

    # NMS across all hits (reuse S1's NMS — works on S1Hit shape; adapt by mapping).
    s1_shaped = [
        S1Hit(bbox=h.bbox, score=h.score, rotation_deg=h.rotation_deg, scale=1.0)
        for h in hits
    ]
    kept = _nms(s1_shaped, iou_thresh=nms_iou)

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
            scale=1.0,
        )
        item = DrawingItem(
            id=new_id("item"),
            capture_id=cap.id,
            reference_id=reference_id,
            score=h.score,
            score_breakdown={"dinov3_cosine": h.score},
            stage="rerank",
        )
        results.append(MatchResult(capture=cap, item=item))
    return results
