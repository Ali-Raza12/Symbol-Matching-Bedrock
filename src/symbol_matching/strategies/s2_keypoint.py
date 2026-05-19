"""S2 — Classical keypoint matching (ORB / AKAZE / SIFT).

Detect keypoints + descriptors on the reference and the page, BFMatcher
with Lowe's ratio test, cluster matches by spatial proximity on the page,
fit a RANSAC homography per cluster to get a bbox.

"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..records import BBox, Capture, DrawingItem, MatchResult, new_id


@dataclass
class S2Hit:
    bbox: BBox
    score: float


def _detector(kind: str):
    kind = kind.lower()
    if kind == "orb":
        return cv2.ORB_create(nfeatures=5000)
    if kind == "akaze":
        return cv2.AKAZE_create()
    if kind == "sift":
        return cv2.SIFT_create(nfeatures=5000)
    raise ValueError(f"unknown detector: {kind}")


def _match_descriptors(desc_ref, desc_page, norm_type):
    if desc_ref is None or desc_page is None or len(desc_page) < 2:
        return []
    bf = cv2.BFMatcher(norm_type)
    knn = bf.knnMatch(desc_ref, desc_page, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def _cluster_and_localize(
    good_matches, kp_page, ref_shape, *, eps: float = 60.0, min_samples: int = 4
):
    """Group matched keypoints on the page into spatial clusters → bbox per cluster."""
    if len(good_matches) < min_samples:
        return []
    pts = np.array([kp_page[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    # Cheap DBSCAN-equivalent via greedy spatial grouping.
    clusters: list[list[int]] = []
    used = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if used[i]:
            continue
        d = np.linalg.norm(pts - pts[i], axis=1)
        idx = np.where(d <= eps)[0]
        if len(idx) >= min_samples:
            for j in idx:
                used[j] = True
            clusters.append(list(idx))
    hits: list[S2Hit] = []
    rh, rw = ref_shape[:2]
    for cluster in clusters:
        cluster_pts = pts[cluster]
        x = int(cluster_pts[:, 0].mean() - rw / 2)
        y = int(cluster_pts[:, 1].mean() - rh / 2)
        score = min(1.0, len(cluster) / 10.0)
        hits.append(
            S2Hit(
                bbox=BBox(x=max(x, 0), y=max(y, 0), w=int(rw), h=int(rh)),
                score=float(score),
            )
        )
    return hits


def find_matches_on_page(
    page: np.ndarray,
    reference: np.ndarray,
    *,
    detector: str = "akaze",
    page_idx: int = 0,
    page_meta: dict | None = None,
    reference_id: str = "ref",
) -> list[MatchResult]:
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if page.ndim == 3:
        page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    det = _detector(detector)
    kp_ref, desc_ref = det.detectAndCompute(reference, None)
    kp_page, desc_page = det.detectAndCompute(page, None)

    norm = cv2.NORM_L2 if detector.lower() == "sift" else cv2.NORM_HAMMING
    matches = _match_descriptors(desc_ref, desc_page, norm)
    hits = _cluster_and_localize(matches, kp_page, ref_shape=reference.shape)

    meta = page_meta or {}
    results: list[MatchResult] = []
    for h in hits:
        cap = Capture(
            id=new_id("cap"),
            source_page_id=meta.get("sheet_ref") or f"page-{page_idx}",
            page_idx=page_idx,
            page_name=meta.get("page_name"),
            sheet_ref=meta.get("sheet_ref"),
            page_type=meta.get("page_type"),
            bbox=h.bbox,
            rotation_deg=0,
            scale=1.0,
        )
        item = DrawingItem(
            id=new_id("item"),
            capture_id=cap.id,
            reference_id=reference_id,
            score=h.score,
            score_breakdown={"cluster_keypoints": h.score},
            stage="template",
        )
        results.append(MatchResult(capture=cap, item=item))
    return results
