"""Recall / precision against hand-marked ground truth.
Ground truth is a list of bboxes per page (hand-marked in a notebook cell).
A detection counts as a true positive if its IoU with any unmatched GT box is >= the
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from .records import BBox, MatchResult


@dataclass
class EvalResult:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
        }


def iou(a: BBox, b: BBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def evaluate(
    matches: list[MatchResult],
    ground_truth: list[BBox],
    iou_thresh: float = 0.3,
) -> EvalResult:
    """Greedy match: each detection consumes at most one GT box."""
    unmatched_gt = list(range(len(ground_truth)))
    tp = 0
    fp = 0
    # Highest-score detections get to claim GT first.
    for m in sorted(matches, key=lambda x: x.item.score, reverse=True):
        best_iou = 0.0
        best_idx = -1
        for gi in unmatched_gt:
            v = iou(m.capture.bbox, ground_truth[gi])
            if v > best_iou:
                best_iou = v
                best_idx = gi
        if best_iou >= iou_thresh:
            tp += 1
            unmatched_gt.remove(best_idx)
        else:
            fp += 1
    fn = len(unmatched_gt)
    return EvalResult(tp=tp, fp=fp, fn=fn)
