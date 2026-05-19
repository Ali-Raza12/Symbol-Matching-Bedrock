"""Annotated-PNG output for matches."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .records import MatchResult


def annotate_page(
    page_image: np.ndarray,
    matches: list[MatchResult],
    *,
    color: tuple[int, int, int] = (0, 0, 255),  # BGR red
    thickness: int = 3,
    font_scale: float = 0.6,
    show_score: bool = True,
) -> np.ndarray:
    """Draw match boxes + scores onto a copy of `page_image`. BGR uint8 in/out."""
    if page_image.ndim == 2:
        canvas = cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = page_image.copy()

    for m in matches:
        b = m.capture.bbox
        cv2.rectangle(canvas, (b.x, b.y), (b.x + b.w, b.y + b.h), color, thickness)
        if show_score:
            label = f"{m.item.score:.2f}"
            if m.capture.rotation_deg:
                label += f" @{m.capture.rotation_deg}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            ty = max(b.y - 6, th + 4)
            cv2.rectangle(canvas, (b.x, ty - th - 4), (b.x + tw + 4, ty + 2), color, -1)
            cv2.putText(
                canvas,
                label,
                (b.x + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def save_annotated(
    page_image: np.ndarray,
    matches: list[MatchResult],
    out_path: str | Path,
    **kwargs,
) -> None:
    annotated = annotate_page(page_image, matches, **kwargs)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)


def side_by_side(
    images: list[np.ndarray], titles: list[str] | None = None, pad: int = 20
) -> np.ndarray:
    """Stack images horizontally with a separator. For comparison notebook."""
    h = max(im.shape[0] for im in images)
    bgr = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h))
        bgr.append(im)
    sep = np.full((h, pad, 3), 255, dtype=np.uint8)
    out_parts = []
    for i, im in enumerate(bgr):
        if i > 0:
            out_parts.append(sep)
        out_parts.append(im)
    out = np.concatenate(out_parts, axis=1)

    if titles:
        bar_h = 40
        bar = np.full((bar_h, out.shape[1], 3), 240, dtype=np.uint8)
        x = 0
        for i, (im, title) in enumerate(zip(bgr, titles)):
            if i > 0:
                x += pad
            cv2.putText(
                bar,
                title,
                (x + 8, bar_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            x += im.shape[1]
        out = np.concatenate([bar, out], axis=0)
    return out
