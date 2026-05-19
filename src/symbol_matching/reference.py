"""Build a rotated/scaled bank of templates from the user's reference crop."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_ROTATIONS = (0, 45, 90, 135, 180, 225, 270, 315)
DEFAULT_SCALES = (0.9, 1.0, 1.1)


@dataclass
class Template:
    image: np.ndarray  # HxW grayscale, uint8
    rotation_deg: int
    scale: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]


def build_template_bank(
    reference: np.ndarray,
    rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> list[Template]:
    """Return one Template per (rotation, scale) combination.

    `reference` is grayscale uint8 (the user-cropped reference symbol). Rotations
    use cv2.warpAffine with border replication so corners don't introduce
    spurious edges; scales use cv2.resize with INTER_AREA for downscale and
    INTER_CUBIC for upscale.
    """
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    templates: list[Template] = []
    for rot in rotations:
        rotated = _rotate(reference, rot)
        for scale in scales:
            scaled = _rescale(rotated, scale)
            templates.append(
                Template(image=scaled, rotation_deg=int(rot), scale=float(scale))
            )
    return templates


def _rotate(img: np.ndarray, deg: int) -> np.ndarray:
    if deg % 360 == 0:
        return img.copy()
    if deg % 90 == 0:
        # Lossless 90° rotations via numpy.
        k = (deg // 90) % 4
        return np.rot90(img, k=k).copy()
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    # Compute new canvas size so corners fit.
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    # Fill with the background colour (assume white-ish AEC drawings).
    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _rescale(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img.copy()
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
