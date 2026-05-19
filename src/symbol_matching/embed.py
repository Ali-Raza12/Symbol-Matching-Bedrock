"""DINOv3 embedding encoder.

DINOv3 is a self-supervised vision transformer that produces dense
patch embeddings suitable for one-shot visual correspondence. We load
the public checkpoint via Hugging Face. Access is gated, so HF_TOKEN
must be set (or `huggingface-cli login` run) before first use.

- `encode_crop(image)` and `encode_crops(images)` produce a single
  embedding per image (CLS + patch mean), used both for the reference
  symbol and for proposal reranking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# DINOv3 ViT-B/16 fits comfortably in 15 GB. Upgrade to ViT-L/16 for higher
# quality if GPU memory allows.
DINOV3_CHECKPOINT = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@dataclass
class _LoadedModel:
    model: torch.nn.Module
    processor: object
    name: str
    patch_size: int
    embed_dim: int
    device: str


@lru_cache(maxsize=1)
def _load_model(name: str = DINOV3_CHECKPOINT) -> _LoadedModel:
    """Load the DINOv3 ViT. Cached for the process."""
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    patch_size = getattr(model.config, "patch_size", 16)
    embed_dim = getattr(model.config, "hidden_size", 768)
    logger.info("loaded %s on %s (patch=%d, dim=%d)", name, device, patch_size, embed_dim)
    return _LoadedModel(
        model=model, processor=processor, name=name,
        patch_size=patch_size, embed_dim=embed_dim, device=device,
    )


def model_name() -> str:
    return _load_model().name


CROP_INPUT_SIZE = 224  # crops are resized to this square before encoding


@torch.inference_mode()
def encode_crop(image: np.ndarray | Image.Image) -> np.ndarray:
    """Encode a single image crop to one D-dim embedding (CLS + patch mean).

    The crop is resized to CROP_INPUT_SIZE×CROP_INPUT_SIZE first. Small
    symbols (~50-100 px on the source page) need this upsampling to span
    enough patches for cosine similarity to be meaningful — at native size
    they cover fewer than one patch on a downsampled page.
    """
    return encode_crops([image])[0]


@torch.inference_mode()
def encode_crops(
    images: list[np.ndarray | Image.Image],
    batch_size: int = 128,
) -> np.ndarray:
    """Batch-encode crops to (N, D) L2-normalized embeddings.

    Resizing happens on CPU via ``cv2.resize`` (C-SIMD fast), then a single
    bulk transfer to GPU per batch — avoids per-crop CPU↔GPU sync that
    crippled the earlier implementation.
    """
    import cv2

    if not images:
        lm = _load_model()
        return np.zeros((0, lm.embed_dim), dtype=np.float32)

    lm = _load_model()
    mean_t = torch.tensor(_processor_mean(lm.processor), device=lm.device).view(1, 3, 1, 1)
    std_t = torch.tensor(_processor_std(lm.processor), device=lm.device).view(1, 3, 1, 1)

    # Pre-resize everything on CPU, stack to one big np array.
    resized = np.empty((len(images), CROP_INPUT_SIZE, CROP_INPUT_SIZE, 3), dtype=np.uint8)
    for i, im in enumerate(images):
        arr = _to_rgb_array(im)
        if arr.shape[0] != CROP_INPUT_SIZE or arr.shape[1] != CROP_INPUT_SIZE:
            arr = cv2.resize(arr, (CROP_INPUT_SIZE, CROP_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        resized[i] = arr

    outputs = []
    for i in range(0, len(images), batch_size):
        batch_np = resized[i : i + batch_size]  # (B, H, W, 3) uint8
        batch = torch.from_numpy(batch_np).to(lm.device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).float() / 255.0  # (B, 3, H, W)
        batch = (batch - mean_t) / std_t
        out = lm.model(pixel_values=batch)
        last = out.last_hidden_state
        cls = last[:, 0]
        patch_mean = last[:, 1:].mean(dim=1)
        emb = torch.nn.functional.normalize(cls + patch_mean, dim=-1)
        outputs.append(emb.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _to_rgb_array(image: np.ndarray | Image.Image) -> np.ndarray:
    """np.uint8 HxWx3 from any of (PIL, gray ndarray, BGR ndarray)."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1].astype(np.uint8)  # BGR → RGB
    raise ValueError(f"unsupported image shape: {image.shape}")


def _processor_mean(processor) -> np.ndarray:
    m = getattr(processor, "image_mean", None) or [0.485, 0.456, 0.406]
    return np.array(m, dtype=np.float32)


def _processor_std(processor) -> np.ndarray:
    s = getattr(processor, "image_std", None) or [0.229, 0.224, 0.225]
    return np.array(s, dtype=np.float32)
