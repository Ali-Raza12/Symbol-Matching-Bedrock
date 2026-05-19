# Symbol Matching on Construction Drawings

Given a user-drawn box around a symbol on one page of a
construction-drawing PDF, find every matching instance across the rest
of the drawing set.

## Approach in one paragraph

A hybrid pipeline (S4 in the notebooks). Stage A is multi-rotation
normalized cross-correlation template matching — the classical baseline —
run at a deliberately loose threshold to over-generate candidates and
protect recall. Stage B is a DINOv3 cosine-similarity rerank that catches
stylistic variation between instances. Stage C is a DINOv3 dense-patch
heatmap fallback that fires only on pages where Stage A returned little
or nothing, as a false-negative safety net. The classical → deep
decomposition keeps the classical stage in charge of *where* a candidate
lives and the deep stage in charge of *whether* it is the right thing.

Three alternative strategies (pure template, classical keypoints, DINOv3
alone) are implemented in `src/symbol_matching/strategies/` and each has
its own notebook so the "why S4" argument is empirical, not theoretical.

## Setup

```bash
# Use the project virtualenv (Python 3.12, CUDA 12.4)
source /home/ubuntu/Documents/env/bin/activate

# Editable install of this project
pip install -e .
```

## Run

The primary deliverable is the notebooks. Launch Jupyter and open them
in order:

```bash
jupyter lab notebooks/
```

| Notebook | Purpose |
|---|---|
| `Setup_User_Selection.ipynb` | Render the test PDF and pick the user-selected reference symbol used by every downstream notebook |
| `S1_template_matching.ipynb` | Multi-rotation NCC baseline |
| `S2_keypoint_matching.ipynb` | ORB / AKAZE  — demonstrates the line-art failure mode |
| `S3_dinov3_dense.ipynb` | Zero-shot DINOv3 dense-patch matching alone |
| `S4_hybrid.ipynb` | The recommended pipeline (S1 proposals → DINOv3 rerank → S3 fallback), end-to-end demo |
| `Bonus_S4_non_symbol.ipynb` | Bonus: S4 applied to a non-symbol repeating pattern (toilet fixtures across plumbing pages) |

Outputs from an end-to-end S4 run on the supplied CD set are already in
[outputs/s4/](outputs/s4/): annotated PNGs per page plus a `matches.json`
with 188 records.

## Output schema

Each match emits a `drawingItem` + `capture` pair (the spec's terms) as JSON.
A `capture` carries the bounding-box coordinates in page-pixel space plus the
source page metadata; a `drawingItem` carries the match score, the reference
symbol id, and a link to its capture. See
[`src/symbol_matching/records.py`](src/symbol_matching/records.py) for the
dataclasses.

## Layout

```
src/symbol_matching/
├── pdf_io.py             # PyMuPDF render + title-block metadata
├── reference.py          # rotated/scaled template bank from a user crop
├── strategies/
│   ├── s1_template.py    # multi-rotation NCC
│   ├── s2_keypoint.py    # ORB/AKAZE/SIFT + RANSAC
│   ├── s3_dinov3.py      # DINOv3 dense-patch heatmap
│   └── s4_hybrid.py      # S1 (loose) → DINOv3 rerank → S3 fallback
├── embed.py              # DINOv3 loader + batched crop encoder
├── benchmark.py          # canonical reference + approximate GT
├── records.py            # drawingItem + capture JSON
├── viz.py                # annotated PNG output
└── eval.py               # recall/precision against GT
```
