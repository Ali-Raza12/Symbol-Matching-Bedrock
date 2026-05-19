from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BBox:
    """Axis-aligned bounding box in page-pixel coords (origin = top-left)."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class Capture:
    """A geometric region on a page. One per match."""

    id: str
    source_page_id: str  # we use sheet_ref when present, else "page-{idx}"
    page_idx: int  # 0-based page index in source PDF
    page_name: str | None
    sheet_ref: str | None
    page_type: str | None
    bbox: BBox
    rotation_deg: int  # rotation that matched best at this location
    scale: float  # scale that matched best


@dataclass
class DrawingItem:
    """A semantic instance of the queried symbol at a Capture."""

    id: str
    capture_id: str
    reference_id: str  # identifies which queried symbol
    score: float  # final combined score (higher = better)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    stage: Literal["template", "rerank", "fallback"] = "rerank"


@dataclass
class MatchResult:
    """One match = one capture + one drawing item, plus runtime info."""

    capture: Capture
    item: DrawingItem


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def dump_matches(matches: list[MatchResult], out_path: str | Path) -> None:
    """Serialize matches to a single JSON file."""
    payload = [
        {"capture": asdict(m.capture), "drawingItem": asdict(m.item)} for m in matches
    ]
    Path(out_path).write_text(json.dumps(payload, indent=2))
