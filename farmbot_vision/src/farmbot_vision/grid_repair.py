"""Detect and repair incomplete FarmBot photo-grid runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, field_validator

from .models import InventoryImage

CLUSTER_WINDOW = timedelta(hours=1)
COORDINATE_TOLERANCE_MM = 25.0
MINIMUM_GRID_COVERAGE = 0.6

# The companion integration's start_vision_grid_repair service accepts one to
# twelve targets per call (see docs/integration-contract.md); a larger grid
# with more missing/gantry cells than this must be repaired across more than
# one call.
MAX_REPAIR_TARGETS_PER_CALL = 12


class GridRepairSettings(BaseModel):
    """User-owned settings persisted independently of Supervisor options."""

    enabled: bool = True
    delay_minutes: int = 5

    @field_validator("delay_minutes")
    @classmethod
    def valid_delay(cls, value: int) -> int:
        if not 1 <= value <= 1440:
            raise ValueError("delay_minutes must be between 1 and 1440")
        return value


class GridRepairSettingsStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> GridRepairSettings:
        if not self.path.exists():
            return GridRepairSettings()
        return GridRepairSettings.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, value: GridRepairSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value.model_dump(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


@dataclass(frozen=True)
class RepairTarget:
    x: float
    y: float
    z: float
    reason: str
    image_id: int | None = None


@dataclass(frozen=True)
class GridRun:
    started_at: datetime
    completed_at: datetime
    images: tuple[InventoryImage, ...]
    expected_count: int
    coverage: float
    targets: tuple[RepairTarget, ...]


def _axis(values: list[float]) -> list[float]:
    """Snap small coordinate noise into stable grid-axis positions."""
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or abs(value - sum(groups[-1]) / len(groups[-1])) > COORDINATE_TOLERANCE_MM:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _nearest(value: float, axis: list[float]) -> float:
    return min(axis, key=lambda candidate: abs(candidate - value))


def _time_clusters(images: list[InventoryImage]) -> list[list[InventoryImage]]:
    ordered = sorted(images, key=lambda item: item.created_at)
    clusters: list[list[InventoryImage]] = []
    for image in ordered:
        if not clusters or image.created_at - clusters[-1][0].created_at > CLUSTER_WINDOW:
            clusters.append([image])
        else:
            clusters[-1].append(image)
    return clusters


def detect_latest_grid_run(
    images: list[InventoryImage], gantry_image_ids: set[int] | None = None
) -> GridRun | None:
    """Return the newest hour-scale cluster that forms most of a 2-D grid."""
    gantry_image_ids = gantry_image_ids or set()
    for cluster in reversed(_time_clusters(images)):
        if len(cluster) < 4:
            continue
        xs = _axis([item.meta.x for item in cluster])
        ys = _axis([item.meta.y for item in cluster])
        if len(xs) < 2 or len(ys) < 2:
            continue
        cells: dict[tuple[float, float], InventoryImage] = {}
        for image in cluster:
            cell = (_nearest(image.meta.x, xs), _nearest(image.meta.y, ys))
            current = cells.get(cell)
            if current is None or image.created_at > current.created_at:
                cells[cell] = image
        expected = len(xs) * len(ys)
        coverage = len(cells) / expected
        if coverage < MINIMUM_GRID_COVERAGE:
            continue
        ordinary_z = [image.meta.z for image in cluster if image.id not in gantry_image_ids] or [
            image.meta.z for image in cluster
        ]
        typical_z = float(np.median(ordinary_z))
        targets: list[RepairTarget] = []
        for x in xs:
            for y in ys:
                image = cells.get((x, y))
                if image is None:
                    targets.append(RepairTarget(x, y, typical_z, "missing"))
                # Metal garden-bed edging is easily mistaken for the gantry in
                # perimeter photos. Only a positive in a fully interior cell
                # is repair evidence; all classifier positives remain
                # available to the dashboard's debug viewer.
                elif (
                    image.id in gantry_image_ids
                    and x not in {xs[0], xs[-1]}
                    and y not in {ys[0], ys[-1]}
                ):
                    targets.append(RepairTarget(x, y, image.meta.z, "gantry", image.id))
        return GridRun(
            started_at=min(item.created_at for item in cluster),
            completed_at=max(item.created_at for item in cluster),
            images=tuple(cluster),
            expected_count=expected,
            coverage=coverage,
            targets=tuple(targets),
        )
    return None


def looks_like_gantry_photo(jpeg: bytes, name: str | None = None) -> bool:
    """Conservatively detect the long, bright gantry rail in a garden photo.

    A metadata name is authoritative when FarmBot supplies one. The image
    fallback requires two strong, near-vertical edges around a bright,
    low-saturation strip spanning most of the frame, avoiding ordinary stems.
    """
    if name and "gantry" in name.casefold():
        return True
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or min(frame.shape[:2]) < 80:
        return False
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / width)
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
        height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 155)).astype(np.uint8)
    candidates = np.flatnonzero(bright.mean(axis=0) > 0.75)
    if not len(candidates):
        return False
    minimum_width = max(6, round(width * 0.018))
    maximum_width = max(minimum_width + 1, round(width * 0.18))
    for group in np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1):
        if not minimum_width <= len(group) <= maximum_width:
            continue
        left, right = int(group[0]), int(group[-1])
        margin = max(2, round(width * 0.008))
        if left < margin or right + margin >= width:
            continue
        left_edge = np.abs(gray[:, left].astype(np.int16) - gray[:, left - margin].astype(np.int16))
        right_edge = np.abs(
            gray[:, right].astype(np.int16) - gray[:, right + margin].astype(np.int16)
        )
        strip = bright[:, left : right + 1]
        if strip.mean() > 0.8 and (left_edge > 24).mean() > 0.6 and (right_edge > 24).mean() > 0.6:
            return True
    return False


def target_payload(targets: tuple[RepairTarget, ...]) -> list[dict[str, float]]:
    return [
        {"x": target.x, "y": target.y, "z": target.z}
        for target in targets
        if all(math.isfinite(value) for value in (target.x, target.y, target.z))
    ]
