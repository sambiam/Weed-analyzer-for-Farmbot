"""Plant-focused standard/diagnostic review artifact construction."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .plant_measurement import (
    parse_transform,
    relative_pixel_to_plant_mm_transform,
    select_measurement_evidence,
    selection_diagnostics,
)


def _grid_window(selection, grid_record) -> tuple[set[int], dict[str, int]] | None:
    if grid_record is None or not selection.used:
        return None
    frames_by_image = {int(frame.image_id): frame for frame in grid_record.frames}
    targets_by_index = {int(target.index): target for target in grid_record.targets}
    used_targets = [
        targets_by_index[frames_by_image[int(item.image_id)].target_index]
        for item in selection.used
        if int(item.image_id) in frames_by_image
        and frames_by_image[int(item.image_id)].target_index in targets_by_index
    ]
    if len(used_targets) != len(selection.used):
        return None
    rows = [int(target.row) for target in grid_record.targets]
    columns = [int(target.column) for target in grid_record.targets]
    row_min = max(min(rows), min(int(target.row) for target in used_targets) - 1)
    row_max = min(max(rows), max(int(target.row) for target in used_targets) + 1)
    col_min = max(min(columns), min(int(target.column) for target in used_targets) - 1)
    col_max = min(max(columns), max(int(target.column) for target in used_targets) + 1)
    indexes = {
        int(target.index)
        for target in grid_record.targets
        if row_min <= int(target.row) <= row_max and col_min <= int(target.column) <= col_max
    }
    return indexes, {
        "row_min": row_min,
        "row_max": row_max,
        "column_min": col_min,
        "column_max": col_max,
        "rows": row_max - row_min + 1,
        "columns": col_max - col_min + 1,
    }


def build_plant_review(
    measurements: list,
    output_path: Path,
    overlay_output_path: Path | None = None,
    *,
    plants: list | None = None,
    proposed_radii: dict[int, float] | None = None,
    grid_record=None,
) -> bool:
    """Write geometry-identical standard and target-mask diagnostic images."""

    selection = select_measurement_evidence(measurements)
    if not selection.used:
        return False
    grid_window = (
        None if selection.mode == "single_complete" else _grid_window(selection, grid_record)
    )
    allowed_image_ids: set[int]
    window_details: dict[str, int] | None = None
    theoretical_targets = []
    if grid_window is not None:
        allowed_target_indexes, window_details = grid_window
        frames_by_target = {
            int(frame.target_index): frame
            for frame in grid_record.frames
            if int(frame.target_index) in allowed_target_indexes
        }
        allowed_image_ids = {int(frame.image_id) for frame in frames_by_target.values()}
        theoretical_targets = [
            target for target in grid_record.targets if int(target.index) in allowed_target_indexes
        ]
    else:
        allowed_image_ids = {int(item.image_id) for item in selection.used}

    frames: list[dict] = []
    for item in measurements:
        if int(item.image_id) not in allowed_image_ids or not item.source_image_path:
            continue
        image = cv2.imread(item.source_image_path, cv2.IMREAD_COLOR)
        relative = relative_pixel_to_plant_mm_transform(item)
        if image is None or relative is None:
            continue
        transform, scale = relative
        height, width = image.shape[:2]
        corners = cv2.transform(
            np.float64([[[0, 0], [width, 0], [0, height], [width, height]]]),
            transform,
        )[0]
        frames.append(
            {
                "item": item,
                "image": image,
                "transform": transform,
                "scale": scale,
                "bounds": (
                    float(corners[:, 0].min()),
                    float(corners[:, 0].max()),
                    float(corners[:, 1].min()),
                    float(corners[:, 1].max()),
                ),
            }
        )
    if not frames:
        return False

    min_x = min(frame["bounds"][0] for frame in frames)
    max_x = max(frame["bounds"][1] for frame in frames)
    min_y = min(frame["bounds"][2] for frame in frames)
    max_y = max(frame["bounds"][3] for frame in frames)
    # Preserve empty cells inside the expanded grid rectangle. Their theoretical
    # footprint extends the crop, but no neighbouring image is stretched into
    # the unavailable space.
    if theoretical_targets:
        prototype = frames[0]
        transform_meta = parse_transform(prototype["item"])
        image_x = transform_meta.get("image_x")
        image_y = transform_meta.get("image_y")
        if image_x is not None and image_y is not None:
            base = prototype["bounds"]
            for target in theoretical_targets:
                shift_x = float(target.x) - float(image_x)
                shift_y = float(target.y) - float(image_y)
                min_x = min(min_x, base[0] + shift_x)
                max_x = max(max_x, base[1] + shift_x)
                min_y = min(min_y, base[2] + shift_y)
                max_y = max(max_y, base[3] + shift_y)
    else:
        current = float(selection.used[0].current_radius_mm)
        proposed = max(float(item.recommended_protection_radius_mm) for item in selection.used)
        focus = max(60.0, current, proposed) * 2.5
        focused = (
            max(min_x, -focus),
            min(max_x, focus),
            max(min_y, -focus),
            min(max_y, focus),
        )
        if focused[1] - focused[0] > 1 and focused[3] - focused[2] > 1:
            min_x, max_x, min_y, max_y = focused

    ppm = float(np.median([frame["scale"] for frame in frames]))
    ppm = min(ppm, 2400 / max(1.0, max(max_x - min_x, max_y - min_y)))
    canvas_width = max(1, round((max_x - min_x) * ppm))
    canvas_height = max(1, round((max_y - min_y) * ppm))
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    priority = np.full((canvas_height, canvas_width), -np.inf, dtype=np.float32)
    ownership = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    newest = max(item.image_timestamp for item in measurements)
    for frame in sorted(frames, key=lambda frame: frame["item"].image_timestamp):
        affine = frame["transform"].copy()
        affine[0] *= ppm
        affine[1] *= ppm
        affine[0, 2] -= min_x * ppm
        affine[1, 2] -= min_y * ppm
        size = (canvas_width, canvas_height)
        warped = cv2.warpAffine(
            frame["image"],
            affine,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.warpAffine(
            np.full(frame["image"].shape[:2], 255, dtype=np.uint8),
            affine,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        age_seconds = max(0.0, (newest - frame["item"].image_timestamp).total_seconds())
        score = float(frame["item"].image_quality) + 1e-6 / (1.0 + age_seconds)
        selected_pixels = (valid > 0) & (score >= priority)
        canvas[selected_pixels] = warped[selected_pixels]
        priority[selected_pixels] = score
        if frame["item"] in selection.used and frame["item"].mask_path:
            mask = cv2.imread(frame["item"].mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                warped_mask = cv2.warpAffine(
                    mask,
                    affine,
                    size,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                ownership = cv2.max(ownership, warped_mask)

    diagnostic = canvas.copy()
    target_pixels = ownership > 0
    diagnostic[target_pixels] = (
        diagnostic[target_pixels].astype(np.float32) * 0.55
        + np.asarray((255, 190, 20), dtype=np.float32) * 0.45
    ).astype(np.uint8)

    representative = selection.used[0]
    target_x = float(representative.recorded_center_x or 0)
    target_y = float(representative.recorded_center_y or 0)
    proposals = proposed_radii or {}
    current_target = float(representative.current_radius_mm)
    proposed_target = float(
        representative.fused_recommended_radius_mm
        if representative.fused_recommended_radius_mm is not None
        else representative.recommended_protection_radius_mm
    )

    overlay_plants = list(plants or [])
    if not any(int(plant.id) == int(representative.plant_id) for plant in overlay_plants):
        overlay_plants.append(
            type(
                "ReviewPlant",
                (),
                {
                    "id": representative.plant_id,
                    "name": representative.crop_slug,
                    "openfarm_slug": representative.crop_slug,
                    "x": target_x,
                    "y": target_y,
                    "radius": current_target,
                },
            )()
        )

    def annotate(target: np.ndarray) -> None:
        thickness = max(2, round(min(canvas_width, canvas_height) / 350))
        for plant in overlay_plants:
            px = round((float(plant.x) - target_x - min_x) * ppm)
            py = round((float(plant.y) - target_y - min_y) * ppm)
            if not (0 <= px < canvas_width and 0 <= py < canvas_height):
                continue
            plant_id = int(plant.id)
            current = (
                current_target if plant_id == int(representative.plant_id) else float(plant.radius)
            )
            proposed = (
                proposed_target
                if plant_id == int(representative.plant_id)
                else proposals.get(plant_id)
            )
            cv2.circle(
                target,
                (px, py),
                max(1, round(current * ppm)),
                (255, 255, 0),
                thickness,
            )
            if proposed is not None:
                cv2.circle(
                    target,
                    (px, py),
                    max(1, round(float(proposed) * ppm)),
                    (0, 0, 255),
                    thickness + 1,
                )
            cv2.drawMarker(
                target,
                (px, py),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                max(12, thickness * 5),
                max(1, thickness),
            )
            label = str(
                getattr(plant, "name", None)
                or getattr(plant, "openfarm_slug", None)
                or f"plant {plant_id}"
            )
            cv2.putText(
                target,
                label,
                (px + 6, max(16, py - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.42, min(0.8, canvas_width / 1500)),
                (20, 20, 20),
                max(3, thickness + 1),
                cv2.LINE_AA,
            )
            cv2.putText(
                target,
                label,
                (px + 6, max(16, py - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.42, min(0.8, canvas_width / 1500)),
                (255, 255, 255),
                max(1, thickness),
                cv2.LINE_AA,
            )

    annotate(canvas)
    annotate(diagnostic)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_written = bool(cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]))
    metadata = selection_diagnostics(selection) | {
        "crop_mm": [min_x, min_y, max_x, max_y],
        "pixel_dimensions": [canvas_width, canvas_height],
        "pixels_per_mm": ppm,
        "tile_window": window_details,
        "standard_and_diagnostic_geometry_identical": True,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    if overlay_output_path is None:
        return clean_written
    overlay_written = bool(
        cv2.imwrite(
            str(overlay_output_path),
            diagnostic,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
    )
    return clean_written and overlay_written
