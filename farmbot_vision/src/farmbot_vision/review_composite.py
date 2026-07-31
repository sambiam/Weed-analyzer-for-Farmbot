"""Plant-focused standard/diagnostic review artifact construction."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .photo_grid import farmbot_cropped_footprint, photo_grid_cell_bounds
from .plant_measurement import (
    relative_pixel_to_plant_mm_transform,
    select_measurement_evidence,
    selection_diagnostics,
)


def _grid_frames(grid_record) -> list:
    return [
        *grid_record.frames,
        *getattr(grid_record, "quality_overlay_frames", []),
    ]


def _grid_window(
    selection,
    grid_record,
    target_x: float,
    target_y: float,
) -> tuple[set[int], dict[str, int]] | None:
    if grid_record is None:
        return None
    frames_by_image = {int(frame.image_id): frame for frame in _grid_frames(grid_record)}
    targets_by_index = {int(target.index): target for target in grid_record.targets}
    used_targets = [
        targets_by_index[frames_by_image[int(item.image_id)].target_index]
        for item in selection.used
        if int(item.image_id) in frames_by_image
        and frames_by_image[int(item.image_id)].target_index in targets_by_index
    ]
    if selection.used and len(used_targets) != len(selection.used):
        return None
    cells = photo_grid_cell_bounds(grid_record)
    target_cell = next(
        (
            targets_by_index[index]
            for index, bounds in cells.items()
            if index in targets_by_index
            and bounds[0] <= target_x <= bounds[2]
            and bounds[1] <= target_y <= bounds[3]
        ),
        None,
    )
    if target_cell is None and targets_by_index:
        target_cell = min(
            targets_by_index.values(),
            key=lambda target: (float(target.x) - target_x) ** 2
            + (float(target.y) - target_y) ** 2,
        )
    anchors = [*used_targets, *([target_cell] if target_cell is not None else [])]
    if not anchors:
        return None
    rows = [int(target.row) for target in grid_record.targets]
    columns = [int(target.column) for target in grid_record.targets]
    row_min = max(min(rows), min(int(target.row) for target in anchors) - 1)
    row_max = min(max(rows), max(int(target.row) for target in anchors) + 1)
    col_min = max(min(columns), min(int(target.column) for target in anchors) - 1)
    col_max = min(max(columns), max(int(target.column) for target in anchors) + 1)
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
    if not selection.candidates:
        return False
    representative = selection.used[0] if selection.used else selection.candidates[0]
    target_x = float(representative.recorded_center_x or 0)
    target_y = float(representative.recorded_center_y or 0)
    grid_window = _grid_window(selection, grid_record, target_x, target_y)
    if not selection.used and grid_window is None:
        return False
    allowed_image_ids: set[int]
    window_details: dict[str, int] | None = None
    theoretical_targets = []
    if grid_window is not None:
        allowed_target_indexes, window_details = grid_window
        allowed_image_ids = {
            int(frame.image_id)
            for frame in _grid_frames(grid_record)
            if int(frame.target_index) in allowed_target_indexes
        }
        theoretical_targets = [
            target for target in grid_record.targets if int(target.index) in allowed_target_indexes
        ]
    else:
        allowed_image_ids = {int(item.image_id) for item in selection.used}

    grid_frames_by_image = (
        {int(frame.image_id): frame for frame in _grid_frames(grid_record)}
        if grid_record is not None
        else {}
    )
    grid_cells = photo_grid_cell_bounds(grid_record) if grid_record is not None else {}
    grid_footprint = (
        farmbot_cropped_footprint(grid_record.calibration) if grid_record is not None else None
    )
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
        grid_frame = grid_frames_by_image.get(int(item.image_id))
        absolute_cell = (
            grid_cells.get(int(grid_frame.target_index)) if grid_frame is not None else None
        )
        if absolute_cell is not None and grid_footprint is not None:
            optical_x = float(grid_frame.x) + float(grid_record.calibration.offset_x_mm)
            optical_y = float(grid_frame.y) + float(grid_record.calibration.offset_y_mm)
            safe_cell = (
                optical_x - grid_footprint[0] / 2,
                optical_y - grid_footprint[1] / 2,
                optical_x + grid_footprint[0] / 2,
                optical_y + grid_footprint[1] / 2,
            )
            absolute_cell = (
                max(absolute_cell[0], safe_cell[0]),
                max(absolute_cell[1], safe_cell[1]),
                min(absolute_cell[2], safe_cell[2]),
                min(absolute_cell[3], safe_cell[3]),
            )
        relative_cell = (
            (
                absolute_cell[0] - target_x,
                absolute_cell[1] - target_y,
                absolute_cell[2] - target_x,
                absolute_cell[3] - target_y,
            )
            if absolute_cell is not None
            else None
        )
        frames.append(
            {
                "item": item,
                "image": image,
                "transform": transform,
                "scale": scale,
                "target_index": (int(grid_frame.target_index) if grid_frame is not None else None),
                "cell_bounds": relative_cell,
                "bounds": (
                    float(corners[:, 0].min()),
                    float(corners[:, 0].max()),
                    float(corners[:, 1].min()),
                    float(corners[:, 1].max()),
                ),
            }
        )
    tessellated_cells: dict[int, tuple[float, float, float, float]] = {}
    if theoretical_targets:
        tessellated_cells = {
            int(target.index): (
                grid_cells[int(target.index)][0] - target_x,
                grid_cells[int(target.index)][1] - target_y,
                grid_cells[int(target.index)][2] - target_x,
                grid_cells[int(target.index)][3] - target_y,
            )
            for target in theoretical_targets
            if int(target.index) in grid_cells
        }

    if tessellated_cells:
        min_x = min(cell[0] for cell in tessellated_cells.values())
        min_y = min(cell[1] for cell in tessellated_cells.values())
        max_x = max(cell[2] for cell in tessellated_cells.values())
        max_y = max(cell[3] for cell in tessellated_cells.values())
    elif frames:
        min_x = min(frame["bounds"][0] for frame in frames)
        max_x = max(frame["bounds"][1] for frame in frames)
        min_y = min(frame["bounds"][2] for frame in frames)
        max_y = max(frame["bounds"][3] for frame in frames)
    else:
        return False

    if not theoretical_targets:
        current = float(representative.current_radius_mm)
        proposed = max(
            float(item.recommended_protection_radius_mm)
            for item in (selection.used or selection.candidates)
        )
        focus = max(60.0, current, proposed) * 2.5
        focused = (
            max(min_x, -focus),
            min(max_x, focus),
            max(min_y, -focus),
            min(max_y, focus),
        )
        if focused[1] - focused[0] > 1 and focused[3] - focused[2] > 1:
            min_x, max_x, min_y, max_y = focused

    ppm = (
        float(np.median([frame["scale"] for frame in frames]))
        if frames
        else min(1.5, 1200 / max(1.0, max(max_x - min_x, max_y - min_y)))
    )
    ppm = min(ppm, 2400 / max(1.0, max(max_x - min_x, max_y - min_y)))
    canvas_width = max(1, round((max_x - min_x) * ppm))
    canvas_height = max(1, round((max_y - min_y) * ppm))
    canvas = (
        np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        if frames
        else np.full((canvas_height, canvas_width, 3), (25, 33, 28), dtype=np.uint8)
    )
    priority = np.full((canvas_height, canvas_width), -np.inf, dtype=np.float32)
    ownership = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    newest = max(item.image_timestamp for item in measurements)
    overlay_image_ids = (
        {int(frame.image_id) for frame in getattr(grid_record, "quality_overlay_frames", [])}
        if grid_record is not None
        else set()
    )
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
        cell_mask = None
        cell_bounds = frame["cell_bounds"]
        if cell_bounds is not None:
            cell_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
            cell_x0 = max(0, round((cell_bounds[0] - min_x) * ppm))
            cell_y0 = max(0, round((cell_bounds[1] - min_y) * ppm))
            cell_x1 = min(canvas_width, round((cell_bounds[2] - min_x) * ppm))
            cell_y1 = min(canvas_height, round((cell_bounds[3] - min_y) * ppm))
            cell_mask[cell_y0:cell_y1, cell_x0:cell_x1] = 255
            valid = cv2.bitwise_and(valid, cell_mask)
        age_seconds = max(0.0, (newest - frame["item"].image_timestamp).total_seconds())
        # A selected leaf-repair view is an explicit top layer. Its offset
        # means it covers only its real calibrated footprint; exposed pixels
        # from the original tile remain visible underneath.
        layer = 2.0 if int(frame["item"].image_id) in overlay_image_ids else 0.0
        score = layer + float(frame["item"].image_quality) + 1e-6 / (1.0 + age_seconds)
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
                if cell_mask is not None:
                    warped_mask = cv2.bitwise_and(warped_mask, cell_mask)
                ownership = cv2.max(ownership, warped_mask)

    diagnostic = canvas.copy()
    target_pixels = ownership > 0
    diagnostic[target_pixels] = (
        diagnostic[target_pixels].astype(np.float32) * 0.55
        + np.asarray((255, 190, 20), dtype=np.float32) * 0.45
    ).astype(np.uint8)

    def draw_tessellation(target: np.ndarray) -> None:
        if not tessellated_cells:
            return
        line_width = max(1, round(min(canvas_width, canvas_height) / 700))
        for cell in tessellated_cells.values():
            x0 = round((cell[0] - min_x) * ppm)
            y0 = round((cell[1] - min_y) * ppm)
            x1 = round((cell[2] - min_x) * ppm)
            y1 = round((cell[3] - min_y) * ppm)
            cv2.rectangle(
                target,
                (x0, y0),
                (max(x0, x1 - 1), max(y0, y1 - 1)),
                (78, 104, 86),
                line_width,
            )
        cv2.rectangle(
            target,
            (0, 0),
            (canvas_width - 1, canvas_height - 1),
            (32, 78, 48),
            max(2, line_width * 2),
        )

    draw_tessellation(canvas)
    draw_tessellation(diagnostic)

    if not frames:
        lines = ("No captured grid photos", "around this plant yet")
        font_scale = max(0.45, min(1.1, canvas_width / 900))
        thickness = max(1, round(font_scale * 2))
        line_height = max(24, round(38 * font_scale))
        baseline_y = canvas_height // 2 - line_height // 2
        for line_index, line in enumerate(lines):
            (text_width, _), _ = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            origin = (
                max(8, (canvas_width - text_width) // 2),
                baseline_y + line_index * line_height,
            )
            for target in (canvas, diagnostic):
                cv2.putText(
                    target,
                    line,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (220, 230, 222),
                    thickness,
                    cv2.LINE_AA,
                )

    current_target = float(representative.current_radius_mm)
    proposed_target = float(
        representative.fused_recommended_radius_mm
        if representative.fused_recommended_radius_mm is not None
        else representative.recommended_protection_radius_mm
    )

    # A plant-radius review must be visually unambiguous.  The wider grid
    # context remains visible, but only the target plant gets a centre marker,
    # label and radius circles; neighbouring circles previously overwhelmed
    # dense beds and made it unclear which proposal the dialog was reviewing.
    overlay_plants = [
        plant for plant in (plants or []) if int(plant.id) == int(representative.plant_id)
    ][:1]
    if not overlay_plants:
        overlay_plants = [
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
        ]

    def annotate(target: np.ndarray) -> None:
        thickness = max(2, round(min(canvas_width, canvas_height) / 350))
        for plant in overlay_plants:
            px = round((float(plant.x) - target_x - min_x) * ppm)
            py = round((float(plant.y) - target_y - min_y) * ppm)
            if not (0 <= px < canvas_width and 0 <= py < canvas_height):
                continue
            plant_id = int(plant.id)
            current = current_target
            proposed = proposed_target
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
        "tessellated_rectangular_cells": bool(tessellated_cells),
        "garden_border": bool(tessellated_cells),
        "blank_free_source_crop": bool(grid_footprint),
        "captured_frame_count": len(frames),
        "missing_photo_context": not frames,
        "standard_and_diagnostic_geometry_identical": True,
        "annotated_plant_ids": [int(representative.plant_id)],
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
