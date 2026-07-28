from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np

FEATURE_NAMES = (
    "log_area_mm2",
    "aspect_ratio",
    "extent",
    "solidity",
    "circularity",
    "mean_hue",
    "mean_saturation",
    "mean_value",
    "mean_excess_green",
    "strong_green_fraction",
    "edge_density",
    "texture",
    "context_green_fraction",
    "context_orange_fraction",
    "context_neutral_fraction",
    "context_saturation",
)

POSITIVE_LABEL = "weed"
# The legacy grouped labels remain valid for existing training samples. The
# review UI now offers the more useful individual hard-negative categories.
NEGATIVE_LABELS = {
    "crop",
    "mushroom",
    "moss",
    "soil",
    "hardware",
    "mulch_soil",
    "fungus_moss",
    "hardware_other",
}
ALL_LABELS = {POSITIVE_LABEL, *NEGATIVE_LABELS}


def extract_visual_features(
    image: np.ndarray,
    component: np.ndarray,
    area_mm2: float,
    *,
    green_hue_min: int = 25,
    green_hue_max: int = 100,
    strong_green_minimum_saturation: int = 45,
    strong_green_minimum_excess_green: int = 20,
) -> dict[str, float]:
    """Return stable, resolution-independent visual and component features."""
    ys, xs = np.where(component)
    if not len(xs):
        return {name: 0.0 for name in FEATURE_NAMES}
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    area_px = float(len(xs))
    component_u8 = component.astype(np.uint8) * 255
    contours, _ = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    all_points = np.column_stack((xs, ys)).astype(np.int32).reshape(-1, 1, 2)
    hull_area = float(cv2.contourArea(cv2.convexHull(all_points))) if len(all_points) >= 3 else 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = image[component].astype(np.float32)
    hsv_pixels = hsv[component].astype(np.float32)
    b, g, r = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    excess_green = 2 * g - r - b
    strong_green = (
        (hsv_pixels[:, 0] >= green_hue_min)
        & (hsv_pixels[:, 0] <= green_hue_max)
        & (hsv_pixels[:, 1] >= strong_green_minimum_saturation)
        & (excess_green >= strong_green_minimum_excess_green)
    )
    crop = image[y0:y1, x0:x1]
    crop_component = component[y0:y1, x0:x1]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    context_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    context_hue = context_hsv[:, :, 0]
    context_saturation = context_hsv[:, :, 1]
    edges = cv2.Canny(grey, 60, 140) > 0
    selected_grey = grey[crop_component]
    return {
        "log_area_mm2": float(math.log1p(max(0.0, area_mm2)) / 10.0),
        "aspect_ratio": float(min(12.0, max(width, height) / max(1, min(width, height))) / 12.0),
        "extent": float(min(1.0, area_px / (width * height))),
        "solidity": float(min(1.0, area_px / max(1.0, hull_area))),
        "circularity": float(min(1.0, (4.0 * math.pi * area_px) / max(1.0, perimeter * perimeter))),
        "mean_hue": float(np.mean(hsv_pixels[:, 0]) / 179.0),
        "mean_saturation": float(np.mean(hsv_pixels[:, 1]) / 255.0),
        "mean_value": float(np.mean(hsv_pixels[:, 2]) / 255.0),
        "mean_excess_green": float(np.clip(np.mean(excess_green) / 255.0, -1.0, 1.0)),
        "strong_green_fraction": float(np.mean(strong_green)),
        "edge_density": float(np.mean(edges)),
        "texture": float(np.std(selected_grey) / 128.0) if len(selected_grey) else 0.0,
        "context_green_fraction": float(
            np.mean(
                (context_hue >= green_hue_min)
                & (context_hue <= green_hue_max)
                & (context_saturation >= strong_green_minimum_saturation)
            )
        ),
        "context_orange_fraction": float(
            np.mean((context_hue >= 4) & (context_hue <= 24) & (context_saturation >= 40))
        ),
        "context_neutral_fraction": float(np.mean(context_saturation < 35)),
        "context_saturation": float(np.mean(context_saturation) / 255.0),
    }


def encode_candidate_crop(
    image: np.ndarray, component: np.ndarray, *, size: int = 96
) -> bytes | None:
    ys, xs = np.where(component)
    if not len(xs):
        return None
    height, width = image.shape[:2]
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    pad = max(8, round(max(x1 - x0, y1 - y0) * 0.4))
    x0, x1 = max(0, x0 - pad), min(width, x1 + pad)
    y0, y1 = max(0, y0 - pad), min(height, y1 + pad)
    crop = image[y0:y1, x0:x1]
    if not crop.size:
        return None
    scale = min(size / crop.shape[1], size / crop.shape[0])
    resized = cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    oy = (size - resized.shape[0]) // 2
    ox = (size - resized.shape[1]) // 2
    canvas[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return encoded.tobytes() if ok else None


class WeedVisualVerifier:
    """Small locally trained logistic visual verifier with atomic JSON persistence."""

    def __init__(self, path: Path):
        self.path = path
        self.model: dict[str, Any] | None = None
        self.reload()

    @property
    def available(self) -> bool:
        return self.model is not None

    def reload(self) -> None:
        try:
            candidate = json.loads(self.path.read_text(encoding="utf-8"))
            if candidate.get("feature_names") == list(FEATURE_NAMES):
                self.model = candidate
            else:
                self.model = None
        except (OSError, ValueError, TypeError):
            self.model = None

    def clear(self) -> None:
        """Forget the active model and remove its persisted file."""
        self.model = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def predict(self, features: dict[str, float]) -> float | None:
        if self.model is None:
            return None
        vector = np.asarray([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=float)
        mean = np.asarray(self.model["mean"], dtype=float)
        scale = np.asarray(self.model["scale"], dtype=float)
        weights = np.asarray(self.model["weights"], dtype=float)
        score = float(np.dot((vector - mean) / scale, weights) + self.model["bias"])
        score = max(-30.0, min(30.0, score))
        return 1.0 / (1.0 + math.exp(-score))

    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray, iterations: int = 900) -> tuple[np.ndarray, float]:
        weights = np.zeros(x.shape[1], dtype=float)
        bias = 0.0
        positives = max(1, int(np.count_nonzero(y == 1)))
        negatives = max(1, int(np.count_nonzero(y == 0)))
        sample_weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
        for step in range(iterations):
            logits = np.clip(x @ weights + bias, -30, 30)
            predictions = 1.0 / (1.0 + np.exp(-logits))
            errors = (predictions - y) * sample_weights
            learning_rate = 0.12 / (1.0 + step / 300.0)
            weights -= learning_rate * ((x.T @ errors) / len(y) + 0.015 * weights)
            bias -= learning_rate * float(np.mean(errors))
        return weights, bias

    def train(self, samples: list[dict[str, Any]], minimum_per_class: int) -> dict[str, Any]:
        usable = [
            sample
            for sample in samples
            if sample.get("label") in ALL_LABELS and isinstance(sample.get("features"), dict)
        ]
        positives = [sample for sample in usable if sample["label"] == POSITIVE_LABEL]
        negatives = [sample for sample in usable if sample["label"] in NEGATIVE_LABELS]
        if len(positives) < minimum_per_class or len(negatives) < minimum_per_class:
            raise ValueError(
                f"Need at least {minimum_per_class} weed and {minimum_per_class} non-weed labels; "
                f"currently have {len(positives)} and {len(negatives)}"
            )
        ordered = sorted(
            usable,
            key=lambda sample: (
                str(sample.get("created_at", "")),
                str(sample.get("detection_id", "")),
            ),
        )
        raw_x = np.asarray(
            [
                [float(sample["features"].get(name, 0.0)) for name in FEATURE_NAMES]
                for sample in ordered
            ]
        )
        y = np.asarray([1.0 if sample["label"] == POSITIVE_LABEL else 0.0 for sample in ordered])
        mean = raw_x.mean(axis=0)
        scale = raw_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        x = (raw_x - mean) / scale

        validation_indices = [
            index
            for index, sample in enumerate(ordered)
            if index % 5 == 0 and sum(s["label"] == sample["label"] for s in ordered) >= 5
        ]
        train_indices = [index for index in range(len(ordered)) if index not in validation_indices]
        if not validation_indices or len({y[index] for index in train_indices}) < 2:
            validation_indices = list(range(len(ordered)))
            train_indices = list(range(len(ordered)))
        validation_weights, validation_bias = self._fit(x[train_indices], y[train_indices])
        scores = 1.0 / (
            1.0
            + np.exp(
                -np.clip(x[validation_indices] @ validation_weights + validation_bias, -30, 30)
            )
        )
        predicted = scores >= 0.5
        actual = y[validation_indices] == 1
        tp = int(np.count_nonzero(predicted & actual))
        fp = int(np.count_nonzero(predicted & ~actual))
        fn = int(np.count_nonzero(~predicted & actual))
        tn = int(np.count_nonzero(~predicted & ~actual))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        weights, bias = self._fit(x, y)
        model = {
            "version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "feature_names": list(FEATURE_NAMES),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "weights": weights.tolist(),
            "bias": bias,
            "sample_count": len(ordered),
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "validation_samples": len(validation_indices),
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            json.dump(model, handle, indent=2)
            temp = Path(handle.name)
        os.replace(temp, self.path)
        self.model = model
        return model
