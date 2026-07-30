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

# The original sixteen features. Every stored training sample has them, so a
# model can always be trained on this set alone.
CORE_FEATURE_NAMES = (
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

# Spatial-context features added later. Samples labelled by older builds do not
# carry them, so training only uses them when *every* usable sample has them --
# defaulting a missing distance to zero would silently assert "touching a
# crop", which is the opposite of the truth for an unknown sample.
SPATIAL_FEATURE_NAMES = (
    "distance_to_plant",
    "local_vegetation_density",
    "edge_proximity",
)

FEATURE_NAMES = CORE_FEATURE_NAMES + SPATIAL_FEATURE_NAMES
KNOWN_FEATURE_NAMES = frozenset(FEATURE_NAMES)

# Distance at which a candidate is considered "nowhere near a known crop".
# Beyond this the exact value carries no extra information, so it saturates.
PLANT_DISTANCE_SATURATION_MM = 300.0

POSITIVE_LABEL = "weed"
# The legacy grouped labels remain valid for existing training samples. The
# review UI now offers the more useful individual hard-negative categories.
NEGATIVE_LABELS = {
    "crop",
    "fallen_leaf",
    "mushroom",
    "moss",
    "soil",
    "hardware",
    "mulch_soil",
    "fungus_moss",
    "hardware_other",
}
ALL_LABELS = {POSITIVE_LABEL, *NEGATIVE_LABELS}

# Human-readable names for the per-category "best guess" shown during review.
LABEL_DESCRIPTIONS = {
    "weed": "weed",
    "crop": "crop foliage",
    "fallen_leaf": "fallen leaf",
    "mushroom": "mushroom",
    "moss": "moss",
    "soil": "bare soil",
    "hardware": "hardware",
    "mulch_soil": "mulch or soil",
    "fungus_moss": "fungus or moss",
    "hardware_other": "hardware or other",
}

# A per-category head needs enough examples to mean anything. Below this the
# category is folded into the binary decision only.
MINIMUM_SAMPLES_PER_HEAD = 4

# Precision the suggested operating threshold aims for. Weeding acts on the
# garden, so a false positive costs more than a missed weed that will be
# caught on the next pass.
TARGET_PRECISION = 0.95


def extract_visual_features(
    image: np.ndarray,
    component: np.ndarray,
    area_mm2: float,
    *,
    green_hue_min: int = 25,
    green_hue_max: int = 100,
    strong_green_minimum_saturation: int = 45,
    strong_green_minimum_excess_green: int = 20,
    distance_to_plant_mm: float | None = None,
) -> dict[str, float]:
    """Return stable, resolution-independent visual and component features.

    ``distance_to_plant_mm`` is the distance from the candidate to the nearest
    known plant centre. It is the one feature that cannot be derived from the
    crop alone; when the caller has no plant inventory it defaults to the
    saturated "far from any crop" value.
    """
    ys, xs = np.where(component)
    if not len(xs):
        return {name: 0.0 for name in FEATURE_NAMES}
    height_px, width_px = image.shape[:2]
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

    # Neighbourhood vegetation: how much *other* green sits in a padded box
    # around this candidate. An isolated weed in bare soil scores near zero; a
    # leaf tip poking out of a dense canopy scores high.
    pad = max(8, round(max(width, height) * 0.75))
    nx0, nx1 = max(0, x0 - pad), min(width_px, x1 + pad)
    ny0, ny1 = max(0, y0 - pad), min(height_px, y1 + pad)
    neighbourhood = image[ny0:ny1, nx0:nx1]
    neighbourhood_hsv = cv2.cvtColor(neighbourhood, cv2.COLOR_BGR2HSV)
    neighbourhood_bgr = neighbourhood.astype(np.float32)
    neighbourhood_excess_green = (
        2 * neighbourhood_bgr[:, :, 1] - neighbourhood_bgr[:, :, 2] - neighbourhood_bgr[:, :, 0]
    )
    neighbourhood_green = (
        (neighbourhood_hsv[:, :, 0] >= green_hue_min)
        & (neighbourhood_hsv[:, :, 0] <= green_hue_max)
        & (neighbourhood_hsv[:, :, 1] >= strong_green_minimum_saturation)
        & (neighbourhood_excess_green >= strong_green_minimum_excess_green)
    )
    outside = np.ones(neighbourhood_green.shape, dtype=bool)
    outside[y0 - ny0 : y1 - ny0, x0 - nx0 : x1 - nx0] = False
    outside_count = int(np.count_nonzero(outside))
    local_vegetation_density = (
        float(np.count_nonzero(neighbourhood_green & outside) / outside_count)
        if outside_count
        else 0.0
    )

    # How close the candidate is to a frame edge. Truncated blobs entering the
    # frame are disproportionately crop leaves rather than whole weeds.
    edge_gap = min(x0, y0, width_px - x1, height_px - y1)
    edge_proximity = 1.0 - min(1.0, edge_gap / max(1.0, min(width_px, height_px) * 0.1))

    distance = (
        PLANT_DISTANCE_SATURATION_MM if distance_to_plant_mm is None else distance_to_plant_mm
    )
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
        "distance_to_plant": float(
            np.clip(max(0.0, distance) / PLANT_DISTANCE_SATURATION_MM, 0.0, 1.0)
        ),
        "local_vegetation_density": local_vegetation_density,
        "edge_proximity": float(edge_proximity),
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


def _metrics_at(scores: np.ndarray, actual: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.2816) -> float:
    """One-sided lower confidence bound on a proportion (90% by default).

    Three correct predictions out of three is not evidence of 100% precision.
    Judging thresholds on the point estimate makes the sweep pick the lowest
    threshold on the curve whenever a small validation fold happens to be
    separable; the bound shrinks toward zero when the evidence is thin, so a
    threshold has to be supported by enough samples to be recommended.
    """
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - spread)


def _threshold_curve(
    scores: np.ndarray, actual: np.ndarray, target_precision: float
) -> tuple[list[dict[str, float | int]], float]:
    """Sweep operating thresholds and pick the one this add-on should run at.

    A weeding action is destructive, so the suggestion is the highest-recall
    threshold whose precision is *confidently* at or above ``target_precision``
    -- confidence measured by the Wilson lower bound, so a handful of lucky
    predictions cannot justify a permissive threshold. When nothing qualifies,
    fall back to best F1. Ties are broken by taking the middle of the tied
    range rather than either end, because the middle of a plateau is the point
    least likely to move when the next label arrives.
    """
    candidates = sorted({round(float(score), 3) for score in scores} | {0.5})
    curve = []
    for threshold in candidates:
        point = _metrics_at(scores, actual, threshold)
        selected = int(point["true_positive"]) + int(point["false_positive"])
        point["precision_lower_bound"] = round(
            _wilson_lower_bound(int(point["true_positive"]), selected), 4
        )
        curve.append(point)

    qualifying = [
        point
        for point in curve
        if float(point["precision_lower_bound"]) >= target_precision
        and int(point["true_positive"]) > 0
    ]
    if qualifying:
        best_recall = max(float(point["recall"]) for point in qualifying)
        tied = [point for point in qualifying if float(point["recall"]) == best_recall]
    else:
        best_f1 = max(float(point["f1"]) for point in curve)
        tied = [point for point in curve if float(point["f1"]) == best_f1]
    thresholds = sorted(float(point["threshold"]) for point in tied)
    return curve, thresholds[len(thresholds) // 2]


class WeedVisualVerifier:
    """Small locally trained logistic visual verifier with atomic JSON persistence.

    ``bundled_path`` points at an optional model shipped with the add-on. It is
    used only when no locally trained model exists, so a fresh install can
    filter candidates before its owner has labelled anything, and is replaced
    the moment local training succeeds.
    """

    def __init__(self, path: Path, bundled_path: Path | None = None):
        self.path = path
        self.bundled_path = bundled_path
        self.model: dict[str, Any] | None = None
        self.reload()

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def is_bundled(self) -> bool:
        return bool(self.model and self.model.get("source") == "bundled")

    @staticmethod
    def _load(path: Path) -> dict[str, Any] | None:
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        names = candidate.get("feature_names")
        if not isinstance(names, list) or not names or not set(names) <= KNOWN_FEATURE_NAMES:
            return None
        expected = len(names)
        if (
            len(candidate.get("mean", [])) != expected
            or len(candidate.get("scale", [])) != expected
        ):
            return None
        if len(candidate.get("weights", [])) != expected:
            return None
        return candidate

    def reload(self) -> None:
        self.model = self._load(self.path)
        if self.model is None and self.bundled_path is not None:
            bundled = self._load(self.bundled_path)
            if bundled is not None:
                bundled["source"] = "bundled"
            self.model = bundled

    def clear(self) -> None:
        """Forget the locally trained model and remove its persisted file."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        # Fall back to the bundled model rather than leaving the add-on
        # unfiltered after a training reset.
        self.reload()

    def _vector(self, features: dict[str, float]) -> np.ndarray | None:
        if self.model is None:
            return None
        names = self.model["feature_names"]
        vector = np.asarray([float(features.get(name, 0.0)) for name in names], dtype=float)
        mean = np.asarray(self.model["mean"], dtype=float)
        scale = np.asarray(self.model["scale"], dtype=float)
        return (vector - mean) / scale

    def predict(self, features: dict[str, float]) -> float | None:
        vector = self._vector(features)
        if vector is None or self.model is None:
            return None
        score = float(np.dot(vector, np.asarray(self.model["weights"], dtype=float)))
        score += float(self.model["bias"])
        score = max(-30.0, min(30.0, score))
        return 1.0 / (1.0 + math.exp(-score))

    def explain(self, features: dict[str, float]) -> list[tuple[str, float]]:
        """Return the per-category best guess, most likely first.

        The heads are one-vs-rest, so their probabilities are renormalised to
        sum to one. This is a description of what the candidate looks like --
        useful when reviewing a low-confidence detection -- and never feeds the
        accept/reject decision, which stays with :meth:`predict`.
        """
        vector = self._vector(features)
        heads = (self.model or {}).get("class_heads") or {}
        if vector is None or not heads:
            return []
        scores: dict[str, float] = {}
        for label, head in heads.items():
            logit = float(np.dot(vector, np.asarray(head["weights"], dtype=float)))
            logit = max(-30.0, min(30.0, logit + float(head["bias"])))
            scores[label] = 1.0 / (1.0 + math.exp(-logit))
        total = sum(scores.values())
        if total <= 0:
            return []
        return sorted(
            ((label, value / total) for label, value in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )

    @property
    def suggested_threshold(self) -> float | None:
        value = (self.model or {}).get("suggested_threshold")
        return float(value) if isinstance(value, int | float) else None

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

    @staticmethod
    def _prior_log_odds(y: np.ndarray) -> float:
        """Offset that converts a class-balanced fit back to the observed prior.

        ``_fit`` reweights the classes to 50/50, so its probabilities answer
        "given an equal number of weeds and non-weeds, which is this?". Real
        candidate streams are not balanced, and a threshold chosen against the
        balanced answer is far too permissive. Adding log(pos/neg) restores the
        prevalence actually seen in the labels.
        """
        positives = max(1, int(np.count_nonzero(y == 1)))
        negatives = max(1, int(np.count_nonzero(y == 0)))
        return math.log(positives / negatives)

    @staticmethod
    def _scores(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x @ weights + bias, -30, 30)))

    @staticmethod
    def _grouped_split(groups: list[str]) -> tuple[list[int], list[int]]:
        """Hold out whole images, never individual candidates.

        Two candidates cut from the same photo are usually the same weed patch
        under the same light. Splitting between them leaks the answer into the
        validation set and reports a precision the model does not have.
        """
        ordered_groups = sorted(set(groups))
        held_out = {group for index, group in enumerate(ordered_groups) if index % 4 == 0}
        validation = [index for index, group in enumerate(groups) if group in held_out]
        train = [index for index, group in enumerate(groups) if group not in held_out]
        return train, validation

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
        # Only train on the spatial features when every sample carries them.
        # Mixing in zero-filled legacy rows would teach the model that "no
        # recorded distance" means "touching a crop".
        feature_names = list(CORE_FEATURE_NAMES)
        if all(
            all(name in sample["features"] for name in SPATIAL_FEATURE_NAMES) for sample in ordered
        ):
            feature_names.extend(SPATIAL_FEATURE_NAMES)

        raw_x = np.asarray(
            [
                [float(sample["features"].get(name, 0.0)) for name in feature_names]
                for sample in ordered
            ]
        )
        y = np.asarray([1.0 if sample["label"] == POSITIVE_LABEL else 0.0 for sample in ordered])
        mean = raw_x.mean(axis=0)
        scale = raw_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        x = (raw_x - mean) / scale

        # Fall back to the sample id so a sample with no recorded image still
        # forms its own group rather than joining everything else in one bucket.
        groups = [
            str(sample.get("image_id") or f"detection:{sample.get('detection_id', '')}")
            for sample in ordered
        ]
        train_indices, validation_indices = self._grouped_split(groups)
        if (
            not validation_indices
            or not train_indices
            or len({y[index] for index in train_indices}) < 2
            or len({y[index] for index in validation_indices}) < 2
        ):
            # Too few distinct images to hold any out honestly. Report metrics
            # on the training data and say so in the model record.
            train_indices = list(range(len(ordered)))
            validation_indices = list(range(len(ordered)))
            grouped_validation = False
        else:
            grouped_validation = True

        validation_weights, validation_bias = self._fit(x[train_indices], y[train_indices])
        validation_bias += self._prior_log_odds(y[train_indices])
        scores = self._scores(x[validation_indices], validation_weights, validation_bias)
        actual = y[validation_indices] == 1
        curve, suggested = _threshold_curve(scores, actual, TARGET_PRECISION)
        headline = _metrics_at(scores, actual, suggested)

        weights, bias = self._fit(x, y)
        bias += self._prior_log_odds(y)

        # One-vs-rest heads describing *what* the candidate looks like. They
        # share the binary model's scaling so the same feature vector serves
        # both, and only categories with enough examples get a head.
        class_heads: dict[str, dict[str, Any]] = {}
        for label in sorted({sample["label"] for sample in ordered}):
            target = np.asarray(
                [1.0 if sample["label"] == label else 0.0 for sample in ordered], dtype=float
            )
            count = int(np.count_nonzero(target == 1))
            if count < MINIMUM_SAMPLES_PER_HEAD or count == len(ordered):
                continue
            head_weights, head_bias = self._fit(x, target)
            class_heads[label] = {
                "weights": head_weights.tolist(),
                "bias": head_bias + self._prior_log_odds(target),
                "sample_count": count,
            }

        model = {
            "version": 2,
            "created_at": datetime.now(UTC).isoformat(),
            "source": "local",
            "feature_names": feature_names,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "weights": weights.tolist(),
            "bias": bias,
            "class_heads": class_heads,
            "sample_count": len(ordered),
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "prior_corrected": True,
            "grouped_validation": grouped_validation,
            "validation_groups": len({groups[index] for index in validation_indices}),
            "suggested_threshold": suggested,
            "target_precision": TARGET_PRECISION,
            "threshold_curve": curve,
            "metrics": {
                "precision": headline["precision"],
                "precision_lower_bound": headline.get("precision_lower_bound", 0.0),
                "recall": headline["recall"],
                "true_positive": headline["true_positive"],
                "false_positive": headline["false_positive"],
                "false_negative": headline["false_negative"],
                "true_negative": headline["true_negative"],
                "validation_samples": len(validation_indices),
                "at_threshold": suggested,
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
