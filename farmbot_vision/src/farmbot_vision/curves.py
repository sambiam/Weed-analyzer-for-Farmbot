from __future__ import annotations

from collections import defaultdict

import numpy as np


def pava(values: list[float], weights: list[float] | None = None) -> list[float]:
    if not values:
        return []
    blocks = [
        [float(value), float((weights or [1] * len(values))[i]), 1]
        for i, value in enumerate(values)
    ]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][0] <= blocks[index + 1][0]:
            index += 1
            continue
        total_weight = blocks[index][1] + blocks[index + 1][1]
        mean = (
            blocks[index][0] * blocks[index][1] + blocks[index + 1][0] * blocks[index + 1][1]
        ) / total_weight
        blocks[index : index + 2] = [[mean, total_weight, blocks[index][2] + blocks[index + 1][2]]]
        index = max(0, index - 1)
    return [block[0] for block in blocks for _ in range(int(block[2]))]


def fit_monotonic_curve(
    observations: list[tuple[int, float]],
    *,
    safety_margin_mm: float = 0,
    quantile: float = 0.9,
    max_points: int = 10,
    bin_days: int = 3,
) -> dict[str, float]:
    if not observations:
        return {}
    bins: dict[int, list[float]] = defaultdict(list)
    for age, radius in observations:
        bins[max(0, age) // bin_days * bin_days].append(radius)
    ages = sorted(bins)
    upper = [float(np.quantile(bins[age], quantile)) + safety_margin_mm for age in ages]
    fitted = pava(upper, [len(bins[age]) for age in ages])
    if len(ages) > max_points:
        selected = sorted(
            set(np.linspace(0, len(ages) - 1, max_points).round().astype(int).tolist())
        )
        ages = [ages[i] for i in selected]
        fitted = [fitted[i] for i in selected]
    return {str(age): round(2 * radius, 1) for age, radius in zip(ages, fitted, strict=True)}
