from __future__ import annotations

from .models import Decision, Measurement, OperatingMode
from .settings import Settings


def decide(
    measurement: Measurement,
    mode: OperatingMode,
    settings: Settings,
    days_since_previous: float = 1.0,
) -> Measurement:
    current = measurement.current_radius_mm
    proposed = measurement.recommended_protection_radius_mm
    if measurement.ambiguous:
        return measurement.model_copy(
            update={
                "decision": Decision.UNCERTAIN,
                "reason": "overlapping canopy ownership is ambiguous",
            }
        )
    if proposed <= current:
        return measurement.model_copy(
            update={"decision": Decision.RETAIN, "reason": "automatic shrinking is disabled"}
        )
    if mode == OperatingMode.OBSERVE:
        return measurement.model_copy(
            update={
                "decision": Decision.OBSERVED,
                "reason": "observe mode does not write FarmBot data",
            }
        )
    percent = 100 * (proposed - current) / max(current, 1)
    if percent > settings.maximum_single_update_percent:
        return measurement.model_copy(
            update={
                "decision": Decision.UNCERTAIN,
                "reason": "increase exceeds maximum single-update percentage",
            }
        )
    if proposed - current > settings.maximum_daily_radius_growth_mm * max(days_since_previous, 1):
        return measurement.model_copy(
            update={
                "decision": Decision.UNCERTAIN,
                "reason": "increase exceeds maximum daily growth",
            }
        )
    if (
        mode == OperatingMode.AUTO_RADIUS
        and measurement.confidence < settings.minimum_auto_confidence
    ):
        return measurement.model_copy(
            update={
                "decision": Decision.UNCERTAIN,
                "reason": "confidence is below automatic threshold",
            }
        )
    return measurement.model_copy(
        update={
            "decision": Decision.APPLIED
            if mode == OperatingMode.AUTO_RADIUS
            else Decision.RECOMMENDED,
            "reason": "safe radius increase",
        }
    )
