from __future__ import annotations

from .models import Decision, Measurement, OperatingMode
from .settings import Settings


def decide(
    measurement: Measurement,
    mode: OperatingMode,
    settings: Settings,
    days_since_previous: float = 1.0,
    previously_observed_canopy: bool = False,
) -> Measurement:
    current = measurement.current_radius_mm
    proposed = measurement.recommended_protection_radius_mm
    if not measurement.calibrated:
        return measurement.model_copy(
            update={
                "decision": Decision.OBSERVED,
                "reason": "uncalibrated: no millimetre measurement, no write",
            }
        )
    if measurement.vegetation_absent:
        if not settings.removal_detection_enabled:
            return measurement.model_copy(
                update={"decision": Decision.OBSERVED, "reason": "removal detection is disabled"}
            )
        if not previously_observed_canopy:
            return measurement.model_copy(
                update={
                    "decision": Decision.OBSERVED,
                    "reason": "no earlier canopy observation confirms the plant was present",
                }
            )
        if measurement.absent_observations < settings.removal_min_consecutive_absent:
            return measurement.model_copy(
                update={
                    "decision": Decision.OBSERVED,
                    "reason": (
                        f"absence needs {settings.removal_min_consecutive_absent} consecutive "
                        "observations"
                    ),
                }
            )
        if mode == OperatingMode.OBSERVE:
            return measurement.model_copy(
                update={"decision": Decision.OBSERVED, "reason": "observe mode does not remove plants"}
            )
        automatic = mode == OperatingMode.AUTO_RADIUS and settings.removal_auto_apply
        return measurement.model_copy(
            update={
                "decision": Decision.REMOVED if automatic else Decision.REMOVAL_RECOMMENDED,
                "reason": (
                    "consecutive observations confirm the plant canopy is absent; "
                    + ("automatic archival enabled" if automatic else "human approval required")
                ),
            }
        )
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
