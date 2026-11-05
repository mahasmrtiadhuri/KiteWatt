"""Deployment gate: compression is only valuable when it repays its cost."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class GateDecision:
    deploy: bool
    reason: str
    break_even_requests: int | None
    latency_saving_pct: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def decide(
    *, reference_ms: float, compiled_ms: float, compile_seconds: float,
    measured_accuracy: float, expected_requests: int = 100_000,
    min_saving_fraction: float = 0.10,
) -> GateDecision:
    """Choose based on accuracy, measured latency, and amortized compilation.

    This is a wall-clock heuristic, *not* an energy or dollar measurement. It
    excludes the original model's training costs and any future maintenance.
    """
    vals = (reference_ms, compiled_ms, compile_seconds, measured_accuracy, min_saving_fraction)
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
        raise ValueError("Measurements must be finite numbers")
    if reference_ms <= 0 or compiled_ms <= 0 or compile_seconds < 0:
        raise ValueError("Invalid cost measurement")
    if not 0 <= measured_accuracy <= 1 or not 0 < min_saving_fraction < 1:
        raise ValueError("Invalid reliability threshold")
    if type(expected_requests) is not int or not 1 <= expected_requests <= 10_000_000:
        raise ValueError("Expected request count is outside supported range")
    savings = (reference_ms - compiled_ms) / reference_ms
    if measured_accuracy < 1.0:
        return GateDecision(False, "accuracy_regression", None, round(savings * 100, 2))
    if savings < min_saving_fraction:
        return GateDecision(False, "not_faster_enough", None, round(savings * 100, 2))
    break_even = math.ceil((compile_seconds * 1000) / (reference_ms - compiled_ms))
    if break_even > expected_requests:
        return GateDecision(False, "not_amortized", break_even, round(savings * 100, 2))
    return GateDecision(True, "estimated_benefit", break_even, round(savings * 100, 2))
