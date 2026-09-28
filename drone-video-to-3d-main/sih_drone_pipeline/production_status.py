from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Iterator


class TerminalState(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL_VALID = "PARTIAL_VALID"
    DEADLINE_LIMITED = "DEADLINE_LIMITED"
    INSUFFICIENT_CAPTURE = "INSUFFICIENT_CAPTURE"
    TELEMETRY_INVALID = "TELEMETRY_INVALID"
    FAILED = "FAILED"


def derive_terminal_state(
    *,
    telemetry_valid: bool,
    capture_sufficient: bool,
    structural_artifacts_valid: bool,
    substantive_geometry_valid: bool,
    production_gates_pass: bool,
    deadline_met: bool,
    stopped_for_deadline: bool = False,
) -> tuple[TerminalState, list[str]]:
    """Return one honest terminal state plus machine-readable reason codes."""
    reasons: list[str] = []
    if not telemetry_valid:
        return TerminalState.TELEMETRY_INVALID, ["telemetry_validation_failed"]
    if not capture_sufficient:
        return TerminalState.INSUFFICIENT_CAPTURE, ["capture_quality_or_overlap_failed"]
    if not structural_artifacts_valid:
        return TerminalState.FAILED, ["required_artifacts_or_stages_failed"]
    if not substantive_geometry_valid:
        return TerminalState.FAILED, ["dense_geometry_is_degenerate"]
    if stopped_for_deadline:
        reasons.append("processing_stopped_at_deadline")
        if not production_gates_pass:
            reasons.append("production_gates_incomplete")
        return TerminalState.DEADLINE_LIMITED, reasons
    if not deadline_met:
        reasons.append("runtime_target_exceeded")
    if not production_gates_pass:
        reasons.append("one_or_more_production_gates_failed")
    if reasons:
        return TerminalState.PARTIAL_VALID, reasons
    return TerminalState.COMPLETE, []


RUNTIME_CATEGORIES = (
    "normal_pipeline",
    "rescue",
    "experiment",
    "merge",
    "export",
)


@dataclass
class RuntimeLedger:
    """Keep normal, optional, and experimental clocks from contaminating SLAs."""

    entries: list[dict] = field(default_factory=list)

    def add_seconds(self, category: str, seconds: float, label: str) -> None:
        if category not in RUNTIME_CATEGORIES:
            raise ValueError(f"Unknown runtime category: {category}")
        if isinstance(seconds, bool) or not math.isfinite(float(seconds)) or seconds < 0:
            raise ValueError("Runtime seconds must be a finite non-negative number")
        if not label or not label.strip():
            raise ValueError("Runtime label must not be empty")
        self.entries.append({
            "category": category,
            "label": label.strip(),
            "seconds": float(seconds),
        })

    @contextmanager
    def measure(self, category: str, label: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_seconds(category, time.perf_counter() - started, label)

    def summary(self) -> dict:
        totals = {
            category: sum(
                entry["seconds"]
                for entry in self.entries
                if entry["category"] == category
            )
            for category in RUNTIME_CATEGORIES
        }
        return {
            "categories_seconds": totals,
            "normal_pipeline_minutes": totals["normal_pipeline"] / 60.0,
            "optional_rescue_minutes": totals["rescue"] / 60.0,
            "experiment_minutes": totals["experiment"] / 60.0,
            "merge_minutes": totals["merge"] / 60.0,
            "export_minutes": totals["export"] / 60.0,
            "release_workflow_minutes": (
                totals["normal_pipeline"] + totals["merge"] + totals["export"]
            ) / 60.0,
            "session_accounted_minutes": sum(totals.values()) / 60.0,
            "entries": list(self.entries),
        }

