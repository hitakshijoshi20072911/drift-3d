"""Telemetry-only flight-complexity classifier.

Recommends keyframe selection mode and a target-frame count from the
flight's own GPS/altitude telemetry, before any video processing -- so the
pipeline adapts its configuration to what kind of flight this is, instead
of one fixed profile being hand-picked for every dataset.

This is a recommendation with its reasoning attached, not a silent
override -- consistent with this project's evidence-gated design
elsewhere (verify_outputs.py, capture_quality.py): it reports what it
measured and why, including the cases where it cannot deliver full
coverage within a runtime budget, rather than hiding the tradeoff.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .telemetry import TelemetrySample

# The frame budget is NOT scaled from telemetry. An earlier version scaled
# it with footprint size and picked 300 frames for DJI_1006, which ran
# well past 30 minutes. Measured runtimes at 120 frames showed why that
# cannot work: DJI_1006 (642 m footprint) took 23.6-24.2 min while the
# larger DJI_1003 (1717 m) took 20.4 min, so cost is driven by scene
# content, which telemetry cannot see. The recommendation therefore stays
# at the budget validated on real runs; footprint only drives an honest
# coverage warning.
VALIDATED_TARGET_FRAMES = 120
# Density reference for the coverage warning only: the HF/Mori-Point clip's
# accepted 253-frame run over a ~507 m footprint diagonal.
REFERENCE_FRAMES = 253
REFERENCE_AREA_DIAGONAL_M = 507.0
VIEWPOINT_COMPLEXITY_THRESHOLD = 0.05
HOVER_SPEED_THRESHOLD_MPS = 0.5
SPEED_WINDOW_S = 1.0


@dataclass(frozen=True)
class FlightComplexity:
    duration_s: float
    altitude_range_m: float
    altitude_std_m: float
    area_diagonal_m: float
    hover_fraction: float
    speed_median_mps: float
    speed_p95_mps: float
    viewpoint_complexity_ratio: float
    recommended_keyframe_mode: str
    recommended_profile: str
    recommended_target_frames: int
    coverage_feasibility_warning: str | None
    reasoning: list[str] = field(default_factory=list)


def classify(telemetry_samples: list[TelemetrySample]) -> FlightComplexity:
    """Classify flight motion complexity and physical scale from telemetry alone."""
    if len(telemetry_samples) < 2:
        raise ValueError("Need at least 2 telemetry samples to classify a flight")

    times = [s.time_s for s in telemetry_samples]
    lats = [s.latitude for s in telemetry_samples]
    lons = [s.longitude for s in telemetry_samples]
    alts = [s.altitude_m for s in telemetry_samples]
    duration_s = times[-1] - times[0]

    lat0, lon0 = lats[0], lons[0]
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    xs = [(lon - lon0) * m_per_deg_lon for lon in lons]
    ys = [(lat - lat0) * m_per_deg_lat for lat in lats]
    area_diagonal_m = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    altitude_range_m = max(alts) - min(alts)
    mean_alt = sum(alts) / len(alts)
    altitude_std_m = math.sqrt(sum((a - mean_alt) ** 2 for a in alts) / len(alts))

    # Resample to >=1 s spacing before differencing. DJI exports log rows at
    # ~60 Hz but GPS position updates far less often, so consecutive rows
    # repeat a position (reads as zero speed) and then jump (reads as an
    # impossible speed). Differencing across >=1 s windows removes that.
    kept = [0]
    for i in range(1, len(times)):
        if times[i] - times[kept[-1]] >= SPEED_WINDOW_S:
            kept.append(i)
    speeds = []
    for a, b in zip(kept, kept[1:]):
        dt = times[b] - times[a]
        dx, dy, dz = xs[b] - xs[a], ys[b] - ys[a], alts[b] - alts[a]
        speeds.append(math.sqrt(dx * dx + dy * dy + dz * dz) / dt)
    speeds_sorted = sorted(speeds)
    speed_median = speeds_sorted[len(speeds_sorted) // 2] if speeds_sorted else 0.0
    speed_p95 = speeds_sorted[int(0.95 * (len(speeds_sorted) - 1))] if speeds_sorted else 0.0
    hover_fraction = (sum(1 for v in speeds if v < HOVER_SPEED_THRESHOLD_MPS) / len(speeds)
                       if speeds else 0.0)

    viewpoint_ratio = altitude_range_m / area_diagonal_m if area_diagonal_m > 1e-6 else 0.0

    reasoning: list[str] = []
    if viewpoint_ratio >= VIEWPOINT_COMPLEXITY_THRESHOLD:
        keyframe_mode = "adaptive"
        reasoning.append(
            f"Altitude range ({altitude_range_m:.1f} m) is {viewpoint_ratio*100:.1f}% "
            f"of the flight's own footprint diagonal ({area_diagonal_m:.0f} m) -- a "
            f"fixed-budget selector can leave a gap too wide to track through a fast "
            f"viewpoint-scale change. Recommending the visually-verified adaptive "
            f"selector, which inserts extra frames wherever a link is weak."
        )
    else:
        keyframe_mode = "geometry"
        reasoning.append(
            f"Altitude range ({altitude_range_m:.1f} m) is only {viewpoint_ratio*100:.1f}% "
            f"of the footprint diagonal -- a roughly planar flight. The fixed-budget "
            f"geometry selector is proven and faster for this profile."
        )

    target_frames = VALIDATED_TARGET_FRAMES
    density_estimate = REFERENCE_FRAMES * area_diagonal_m / REFERENCE_AREA_DIAGONAL_M
    reasoning.append(
        f"Frame budget stays at the validated {VALIDATED_TARGET_FRAMES}: runtime depends on "
        f"scene content that telemetry cannot see, so it is not scaled from footprint size."
    )

    coverage_warning = None
    if density_estimate > VALIDATED_TARGET_FRAMES:
        coverage_warning = (
            f"Matching the reference density over a {area_diagonal_m:.0f} m footprint would "
            f"take about {density_estimate:.0f} frames; {VALIDATED_TARGET_FRAMES} will give "
            f"partial dense-surface coverage. More frames can raise coverage, but runtime "
            f"grows by an amount that varies by scene -- set TARGET_FRAMES explicitly only "
            f"if coverage matters more than runtime for this flight."
        )
        reasoning.append(f"NOTE: {coverage_warning}")

    recommended_profile = "verified-fast-adaptive" if keyframe_mode == "adaptive" else "verified-fast"

    return FlightComplexity(
        duration_s=duration_s,
        altitude_range_m=altitude_range_m,
        altitude_std_m=altitude_std_m,
        area_diagonal_m=area_diagonal_m,
        hover_fraction=hover_fraction,
        speed_median_mps=speed_median,
        speed_p95_mps=speed_p95,
        viewpoint_complexity_ratio=viewpoint_ratio,
        recommended_keyframe_mode=keyframe_mode,
        recommended_profile=recommended_profile,
        recommended_target_frames=target_frames,
        coverage_feasibility_warning=coverage_warning,
        reasoning=reasoning,
    )
