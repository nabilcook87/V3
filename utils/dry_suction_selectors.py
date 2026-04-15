"""Selection helpers for Dry Suction sizing.

These functions implement the same selection rules used in the Dry Suction UI:

* Horizontal: choose the smallest nominal mm where ΔT <= max_penalty.
* Single riser: choose the smallest nominal mm where MOR <= required_oil_duty_pct
  AND ΔT <= max_penalty.
* Double riser: follow the exact candidate-search logic from the standalone
  Dry Suction page's "Double Riser" button block.

They intentionally mirror the loop patterns in the UI code:
  - test each size with the *strongest gauge* available for that size for the
    single-pipe selectors;
  - for double risers, rely on callbacks supplied by the caller that already
    encapsulate the exact standalone-page cached calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import math


@dataclass(frozen=True)
class DoubleRiserSelection:
    """Result of the exact standalone double-riser selector."""

    best_small: str
    best_large: str
    best_dr: Any
    best_MOR_full: float
    best_MOR_large: float


def _strongest_gauge(material_df, size_inch: str) -> Optional[str]:
    rows = material_df[
        material_df["Nominal Size (inch)"].astype(str).str.strip() == str(size_inch).strip()
    ]
    if rows.empty or "Gauge" not in rows.columns or rows["Gauge"].isna().all():
        return None
    return max(rows["Gauge"].dropna().unique())


def select_horizontal_low_dt(
    *,
    pipe_sizes: List[str],
    mm_map: Dict[str, float],
    material_df,
    get_pipe_results: Callable[[str, Optional[str]], Tuple[float, float]],
    max_penalty_K: float,
) -> Tuple[Optional[str], float]:
    """Return (best_size_inch, best_dt). best_size_inch is None if none satisfy."""

    results = []
    best_dt = float("inf")

    for ps in pipe_sizes:
        test_gauge = _strongest_gauge(material_df, ps)
        out = get_pipe_results(ps, test_gauge)
        mor_i, dt_i = out[0], out[1]
        if math.isfinite(dt_i):
            best_dt = min(best_dt, dt_i)
            results.append({"size": ps, "dt": dt_i})

    valid = [r for r in results if r["dt"] <= max_penalty_K]
    if not valid:
        return None, best_dt

    best = min(valid, key=lambda x: mm_map.get(x["size"], float("inf")))
    return best["size"], float(best["dt"])


def select_single_riser(
    *,
    pipe_sizes: List[str],
    mm_map: Dict[str, float],
    material_df,
    get_pipe_results: Callable[[str, Optional[str]], Tuple[float, float]],
    required_oil_duty_pct: float,
    max_penalty_K: float,
) -> Tuple[Optional[str], float, float]:
    """Return (best_size_inch, MOR, dt). best_size_inch is None if none satisfy."""

    results = []
    for ps in pipe_sizes:
        test_gauge = _strongest_gauge(material_df, ps)
        out = get_pipe_results(ps, test_gauge)
        mor_i, dt_i = out[0], out[1]
        if math.isfinite(mor_i) and math.isfinite(dt_i):
            results.append({"size": ps, "MOR": mor_i, "dt": dt_i})

    valid = [
        r for r in results
        if (r["MOR"] <= required_oil_duty_pct) and (r["dt"] <= max_penalty_K)
    ]
    if not valid:
        return None, float("nan"), float("nan")

    best = min(valid, key=lambda x: mm_map.get(x["size"], float("inf")))
    return best["size"], float(best["MOR"]), float(best["dt"])


def select_double_riser_exact(
    *,
    pipe_sizes: List[str],
    mm_map: Dict[str, float],
    required_oil_duty_pct: float,
    max_penalty_K: float,
    MOR_full_cached2: Callable[[str], float],
    eval_pair_cached: Callable[[str, str], Tuple[Any, Optional[float], Optional[float]]],
) -> DoubleRiserSelection:
    """Return the exact standalone Dry Suction double-riser selection."""

    sizes_asc = sorted(pipe_sizes, key=lambda s: mm_map[s])
    max_small_mor = min(required_oil_duty_pct, 50.0)

    small_candidates = []
    for small in sizes_asc:
        MOR_s = MOR_full_cached2(small)
        if not math.isfinite(MOR_s) or MOR_s > max_small_mor:
            continue
        small_candidates.append(small)

    if not small_candidates:
        raise ValueError("❌ No pipe size satisfies full-flow oil return duty.")

    small = max(small_candidates, key=lambda s: mm_map[s])
    small_mm = mm_map[small]

    last_over = None
    first_under_or_equal = None

    for large in sizes_asc:
        large_mm = mm_map[large]
        if large_mm < small_mm:
            continue

        dr, MOR_full, MOR_large = eval_pair_cached(small, large)

        if MOR_full is None or MOR_large is None:
            continue

        if MOR_large > 100.0:
            continue

        candidate = {
            "small": small,
            "large": large,
            "dr": dr,
            "MOR_full": MOR_full,
            "MOR_large": MOR_large,
        }

        if dr.DT_K > max_penalty_K:
            last_over = candidate
        elif first_under_or_equal is None:
            first_under_or_equal = candidate
            break

    best = last_over if last_over is not None else first_under_or_equal

    if best is None:
        raise ValueError("❌ No valid large riser meets ΔT and MOR limits.")

    return DoubleRiserSelection(
        best_small=best["small"],
        best_large=best["large"],
        best_dr=best["dr"],
        best_MOR_full=float(best["MOR_full"]),
        best_MOR_large=float(best["MOR_large"]),
    )
