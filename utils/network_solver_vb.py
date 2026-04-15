"""VB-style multi-pipe network solver.

This is a direct, pragmatic port of the legacy VB logic used in the old
refrigerant pipe sizing tool:

* Build one upstream path per load by following `next_label`.
* Detect broken links and cycles.
* Compute incoming branch counts.
* Enforce: non-header nodes may have at most 2 incoming branches.
* Aggregate upstream duties by summing downstream load duties along each path.

This module is intentionally thermodynamics-agnostic. It only prepares the
network (paths + aggregated duties) for per-segment sizing.

Add-on (VB-style apportionment support):
---------------------------------------
Legacy VB apportioned the *path-level* allowable pressure drop by equivalent
length using:

    MainMaxPD = MainTPD / MPL

where MPL is the worst-case (maximum) total equivalent length among all load
paths. To enable the same style in Python, we provide helpers to compute:

- segment equivalent length (meters)
- path equivalent length (meters)
- worst-case path equivalent length MPL (meters)

These utilities let you apportion a global max penalty across segments in
proportion to equivalent length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


class NetworkError(Exception):
    pass


@dataclass
class Circuit:
    label: str
    next_label: str  # "" means root / end
    is_load: bool
    is_header: bool = False
    duty_kw: float = 0.0
    incoming_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


def _build_index(circuits: List[Circuit]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i, c in enumerate(circuits):
        lab = c.label.strip()
        if not lab:
            raise NetworkError("Blank circuit label")
        if lab in idx:
            raise NetworkError(f"Duplicate label: {lab}")
        idx[lab] = i
    return idx


def enumerate_load_paths(circuits: List[Circuit], max_depth: int = 500) -> List[List[int]]:
    idx = _build_index(circuits)
    paths: List[List[int]] = []

    for i, c in enumerate(circuits):
        if not c.is_load:
            continue

        path = [i]
        seen: Set[int] = {i}
        cur = i
        depth = 0

        while True:
            depth += 1
            if depth > max_depth:
                raise NetworkError(f"Path from load '{c.label}' exceeded max depth (cycle?)")

            nxt = circuits[cur].next_label.strip()
            if nxt == "":
                break
            if nxt not in idx:
                raise NetworkError(f"Broken link: '{circuits[cur].label}' -> '{nxt}' not found")

            cur = idx[nxt]
            if cur in seen:
                raise NetworkError(f"Cycle detected following load '{c.label}' at '{circuits[cur].label}'")
            seen.add(cur)
            path.append(cur)

        paths.append(path)

    if not paths:
        raise NetworkError("No loads found (is_load=True).")
    return paths


def compute_incoming_counts(circuits: List[Circuit]) -> None:
    for c in circuits:
        c.incoming_count = 0

    by_label = {c.label.strip(): c for c in circuits}
    for c in circuits:
        nxt = c.next_label.strip()
        if nxt and nxt in by_label:
            by_label[nxt].incoming_count += 1


def validate_branching(circuits: List[Circuit]) -> None:
    for c in circuits:
        if c.is_load:
            continue
        if (not c.is_header) and c.incoming_count > 2:
            raise NetworkError(
                f"'{c.label}' has {c.incoming_count} incoming branches but is not a header."
            )


def aggregate_node_duties_from_paths(circuits: List[Circuit], paths: List[List[int]]) -> None:
    # reset non-load
    for c in circuits:
        if not c.is_load:
            c.duty_kw = 0.0

    for path in paths:
        load_idx = path[0]
        load_duty = float(circuits[load_idx].duty_kw)
        if load_duty <= 0:
            raise NetworkError(f"Load '{circuits[load_idx].label}' has non-positive duty ({load_duty}).")

        for node_idx in path[1:]:
            if not circuits[node_idx].is_load:
                circuits[node_idx].duty_kw += load_duty


def solve_network(circuits: List[Circuit]) -> Dict[str, Any]:
    paths = enumerate_load_paths(circuits)
    compute_incoming_counts(circuits)
    validate_branching(circuits)
    aggregate_node_duties_from_paths(circuits, paths)
    return {"paths": paths, "label_index": _build_index(circuits)}


# ------------------------------------------------------------
# VB-style "allowable PD per equivalent length" support
# ------------------------------------------------------------

def segment_equivalent_length_m(c: Circuit) -> float:
    """
    Equivalent length for a circuit in meters.

    CHANGE vs earlier version:
    - Loads (is_load=True) are now allowed to have geometry too, so we DO NOT
      return 0.0 for loads. This makes load terminals contribute to MPL and
      penalty apportionment like other pipes.

    Notes:
    - Uses straight length 'L' (meters) and a pipe length factor 'PLF' as a
      multiplier: Leq_straight = L * (1 + PLF).
    - Adds a simple equivalent length allowance for fittings/valves.
      Defaults to 1.0 m each, but can be overridden via:

          c.meta["equiv_m_per_fitting"] = {"SRB": 0.5, "ball": 2.0, ...}
    """
    m = c.meta or {}
    L = float(m.get("L", 0.0) or 0.0)
    PLF = float(m.get("PLF", 0.0) or 0.0)

    leq = max(0.0, L) * (1.0 + max(0.0, PLF))

    default_eq = 1.0
    eq_map = m.get("equiv_m_per_fitting") or {}

    def _eq(name: str) -> float:
        try:
            return float(eq_map.get(name, default_eq))
        except Exception:
            return default_eq

    counts = {
        "SRB": float(m.get("SRB", 0) or 0),
        "LRB": float(m.get("LRB", 0) or 0),
        "_45": float(m.get("_45", 0) or 0),
        "MAC": float(m.get("MAC", 0) or 0),
        "ptrap": float(m.get("ptrap", 0) or 0),
        "ubend": float(m.get("ubend", 0) or 0),
        "ball": float(m.get("ball", 0) or 0),
        "globe": float(m.get("globe", 0) or 0),
    }

    fittings_leq = 0.0
    for k, n in counts.items():
        if n > 0:
            fittings_leq += n * _eq(k)

    return leq + fittings_leq


def path_equivalent_length_m(circuits: List[Circuit], path: List[int]) -> float:
    """
    Sum equivalent length across ALL circuits in a load->root path.

    CHANGE vs earlier version:
    - includes load terminal geometry too.
    """
    total = 0.0
    for node_idx in path:
        total += segment_equivalent_length_m(circuits[node_idx])
    return total


def worst_case_path_equivalent_length_m(
    circuits: List[Circuit], paths: List[List[int]]
) -> Tuple[float, List[float]]:
    lengths = [path_equivalent_length_m(circuits, p) for p in paths]
    mpl = max(lengths) if lengths else 0.0
    return float(mpl), [float(x) for x in lengths]
