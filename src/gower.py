"""Gower distance and similarity metric implementation for mixed-type attributes.

References:
- Gower, J. C. (1971). "A General Coefficient of Similarity and Some of Its Properties." Biometrics, 27(4), 857-871.
- Han, J., Kamber, M., & Pei, J. (2011). "Data Mining: Concepts and Techniques." 3rd ed., Morgan Kaufmann, Ch 2.4.4.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .dataset import (
    BINARY_COLUMNS,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    ORDINAL_COLUMNS,
    ORDINAL_ORDERS,
)


def is_missing(val: Any) -> bool:
    """Check if a value is missing (None, empty string, or NaN)."""
    if val is None:
        return True
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "null", "none"):
        return True
    if isinstance(val, (int, float)):
        return math.isnan(val)
    return False


def compute_numeric_component(
    val_a: Any,
    val_b: Any,
    attr_range: float,
) -> Tuple[float, float]:
    """Compute normalized difference and validity indicator (d_f, delta_f).

    d_ij = |x_i - x_j| / R_f
    If either value is missing, delta_f = 0.0.
    If range <= 0 (zero-range feature), d_f = 0.0 and delta_f = 1.0 (or 0.0 if uninformative).
    """
    if is_missing(val_a) or is_missing(val_b):
        return 0.0, 0.0

    try:
        a = float(val_a)
        b = float(val_b)
    except (ValueError, TypeError):
        return 0.0, 0.0

    if math.isnan(a) or math.isnan(b):
        return 0.0, 0.0

    if attr_range <= 0.0:
        # Zero-range feature: values are constant across the training population
        return 0.0, 1.0

    diff = abs(a - b) / attr_range
    diff = min(1.0, max(0.0, diff))
    return diff, 1.0


def compute_nominal_component(val_a: Any, val_b: Any) -> Tuple[float, float]:
    """Compute distance for nominal attribute: 0 if equal, 1 if unequal.

    Returns (d_f, delta_f). If either is missing, delta_f = 0.0.
    """
    if is_missing(val_a) or is_missing(val_b):
        return 0.0, 0.0

    match = str(val_a).strip() == str(val_b).strip()
    return (0.0 if match else 1.0), 1.0


def compute_ordinal_component(
    val_a: Any,
    val_b: Any,
    ordered_states: List[str],
) -> Tuple[float, float]:
    """Compute distance for ordinal attribute using rank normalization.

    r in {1, ..., M} -> z = (r - 1) / (M - 1)
    d_ij = |z_i - z_j|
    Returns (d_f, delta_f). If missing or state unknown, delta_f = 0.0.
    """
    if is_missing(val_a) or is_missing(val_b):
        return 0.0, 0.0

    str_a = str(val_a).strip()
    str_b = str(val_b).strip()

    if str_a not in ordered_states or str_b not in ordered_states:
        return 0.0, 0.0

    m = len(ordered_states)
    if m <= 1:
        return 0.0, 1.0

    rank_map = {state: i + 1 for i, state in enumerate(ordered_states)}
    rank_a = rank_map[str_a]
    rank_b = rank_map[str_b]

    z_a = (rank_a - 1) / (m - 1)
    z_b = (rank_b - 1) / (m - 1)

    diff = abs(z_a - z_b)
    diff = min(1.0, max(0.0, diff))
    return diff, 1.0


def compute_binary_component(
    val_a: Any,
    val_b: Any,
    asymmetric: bool = False,
) -> Tuple[float, float]:
    """Compute binary distance and validity indicator (d_ij, delta_ij).

    - Symmetric binary: delta=1; d=0 if equal, 1 if unequal.
    - Asymmetric binary: if both are 0, delta=0 (negative match ignored).
    - Missing values: delta=0.
    """
    if is_missing(val_a) or is_missing(val_b):
        return 0.0, 0.0

    try:
        b_a = int(float(val_a))
        b_b = int(float(val_b))
    except (ValueError, TypeError):
        return 0.0, 0.0

    if b_a not in (0, 1) or b_b not in (0, 1):
        return 0.0, 0.0

    if asymmetric and b_a == 0 and b_b == 0:
        return 0.0, 0.0  # Asymmetric negative match is non-informative

    dist = 0.0 if b_a == b_b else 1.0
    return dist, 1.0


def gower_distance_single(
    record_a: Dict[str, Any],
    record_b: Dict[str, Any],
    numeric_ranges: Dict[str, Tuple[float, float, float]],
    weights: Optional[Dict[str, float]] = None,
    numeric_cols: Optional[List[str]] = None,
    nominal_cols: Optional[List[str]] = None,
    ordinal_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
    ordinal_orders: Optional[Dict[str, List[str]]] = None,
    asymmetric_binary: bool = False,
    return_breakdown: bool = False,
) -> float | Tuple[float, Dict[str, float]]:
    """Compute Gower distance between two records over mixed attributes.

    Returns:
        float in [0, 1] representing distance (similarity = 1 - distance).
        Optionally returns per-attribute component distances dictionary.
    """
    num_cols = numeric_cols if numeric_cols is not None else NUMERIC_COLUMNS
    nom_cols = nominal_cols if nominal_cols is not None else NOMINAL_COLUMNS
    ord_cols = ordinal_cols if ordinal_cols is not None else ORDINAL_COLUMNS
    bin_cols = binary_cols if binary_cols is not None else BINARY_COLUMNS
    ord_orders = ordinal_orders if ordinal_orders is not None else ORDINAL_ORDERS

    total_weighted_dist = 0.0
    total_weight_delta = 0.0
    breakdown: Dict[str, float] = {}

    # 1. Numeric attributes
    for col in num_cols:
        w = weights.get(col, 1.0) if weights else 1.0
        val_a = record_a.get(col)
        val_b = record_b.get(col)
        _, _, span = numeric_ranges.get(col, (0.0, 0.0, 1.0))
        d_f, delta_f = compute_numeric_component(val_a, val_b, span)

        total_weighted_dist += w * delta_f * d_f
        total_weight_delta += w * delta_f
        if return_breakdown:
            breakdown[col] = d_f

    # 2. Nominal attributes
    for col in nom_cols:
        w = weights.get(col, 1.0) if weights else 1.0
        val_a = record_a.get(col)
        val_b = record_b.get(col)
        d_f, delta_f = compute_nominal_component(val_a, val_b)

        total_weighted_dist += w * delta_f * d_f
        total_weight_delta += w * delta_f
        if return_breakdown:
            breakdown[col] = d_f

    # 3. Ordinal attributes
    for col in ord_cols:
        w = weights.get(col, 1.0) if weights else 1.0
        states = ord_orders.get(col, [])
        val_a = record_a.get(col)
        val_b = record_b.get(col)
        d_f, delta_f = compute_ordinal_component(val_a, val_b, states)

        total_weighted_dist += w * delta_f * d_f
        total_weight_delta += w * delta_f
        if return_breakdown:
            breakdown[col] = d_f

    # 4. Binary attributes
    for col in bin_cols:
        w = weights.get(col, 1.0) if weights else 1.0
        val_a = record_a.get(col)
        val_b = record_b.get(col)
        d_f, delta_f = compute_binary_component(val_a, val_b, asymmetric=asymmetric_binary)

        total_weighted_dist += w * delta_f * d_f
        total_weight_delta += w * delta_f
        if return_breakdown:
            breakdown[col] = d_f

    dist = total_weighted_dist / total_weight_delta if total_weight_delta > 0.0 else 0.0
    dist = min(1.0, max(0.0, dist))

    if return_breakdown:
        return dist, breakdown
    return dist


def gower_similarity(distance: float) -> float:
    """Convert Gower distance d in [0, 1] to similarity S = 1 - d."""
    return max(0.0, min(1.0, 1.0 - distance))
