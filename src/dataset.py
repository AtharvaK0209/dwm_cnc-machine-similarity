"""Dataset loader, attribute preprocessor, and 3-tier condition mapper for CNC data.

This module loads the AI4I 2020 dataset and provides a 3-tier condition mapping
based on physical operating regimes and breakdown flags.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default attribute schema definitions for Gower distance
NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NOMINAL_COLUMNS = ["Type"]

# By default, Type is treated as nominal. Ordinal is empty by default
# unless a separate, defensible ordered attribute is provided.
ORDINAL_COLUMNS: List[str] = []
ORDINAL_ORDERS: Dict[str, List[str]] = {}

# Operational binary attribute (purely operating regime, NOT failure indicators)
# Operational_High_Load: 1 if Torque >= 40.0 Nm (the dataset median/mean ~40 Nm), else 0.
BINARY_COLUMNS = ["Operational_High_Load"]

# Target classes
CONDITION_CLASSES = ["Normal", "Warning", "Critical"]

# Excluded columns to strictly prevent target leakage & spurious correlation:
EXCLUDED_FROM_FEATURES = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",  # Random Failure flag: excluded to prevent failure-mode leakage
]


def compute_mechanical_power(torque_nm: float, speed_rpm: float) -> float:
    """Compute mechanical power in Watts: P = Torque * angular_velocity."""
    omega = speed_rpm * (2.0 * math.pi / 60.0)
    return torque_nm * omega


def derive_condition(
    machine_failure: int,
    air_temp: float,
    process_temp: float,
    speed_rpm: float,
    torque_nm: float,
    tool_wear: float,
) -> str:
    """Derive 3-tier operational health condition (Normal, Warning, Critical).

    Rules & Overlap Precedence:
    1. Critical (Precedence 1 - Highest):
       machine_failure == 1
       Rationale: An actual simulated breakdown has occurred. When a machine has failed,
       its state is Critical regardless of transient sensor measurements.

    2. Warning (Precedence 2):
       Evaluated ONLY when machine_failure == 0:
       - Tool wear >= 180.0 min (approaching simulated tool wear failure regime ~200-240 min)
       - Temp diff (Process - Air) <= 9.0 K (approaching heat dissipation failure regime < 8.6 K)
       - Mechanical power <= 3800.0 W or >= 8500.0 W (approaching power failure regime < 3500 W or > 9000 W)
       Rationale: Represents incipient fault / high-stress operational zone before breakdown.

    3. Normal (Precedence 3 - Baseline):
       machine_failure == 0 and none of the warning stress criteria are met.
       Rationale: Machine operating within safe baseline envelope.

    Methodological Limitations:
    - These 3-tier labels are synthetic heuristic rules created to fulfill the academic
      assignment specification. They are NOT measured ground truth from physical factory logs.
    - Thresholds (180 min, 9.0 K, 3800/8500 W) are hard boundaries.
    - Evaluation against these labels measures classifier agreement with these rules,
      not validated real-world failure foresight.
    """
    if machine_failure == 1:
        return "Critical"

    temp_diff = process_temp - air_temp
    power = compute_mechanical_power(torque_nm, speed_rpm)

    if (
        tool_wear >= 180.0
        or temp_diff <= 9.0
        or power <= 3800.0
        or power >= 8500.0
    ):
        return "Warning"

    return "Normal"


class CNCDataset:
    """Container for parsed CNC machine records, attribute ranges, and labels."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        numeric_ranges: Optional[Dict[str, Tuple[float, float, float]]] = None,
        numeric_cols: Optional[List[str]] = None,
        nominal_cols: Optional[List[str]] = None,
        ordinal_cols: Optional[List[str]] = None,
        binary_cols: Optional[List[str]] = None,
    ):
        self.records = records
        self.numeric_cols = numeric_cols if numeric_cols is not None else list(NUMERIC_COLUMNS)
        self.nominal_cols = nominal_cols if nominal_cols is not None else list(NOMINAL_COLUMNS)
        self.ordinal_cols = ordinal_cols if ordinal_cols is not None else list(ORDINAL_COLUMNS)
        self.binary_cols = binary_cols if binary_cols is not None else list(BINARY_COLUMNS)

        if numeric_ranges is not None:
            self.ranges = numeric_ranges
        else:
            self.ranges = self._calculate_numeric_ranges()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]

    def _calculate_numeric_ranges(self) -> Dict[str, Tuple[float, float, float]]:
        """Compute (min, max, range) for each numeric attribute for Gower normalization."""
        ranges: Dict[str, Tuple[float, float, float]] = {}
        if not self.records:
            return ranges

        for col in self.numeric_cols:
            vals = [
                float(r[col])
                for r in self.records
                if r.get(col) is not None and str(r.get(col)).strip() != ""
            ]
            if vals:
                min_v = min(vals)
                max_v = max(vals)
                span = max_v - min_v
                ranges[col] = (min_v, max_v, span)
            else:
                ranges[col] = (0.0, 0.0, 0.0)
        return ranges

    def get_features_and_labels(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        features: List[Dict[str, Any]] = []
        labels: List[str] = []
        active_cols = self.numeric_cols + self.nominal_cols + self.ordinal_cols + self.binary_cols
        for r in self.records:
            feat = {k: r.get(k) for k in active_cols}
            features.append(feat)
            labels.append(r["condition"])
        return features, labels


def load_dataset(csv_path: str | Path) -> CNCDataset:
    """Load and parse AI4I 2020 dataset from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    records: List[Dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            air_temp = float(row["Air temperature [K]"])
            proc_temp = float(row["Process temperature [K]"])
            speed_rpm = float(row["Rotational speed [rpm]"])
            torque_nm = float(row["Torque [Nm]"])
            tool_wear = float(row["Tool wear [min]"])
            machine_failure = int(row["Machine failure"])
            rnf = int(row["RNF"])
            machine_type = row["Type"].strip()

            # Operational binary attribute (Torque >= 40.0 Nm -> high load operation)
            # This is an operational regime flag, NOT a failure indicator
            operational_high_load = 1 if torque_nm >= 40.0 else 0

            condition = derive_condition(
                machine_failure=machine_failure,
                air_temp=air_temp,
                process_temp=proc_temp,
                speed_rpm=speed_rpm,
                torque_nm=torque_nm,
                tool_wear=tool_wear,
            )

            record = {
                "udi": int(row.get("UDI") or row.get("\ufeffUDI") or 0),
                "product_id": row["Product ID"].strip(),
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": proc_temp,
                "Rotational speed [rpm]": speed_rpm,
                "Torque [Nm]": torque_nm,
                "Tool wear [min]": tool_wear,
                "Operational_High_Load": operational_high_load,
                # Retained purely for audit verification and excluded from features:
                "RNF": rnf,
                "machine_failure": machine_failure,
                "condition": condition,
            }
            records.append(record)

    return CNCDataset(records)


def train_test_split(
    dataset: CNCDataset,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[CNCDataset, CNCDataset]:
    """Split dataset into training and testing subsets.

    Numeric normalization ranges are calculated STRICTLY from train_records,
    and then assigned to both train and test CNCDataset instances to prevent
    information leakage from the test split.
    """
    rng = random.Random(seed)

    if not stratify:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        split_point = int(len(dataset) * (1.0 - test_ratio))
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
    else:
        # Stratify by condition
        class_buckets: Dict[str, List[int]] = {c: [] for c in CONDITION_CLASSES}
        for idx, rec in enumerate(dataset.records):
            class_buckets[rec["condition"]].append(idx)

        train_idx = []
        test_idx = []
        for c, idxs in class_buckets.items():
            shuffled = list(idxs)
            rng.shuffle(shuffled)
            split_point = int(len(shuffled) * (1.0 - test_ratio))
            train_idx.extend(shuffled[:split_point])
            test_idx.extend(shuffled[split_point:])

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    train_records = [dataset.records[i] for i in train_idx]
    test_records = [dataset.records[i] for i in test_idx]

    # Fit numeric ranges ONLY on training records
    train_dataset = CNCDataset(
        train_records,
        numeric_cols=dataset.numeric_cols,
        nominal_cols=dataset.nominal_cols,
        ordinal_cols=dataset.ordinal_cols,
        binary_cols=dataset.binary_cols,
    )
    # Test dataset uses training ranges (no leakage from test set)
    test_dataset = CNCDataset(
        test_records,
        numeric_ranges=train_dataset.ranges,
        numeric_cols=dataset.numeric_cols,
        nominal_cols=dataset.nominal_cols,
        ordinal_cols=dataset.ordinal_cols,
        binary_cols=dataset.binary_cols,
    )

    return train_dataset, test_dataset
