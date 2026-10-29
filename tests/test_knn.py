"""Unit tests for k-NN classifier, neighbor exclusion, and condition prediction."""

import unittest
from src.dataset import CNCDataset, derive_condition, make_query_record, train_test_split, EXCLUDED_FROM_FEATURES
from src.knn import GowerKNNClassifier


class TestKNNClassifier(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset of 4 known machines
        self.mock_records = [
            {
                "udi": 1,
                "product_id": "L1",
                "Type": "L",
                "Air temperature [K]": 298.0,
                "Process temperature [K]": 308.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 20.0,
                "Operational_High_Load": 1,
                "machine_failure": 0,
                "condition": "Normal",
            },
            {
                "udi": 2,
                "product_id": "M1",
                "Type": "M",
                "Air temperature [K]": 298.2,
                "Process temperature [K]": 308.2,
                "Rotational speed [rpm]": 1490.0,
                "Torque [Nm]": 41.0,
                "Tool wear [min]": 25.0,
                "Operational_High_Load": 1,
                "machine_failure": 0,
                "condition": "Normal",
            },
            {
                "udi": 3,
                "product_id": "L2",
                "Type": "L",
                "Air temperature [K]": 302.0,
                "Process temperature [K]": 311.0,
                "Rotational speed [rpm]": 1350.0,
                "Torque [Nm]": 55.0,
                "Tool wear [min]": 210.0,
                "Operational_High_Load": 1,
                "machine_failure": 0,
                "condition": "Warning",
            },
            {
                "udi": 4,
                "product_id": "H1",
                "Type": "H",
                "Air temperature [K]": 304.0,
                "Process temperature [K]": 312.0,
                "Rotational speed [rpm]": 1250.0,
                "Torque [Nm]": 65.0,
                "Tool wear [min]": 230.0,
                "Operational_High_Load": 1,
                "machine_failure": 1,
                "condition": "Critical",
            },
        ]
        self.dataset = CNCDataset(self.mock_records)
        self.clf = GowerKNNClassifier(k=3, weighting="distance").fit(self.dataset)

    def test_nearest_neighbor_ordering(self):
        query = {
            "udi": 999,
            "product_id": "NEW",
            "Type": "L",
            "Air temperature [K]": 298.05,
            "Process temperature [K]": 308.05,
            "Rotational speed [rpm]": 1502.0,
            "Torque [Nm]": 39.8,
            "Tool wear [min]": 21.0,
            "Operational_High_Load": 1,
        }
        neighbors = self.clf.find_k_nearest(query, k=3, exclude_self=True)
        self.assertEqual(len(neighbors), 3)
        self.assertEqual(neighbors[0]["record"]["udi"], 1)
        self.assertGreater(neighbors[0]["similarity"], 0.95)

        # Monotonically non-decreasing distances
        for i in range(len(neighbors) - 1):
            self.assertLessEqual(neighbors[i]["distance"], neighbors[i + 1]["distance"])

    def test_exclude_self_from_neighbors(self):
        """Verify a record in the training dataset does not select itself as neighbor."""
        # Query record 1 directly from dataset
        rec_1 = self.dataset.records[0]
        self.assertEqual(rec_1["udi"], 1)

        # When exclude_self=True, record 1 MUST NOT be in retrieved neighbors
        neighbors = self.clf.find_k_nearest(rec_1, k=3, exclude_self=True)
        returned_udis = [n["record"]["udi"] for n in neighbors]
        self.assertNotIn(1, returned_udis)
        self.assertEqual(len(neighbors), 3)

    def test_target_leakage_exclusion(self):
        """Verify excluded columns are not in default active predictor features."""
        active_features = (
            self.dataset.numeric_cols
            + self.dataset.nominal_cols
            + self.dataset.ordinal_cols
            + self.dataset.binary_cols
        )
        for col in EXCLUDED_FROM_FEATURES:
            self.assertNotIn(
                col,
                active_features,
                f"Target leakage detected: {col} is present in predictor features!",
            )

    def test_train_test_split_range_fitting(self):
        """Verify numeric ranges are fitted strictly on train split, not test split."""
        recs = [
            {
                "udi": i,
                "Type": "L",
                "Air temperature [K]": float(i * 10),
                "Process temperature [K]": 308.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 20.0,
                "Operational_High_Load": 0,
                "condition": "Normal" if i % 2 == 0 else "Warning",
            }
            for i in range(1, 11)
        ]
        ds = CNCDataset(recs)
        train_ds, test_ds = train_test_split(ds, test_ratio=0.3, seed=42, stratify=False)

        # Train dataset calculated ranges
        self.assertIn("Air temperature [K]", train_ds.ranges)
        # Test dataset inherited train ranges (no independent range calculation)
        self.assertEqual(test_ds.ranges, train_ds.ranges)

    def test_prediction_normal(self):
        query = {
            "Type": "L",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.1,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 22.0,
            "Operational_High_Load": 1,
        }
        pred = self.clf.predict(query, k=3)
        self.assertEqual(pred["predicted_condition"], "Normal")

    def test_prediction_critical(self):
        query = {
            "Type": "H",
            "Air temperature [K]": 304.1,
            "Process temperature [K]": 312.1,
            "Rotational speed [rpm]": 1240.0,
            "Torque [Nm]": 66.0,
            "Tool wear [min]": 235.0,
            "Operational_High_Load": 1,
        }
        pred = self.clf.predict(query, k=3)
        self.assertEqual(pred["predicted_condition"], "Critical")

    def test_condition_derivation_precedence(self):
        # Precedence 1: Machine failure = 1 -> Critical (even with low tool wear)
        self.assertEqual(
            derive_condition(machine_failure=1, air_temp=298.0, process_temp=308.0, speed_rpm=1500.0, torque_nm=40.0, tool_wear=10.0),
            "Critical",
        )
        # Precedence 2: Machine failure = 0, Tool wear >= 180 -> Warning
        self.assertEqual(
            derive_condition(machine_failure=0, air_temp=298.0, process_temp=308.0, speed_rpm=1500.0, torque_nm=40.0, tool_wear=185.0),
            "Warning",
        )
        # Precedence 2: Machine failure = 0, Temp diff <= 9.0 K -> Warning
        self.assertEqual(
            derive_condition(machine_failure=0, air_temp=300.0, process_temp=308.5, speed_rpm=1500.0, torque_nm=40.0, tool_wear=50.0),
            "Warning",
        )
        # Precedence 3: Machine failure = 0, all nominal -> Normal
        self.assertEqual(
            derive_condition(machine_failure=0, air_temp=298.0, process_temp=308.5, speed_rpm=1500.0, torque_nm=40.0, tool_wear=50.0),
            "Normal",
        )

    def test_make_query_record_matches_feature_schema(self):
        query = make_query_record("m", 298.5, 308.9, 1450.0, 42.0, 0.0)
        active_cols = (
            self.dataset.numeric_cols + self.dataset.nominal_cols
            + self.dataset.ordinal_cols + self.dataset.binary_cols
        )
        self.assertEqual(set(query.keys()), set(active_cols))
        self.assertEqual(query["Type"], "M")
        # Zero tool wear is a valid reading and must not be replaced by a default
        self.assertEqual(query["Tool wear [min]"], 0.0)
        # High-load flag derived from torque when not supplied
        self.assertEqual(query["Operational_High_Load"], 1)
        self.assertEqual(make_query_record("L", 298.5, 308.9, 1450.0, 39.9, 10.0)["Operational_High_Load"], 0)

    def test_make_query_record_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            make_query_record("X", 298.5, 308.9, 1450.0, 42.0, 10.0)
        with self.assertRaises(ValueError):
            make_query_record("L", None, 308.9, 1450.0, 42.0, 10.0)
        with self.assertRaises(ValueError):
            make_query_record("L", 298.5, 308.9, -5.0, 42.0, 10.0)
        with self.assertRaises(ValueError):
            make_query_record("L", 298.5, 308.9, 1450.0, 42.0, 10.0, high_load=2)


if __name__ == "__main__":
    unittest.main()
