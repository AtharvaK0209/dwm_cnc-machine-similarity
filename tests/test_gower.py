"""Unit tests for Gower distance and similarity metric calculation."""

import math
import unittest
from src.gower import (
    compute_binary_component,
    compute_nominal_component,
    compute_numeric_component,
    compute_ordinal_component,
    gower_distance_single,
    gower_similarity,
)


class TestGowerMetric(unittest.TestCase):
    def test_numeric_component(self):
        # Difference 5 with range 20 -> 5/20 = 0.25, delta=1.0
        d, delta = compute_numeric_component(10.0, 15.0, 20.0)
        self.assertAlmostEqual(d, 0.25)
        self.assertEqual(delta, 1.0)

        # Identical values -> 0.0
        d, delta = compute_numeric_component(10.0, 10.0, 20.0)
        self.assertAlmostEqual(d, 0.0)
        self.assertEqual(delta, 1.0)

        # Difference exceeding range capped at 1.0
        d, delta = compute_numeric_component(0.0, 50.0, 20.0)
        self.assertAlmostEqual(d, 1.0)

        # Zero range handled gracefully without ZeroDivisionError
        d, delta = compute_numeric_component(10.0, 15.0, 0.0)
        self.assertAlmostEqual(d, 0.0)
        self.assertEqual(delta, 1.0)

    def test_numeric_missing_values(self):
        # Missing values (None, NaN, empty string) must yield delta = 0.0
        d, delta = compute_numeric_component(None, 10.0, 20.0)
        self.assertEqual(delta, 0.0)

        d, delta = compute_numeric_component(10.0, float("nan"), 20.0)
        self.assertEqual(delta, 0.0)

        d, delta = compute_numeric_component("", 10.0, 20.0)
        self.assertEqual(delta, 0.0)

    def test_nominal_component(self):
        d, delta = compute_nominal_component("L", "L")
        self.assertEqual((d, delta), (0.0, 1.0))

        d, delta = compute_nominal_component("M", "H")
        self.assertEqual((d, delta), (1.0, 1.0))

        # Whitespace stripping
        d, delta = compute_nominal_component(" L ", "L")
        self.assertEqual((d, delta), (0.0, 1.0))

        # Missing values
        d, delta = compute_nominal_component(None, "L")
        self.assertEqual(delta, 0.0)

        d, delta = compute_nominal_component("M", "")
        self.assertEqual(delta, 0.0)

    def test_ordinal_component(self):
        states = ["L", "M", "H"]
        # L (rank 1 -> z=0.0) and H (rank 3 -> z=1.0) -> dist = 1.0
        d, delta = compute_ordinal_component("L", "H", states)
        self.assertAlmostEqual(d, 1.0)
        self.assertEqual(delta, 1.0)

        # L (z=0.0) and M (rank 2 -> z=0.5) -> dist = 0.5
        d, delta = compute_ordinal_component("L", "M", states)
        self.assertAlmostEqual(d, 0.5)

        # M (z=0.5) and H (z=1.0) -> dist = 0.5
        d, delta = compute_ordinal_component("M", "H", states)
        self.assertAlmostEqual(d, 0.5)

        # Missing or unknown state yields delta = 0.0
        d, delta = compute_ordinal_component(None, "M", states)
        self.assertEqual(delta, 0.0)

        d, delta = compute_ordinal_component("UNKNOWN", "M", states)
        self.assertEqual(delta, 0.0)

    def test_binary_component(self):
        # Symmetric binary
        d, delta = compute_binary_component(0, 0, asymmetric=False)
        self.assertEqual((d, delta), (0.0, 1.0))
        d, delta = compute_binary_component(1, 1, asymmetric=False)
        self.assertEqual((d, delta), (0.0, 1.0))
        d, delta = compute_binary_component(1, 0, asymmetric=False)
        self.assertEqual((d, delta), (1.0, 1.0))

        # Asymmetric binary (both 0 means non-informative -> delta = 0)
        d, delta = compute_binary_component(0, 0, asymmetric=True)
        self.assertEqual((d, delta), (0.0, 0.0))
        d, delta = compute_binary_component(1, 1, asymmetric=True)
        self.assertEqual((d, delta), (0.0, 1.0))

        # Missing value
        d, delta = compute_binary_component(None, 1, asymmetric=False)
        self.assertEqual(delta, 0.0)

    def test_gower_mixed_all_four_types(self):
        """Test Gower metric with nominal, binary, ordinal, and numeric simultaneously."""
        rec_a = {
            "Type": "L",
            "Speed_Tier": "Low",  # Ordinal
            "Operational_High_Load": 0,  # Binary
            "Air temperature [K]": 300.0,  # Numeric
        }
        rec_b = {
            "Type": "L",  # Nominal match -> 0.0
            "Speed_Tier": "High",  # Ordinal: Low (0.0) vs High (1.0) -> 1.0
            "Operational_High_Load": 1,  # Binary mismatch -> 1.0
            "Air temperature [K]": 305.0,  # Numeric: diff 5.0 / range 10.0 -> 0.5
        }
        ranges = {"Air temperature [K]": (295.0, 305.0, 10.0)}

        # Expected distance = (0.0 + 1.0 + 1.0 + 0.5) / 4 = 2.5 / 4 = 0.625
        dist = gower_distance_single(
            record_a=rec_a,
            record_b=rec_b,
            numeric_ranges=ranges,
            numeric_cols=["Air temperature [K]"],
            nominal_cols=["Type"],
            ordinal_cols=["Speed_Tier"],
            binary_cols=["Operational_High_Load"],
            ordinal_orders={"Speed_Tier": ["Low", "Medium", "High"]},
        )
        self.assertAlmostEqual(dist, 0.625)
        self.assertAlmostEqual(gower_similarity(dist), 1.0 - 0.625)

    def test_gower_identical_records(self):
        rec = {
            "Type": "M",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100.0,
            "Operational_High_Load": 1,
        }
        ranges = {
            "Air temperature [K]": (295.0, 305.0, 10.0),
            "Process temperature [K]": (305.0, 315.0, 10.0),
            "Rotational speed [rpm]": (1000.0, 2500.0, 1500.0),
            "Torque [Nm]": (10.0, 80.0, 70.0),
            "Tool wear [min]": (0.0, 250.0, 250.0),
        }
        dist = gower_distance_single(rec, rec, numeric_ranges=ranges)
        self.assertAlmostEqual(dist, 0.0)
        self.assertAlmostEqual(gower_similarity(dist), 1.0)

    def test_gower_symmetry(self):
        rec_a = {
            "Type": "L",
            "Air temperature [K]": 298.0,
            "Process temperature [K]": 308.0,
            "Rotational speed [rpm]": 1400.0,
            "Torque [Nm]": 35.0,
            "Tool wear [min]": 50.0,
            "Operational_High_Load": 0,
        }
        rec_b = {
            "Type": "H",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 312.0,
            "Rotational speed [rpm]": 1700.0,
            "Torque [Nm]": 55.0,
            "Tool wear [min]": 150.0,
            "Operational_High_Load": 1,
        }
        ranges = {
            "Air temperature [K]": (295.0, 305.0, 10.0),
            "Process temperature [K]": (305.0, 315.0, 10.0),
            "Rotational speed [rpm]": (1000.0, 2500.0, 1500.0),
            "Torque [Nm]": (10.0, 80.0, 70.0),
            "Tool wear [min]": (0.0, 250.0, 250.0),
        }
        dist_ab = gower_distance_single(rec_a, rec_b, numeric_ranges=ranges)
        dist_ba = gower_distance_single(rec_b, rec_a, numeric_ranges=ranges)
        self.assertAlmostEqual(dist_ab, dist_ba)
        self.assertTrue(0.0 <= dist_ab <= 1.0)


if __name__ == "__main__":
    unittest.main()
