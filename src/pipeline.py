"""Interactive CLI pipeline for CNC machine similarity search and condition classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from .dataset import (
    CONDITION_CLASSES,
    CNCDataset,
    load_dataset,
    train_test_split,
)
from .knn import GowerKNNClassifier


def print_banner() -> None:
    banner = """
================================================================================
   DWM Lab: CNC Machine Similarity & Condition Classification System
   Group 4 | TE Computer Engineering | Gower Distance & k-NN Classifier
   [Audit: Zero Failure-Mode Leakage | Type=Nominal | Operational Binary Load]
================================================================================
    """
    print(banner.strip())


def print_neighbor_table(neighbors: list[dict[str, Any]]) -> None:
    header = f"{'Rank':<5} | {'UDI':<6} | {'Product ID':<10} | {'Type':<4} | {'Similarity':<10} | {'Condition':<10} | {'Key Factors (d_f)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for rank, n in enumerate(neighbors, 1):
        rec = n["record"]
        sim_pct = f"{n['similarity'] * 100:.2f}%"
        udi = str(rec.get("udi", "N/A"))
        pid = str(rec.get("product_id", "N/A"))
        mtype = str(rec.get("Type", "N/A"))
        cond = str(n["condition"])

        # Highlight top non-zero distance contributors
        bd = n["breakdown"]
        top_diffs = sorted(bd.items(), key=lambda x: x[1], reverse=True)[:2]
        diff_str = ", ".join(f"{k.split()[0]} diff={v:.2f}" for k, v in top_diffs)

        print(
            f"{rank:<5} | {udi:<6} | {pid:<10} | {mtype:<4} | {sim_pct:<10} | {cond:<10} | {diff_str}"
        )
    print("-" * len(header))


def run_demo(dataset: CNCDataset, k: int = 5) -> None:
    print("\n>>> Running Automated Demonstration with 3 Canonical Machine Scenarios <<<\n")
    classifier = GowerKNNClassifier(k=k, weighting="distance").fit(dataset)

    test_cases = [
        {
            "name": "Scenario 1: Healthy Operational Machine (Nominal Baseline)",
            "query": {
                "Type": "M",
                "Air temperature [K]": 298.2,
                "Process temperature [K]": 308.7,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 25.0,
                "Operational_High_Load": 1,
            },
        },
        {
            "name": "Scenario 2: Severely Worn Machine (Tool Wear Warning Regime)",
            "query": {
                "Type": "L",
                "Air temperature [K]": 302.5,
                "Process temperature [K]": 311.2,
                "Rotational speed [rpm]": 1350.0,
                "Torque [Nm]": 58.0,
                "Tool wear [min]": 215.0,  # High tool wear
                "Operational_High_Load": 1,
            },
        },
        {
            "name": "Scenario 3: Narrow Thermal Margin & Heavy Mechanical Load",
            "query": {
                "Type": "H",
                "Air temperature [K]": 304.1,
                "Process temperature [K]": 312.0,  # Narrow Delta-T: 7.9 K
                "Rotational speed [rpm]": 1280.0,
                "Torque [Nm]": 68.5,  # High Torque
                "Tool wear [min]": 195.0,
                "Operational_High_Load": 1,
            },
        },
    ]

    for case in test_cases:
        print(f"\n=======================================================")
        print(f"[*] {case['name']}")
        print(f"=======================================================")
        q = case["query"]
        print("Input Predictor Features (No Failure Flags):")
        print(f"  - Machine Type (Nominal): {q['Type']}")
        print(f"  - Air Temperature: {q['Air temperature [K]']} K")
        print(f"  - Process Temperature: {q['Process temperature [K]']} K")
        print(f"  - Rotational Speed: {q['Rotational speed [rpm]']} rpm")
        print(f"  - Torque: {q['Torque [Nm]']} Nm")
        print(f"  - Tool Wear: {q['Tool wear [min]']} min")
        print(f"  - Operational High Load Flag (Binary): {q['Operational_High_Load']}")

        result = classifier.predict(q, k=k, exclude_self=True)
        pred = result["predicted_condition"]
        probs = result["probabilities"]

        print(f"\n[+] Top-{k} Most Similar Machines Identified:")
        print_neighbor_table(result["neighbors"])

        print("\n[+] Classification Decision:")
        for c in CONDITION_CLASSES:
            bar = "#" * int(probs[c] * 30)
            print(f"    {c:<10}: {probs[c] * 100:6.2f}% | {bar}")
        print(f"\n>>> PREDICTED MACHINE CONDITION: [ {pred.upper()} ] <<<\n")


def run_evaluation(dataset: CNCDataset, k: int = 5, sample_size: int = 1000) -> None:
    print(f"\n>>> Evaluating Gower k-NN Classifier (k={k}) on Stratified Test Split <<<")
    train_data, test_data = train_test_split(dataset, test_ratio=0.2, seed=42)

    # Subsample test set if specified
    if len(test_data) > sample_size:
        print(f"Subsampling test set to {sample_size} records for responsive benchmark...")
        test_sample, _ = train_test_split(test_data, test_ratio=1.0 - (sample_size / len(test_data)), seed=42)
    else:
        test_sample = test_data

    print(f"Training instances: {len(train_data)} | Test instances: {len(test_sample)}")
    print("Pre-processing: Numeric normalization ranges fitted strictly on training partition.")

    classifier = GowerKNNClassifier(k=k, weighting="distance").fit(train_data)
    eval_res = classifier.evaluate(test_sample, k=k)

    print("\n--- Evaluation Methodological Context ---")
    print("NOTE: The 3-tier condition classes were derived using physical heuristic rules on")
    print("sensor readings (Tool wear >= 180 min, narrow temp diff, power thresholds, or breakdown).")
    print("Therefore, this evaluation measures k-NN's boundary agreement with the engineered")
    print("rules, NOT validated performance against unobserved real-world predictive maintenance logs.")

    print("\n--- Overall Performance ---")
    print(f"Accuracy: {eval_res['accuracy'] * 100:.2f}% ({eval_res['total_tested']} test instances)")

    print("\n--- Confusion Matrix (Actual vs Predicted) ---")
    header = f"{'Actual \\ Pred':<15} | " + " | ".join(f"{c:<10}" for c in CONDITION_CLASSES)
    print(header)
    print("-" * len(header))
    cm = eval_res["confusion_matrix"]
    for actual in CONDITION_CLASSES:
        row = f"{actual:<15} | " + " | ".join(f"{cm[actual][pred]:<10}" for pred in CONDITION_CLASSES)
        print(row)

    print("\n--- Per-Class Classification Report ---")
    rpt_header = f"{'Condition':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
    print(rpt_header)
    print("-" * len(rpt_header))
    metrics = eval_res["per_class_metrics"]
    for c in CONDITION_CLASSES:
        m = metrics[c]
        print(
            f"{c:<12} | {m['precision'] * 100:9.2f}% | {m['recall'] * 100:9.2f}% | {m['f1_score'] * 100:9.2f}% | {m['support']:<8}"
        )


def main() -> None:
    print_banner()

    parser = argparse.ArgumentParser(
        description="CNC Machine Similarity Search and Condition Classification using Gower's Metric and k-NN."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ai4i2020.csv",
        help="Path to AI4I 2020 dataset CSV file.",
    )
    parser.add_argument(
        "-k",
        "--neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve (default: 5).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run automated demonstration on 3 representative condition scenarios.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run stratified evaluation and output accuracy, confusion matrix, and precision/recall.",
    )
    parser.add_argument(
        "--sample-eval-size",
        type=int,
        default=300,
        help="Evaluation test sample size (default: 300).",
    )
    # Query parameters
    parser.add_argument("--type", type=str, choices=["L", "M", "H"], help="Machine type (nominal: L, M, H).")
    parser.add_argument("--air-temp", type=float, help="Air temperature [K] (e.g., 298.5).")
    parser.add_argument("--proc-temp", type=float, help="Process temperature [K] (e.g., 308.7).")
    parser.add_argument("--speed", type=float, help="Rotational speed [rpm] (e.g., 1500).")
    parser.add_argument("--torque", type=float, help="Torque [Nm] (e.g., 40.0).")
    parser.add_argument("--wear", type=float, help="Tool wear [min] (e.g., 120.0).")
    parser.add_argument(
        "--high-load",
        type=int,
        choices=[0, 1],
        default=None,
        help="Operational high-load flag (binary 0 or 1; auto-derived from torque if omitted).",
    )

    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"Error: Dataset not found at '{csv_path}'. Please check the path.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from: {csv_path.resolve()} ...")
    dataset = load_dataset(csv_path)
    print(f"Loaded {len(dataset)} records. Attribute ranges calculated.")

    if args.evaluate:
        run_evaluation(dataset, k=args.neighbors, sample_size=args.sample_eval_size)
    elif args.demo or (args.type is None and not args.evaluate):
        run_demo(dataset, k=args.neighbors)
    else:
        # Custom query execution
        torque_val = args.torque or 40.0
        high_load_val = args.high_load if args.high_load is not None else (1 if torque_val >= 40.0 else 0)

        query = {
            "Type": args.type,
            "Air temperature [K]": args.air_temp or 300.0,
            "Process temperature [K]": args.proc_temp or 310.0,
            "Rotational speed [rpm]": args.speed or 1500.0,
            "Torque [Nm]": torque_val,
            "Tool wear [min]": args.wear or 100.0,
            "Operational_High_Load": high_load_val,
        }
        print("\n--- User Query Machine ---")
        for k, v in query.items():
            print(f"  {k}: {v}")

        classifier = GowerKNNClassifier(k=args.neighbors, weighting="distance").fit(dataset)
        result = classifier.predict(query, k=args.neighbors, exclude_self=True)

        print(f"\n[+] Top-{args.neighbors} Most Similar Machines Identified:")
        print_neighbor_table(result["neighbors"])

        print(f"\n>>> PREDICTED MACHINE CONDITION: [ {result['predicted_condition'].upper()} ]")
        print("Class Probabilities:")
        for c in CONDITION_CLASSES:
            print(f"  {c:<10}: {result['probabilities'][c] * 100:.2f}%")


if __name__ == "__main__":
    main()
