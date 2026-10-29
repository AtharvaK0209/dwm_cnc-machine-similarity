"""k-Nearest Neighbors classifier powered by Gower distance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .dataset import CNCDataset, CONDITION_CLASSES
from .gower import gower_distance_single, gower_similarity


class GowerKNNClassifier:
    """k-Nearest Neighbors classifier for mixed-attribute CNC machine records."""

    def __init__(
        self,
        k: int = 5,
        weighting: str = "distance",
        weights: Optional[Dict[str, float]] = None,
        asymmetric_binary: bool = False,
    ):
        """Initialize k-NN classifier.

        Args:
            k: Number of nearest neighbors.
            weighting: 'distance' (w = 1 / (d + eps)) or 'uniform' (majority vote).
            weights: Optional attribute weight dictionary for Gower metric.
            asymmetric_binary: Whether binary flags are treated as asymmetric.
        """
        self.k = k
        self.weighting = weighting
        self.weights = weights
        self.asymmetric_binary = asymmetric_binary
        self.train_dataset: Optional[CNCDataset] = None
        self.numeric_ranges: Dict[str, Tuple[float, float, float]] = {}

    def fit(self, train_dataset: CNCDataset) -> "GowerKNNClassifier":
        """Fit classifier by caching training records and attribute normalization ranges."""
        self.train_dataset = train_dataset
        self.numeric_ranges = dict(train_dataset.ranges)
        return self

    def find_k_nearest(
        self,
        query: Dict[str, Any],
        k: Optional[int] = None,
        exclude_self: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k nearest neighbors in training data sorted by Gower distance.

        Args:
            query: Attribute dictionary of query machine.
            k: Number of neighbors to return.
            exclude_self: If True, prevents a query record from selecting itself as
                its own nearest neighbor (matched by identity, UDI, or Product ID).
        """
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError("Classifier has not been fitted with training data.")

        k = k or self.k
        neighbors: List[Dict[str, Any]] = []

        query_udi = query.get("udi")
        query_pid = query.get("product_id")

        for train_rec in self.train_dataset.records:
            # Self-exclusion check
            if exclude_self:
                if query is train_rec:
                    continue
                if query_udi is not None and train_rec.get("udi") == query_udi:
                    continue
                if query_pid is not None and train_rec.get("product_id") == query_pid:
                    continue

            dist, breakdown = gower_distance_single(
                record_a=query,
                record_b=train_rec,
                numeric_ranges=self.numeric_ranges,
                weights=self.weights,
                numeric_cols=self.train_dataset.numeric_cols,
                nominal_cols=self.train_dataset.nominal_cols,
                ordinal_cols=self.train_dataset.ordinal_cols,
                binary_cols=self.train_dataset.binary_cols,
                asymmetric_binary=self.asymmetric_binary,
                return_breakdown=True,
            )
            sim = gower_similarity(dist)
            neighbors.append(
                {
                    "distance": dist,
                    "similarity": sim,
                    "record": train_rec,
                    "breakdown": breakdown,
                    "condition": train_rec["condition"],
                }
            )

        # Sort by distance ascending (similarity descending)
        neighbors.sort(key=lambda item: item["distance"])
        return neighbors[:k]

    def predict(
        self,
        query: Dict[str, Any],
        k: Optional[int] = None,
        exclude_self: bool = True,
    ) -> Dict[str, Any]:
        """Predict machine condition (Normal, Warning, Critical) for query record."""
        k = k or self.k
        k_neighbors = self.find_k_nearest(query, k=k, exclude_self=exclude_self)

        eps = 1e-5
        class_scores: Dict[str, float] = {c: 0.0 for c in CONDITION_CLASSES}

        for neighbor in k_neighbors:
            label = neighbor["condition"]
            dist = neighbor["distance"]
            if self.weighting == "distance":
                weight = 1.0 / (dist + eps)
            else:
                weight = 1.0
            neighbor["vote_weight"] = weight
            class_scores[label] += weight

        total_weight = sum(class_scores.values())
        probabilities = {
            c: (score / total_weight if total_weight > 0 else 0.0)
            for c, score in class_scores.items()
        }

        # Select class with highest score (tie break by class order)
        predicted_condition = max(
            CONDITION_CLASSES,
            key=lambda c: (class_scores[c], -CONDITION_CLASSES.index(c)),
        )

        return {
            "predicted_condition": predicted_condition,
            "probabilities": probabilities,
            "class_scores": class_scores,
            "neighbors": k_neighbors,
        }

    def evaluate(
        self,
        test_dataset: CNCDataset,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate classifier performance on test dataset.

        Note: When test records are distinct from train records, exclude_self is
        naturally satisfied, but kept enabled as a safeguard against identity leakage.
        """
        k = k or self.k
        y_true: List[str] = []
        y_pred: List[str] = []

        for rec in test_dataset.records:
            res = self.predict(rec, k=k, exclude_self=True)
            y_true.append(rec["condition"])
            y_pred.append(res["predicted_condition"])

        # Confusion matrix: rows = actual, cols = predicted
        confusion: Dict[str, Dict[str, int]] = {
            actual: {pred: 0 for pred in CONDITION_CLASSES} for actual in CONDITION_CLASSES
        }
        for true_c, pred_c in zip(y_true, y_pred):
            confusion[true_c][pred_c] += 1

        # Per-class metrics
        metrics: Dict[str, Dict[str, float]] = {}
        total_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        overall_accuracy = total_correct / len(y_true) if y_true else 0.0

        for c in CONDITION_CLASSES:
            tp = confusion[c][c]
            fp = sum(confusion[other][c] for other in CONDITION_CLASSES if other != c)
            fn = sum(confusion[c][other] for other in CONDITION_CLASSES if other != c)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2.0 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics[c] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": tp + fn,
            }

        return {
            "accuracy": overall_accuracy,
            "confusion_matrix": confusion,
            "per_class_metrics": metrics,
            "total_tested": len(y_true),
        }
