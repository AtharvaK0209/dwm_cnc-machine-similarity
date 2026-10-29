# CNC Machine Similarity and Condition Classification

> **DWM Lab Assignment** — Mixed-Attribute Gower Similarity Metric & $k$-Nearest Neighbors Condition Classifier.

---

## Academic Information
- **Class / Division:** Third Year (TE), Computer Engineering
- **Course:** Data Warehousing and Mining (DWM) Lab
- **Group:** 4

### Team Members
- **Atharva Khandagale** — `124A1061`
- **Surya Kaushik** — `124A1055`
- **Prakash Biradar** — `124A1013`
- **Rakesh Biradar** — `124A1014`

---

## Project Overview

This project provides an audited, end-to-end data mining system for CNC machine condition monitoring and similarity search:
1. **Multi-Attribute Proximity Calculation:** Calculates pairwise and query-to-dataset similarity using **Gower's General Distance Metric**, natively supporting nominal, binary, ordinal, and numeric interval-scaled attributes without arbitrary numeric coercion.
2. **Similar Machine Identification:** Finds top-$k$ nearest neighbors from 10,000 industrial machine records with full distance component breakdowns and self-neighbor exclusion.
3. **Condition Classification:** Classifies machines into 3 operational health states (**Normal**, **Warning**, **Critical**) using a distance-weighted $k$-Nearest Neighbors ($k$-NN) classifier.
4. **Target Leakage Prevention:** Eliminates row identifiers and failure-mode flags (`UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from predictor features.
5. **Preprocessing Isolation:** Numeric normalization ranges ($R_f$) are fitted strictly on the training partition to prevent test set data leakage.
6. **Zero Heavy External Dependencies:** Pure Python 3.14 + standard library / NumPy implementation.

---

## Dataset & Attribute Taxonomy

- **Dataset:** `data/ai4i2020.csv` (Stephan Matzka, *AI4I 2020 Predictive Maintenance Dataset*, UCI ML Repository).
- **Scale:** 10,000 data rows, 14 columns, 0 missing values.
- **Data Nature:** Synthetic physical simulation model mirroring milling machine tool wear dynamics.

### Attribute Mapping
| DWM Category | Dataset Feature | Mapping / Normalization Range | Distance Component $d_{ij}^{(f)}$ |
|---|---|---|---|
| **Nominal** (Default) | `Type` | Machine variant (`L`, `M`, `H`) | $d=0.0$ if equal, $1.0$ if unequal |
| **Binary** (Symmetric) | `Operational_High_Load` | $\text{Torque} \ge 40.0\text{ Nm}$ (Operational setting) | $d=0.0$ if equal, $1.0$ if unequal |
| **Numeric Interval** | `Air temperature [K]` | $[295.3, 304.5]\text{ K}$ ($R_f = 9.2\text{ K}$) | $\frac{\|x_i - x_j\|}{R_f}$ |
| **Numeric Interval** | `Process temperature [K]` | $[305.7, 313.8]\text{ K}$ ($R_f = 8.1\text{ K}$) | $\frac{\|x_i - x_j\|}{R_f}$ |
| **Numeric Ratio** | `Rotational speed [rpm]` | $[1168, 2886]\text{ rpm}$ ($R_f = 1718\text{ rpm}$) | $\frac{\|x_i - x_j\|}{R_f}$ |
| **Numeric Ratio** | `Torque [Nm]` | $[3.8, 76.6]\text{ Nm}$ ($R_f = 72.8\text{ Nm}$) | $\frac{\|x_i - x_j\|}{R_f}$ |
| **Numeric Ratio** | `Tool wear [min]` | $[0, 253]\text{ min}$ ($R_f = 253\text{ min}$) | $\frac{\|x_i - x_j\|}{R_f}$ |

*Ordinal attributes are fully supported by the Gower distance engine with rank normalization ($z = \frac{r-1}{M-1}$) and verified in unit tests, while `Type` is kept nominal by default to avoid ungrounded ordering assumptions.*

---

## Label Engineering: 3-Tier Operational Health

The raw dataset provides only binary `Machine failure` ($0$ or $1$). The 3-tier condition classes are engineered using Stephan Matzka's documented failure physics:

1. **Critical (Precedence 1 - Highest):**
   `Machine failure == 1` (**339 records, 3.39%**). Actual breakdown occurred.
2. **Warning (Precedence 2):**
   `Machine failure == 0` AND (`Tool wear >= 180 min` OR `Process temp - Air temp <= 9.0 K` OR `Power not in [3800, 8500] W`) (**2,997 records, 29.97%**). Incipient fault regime.
3. **Normal (Precedence 3 - Baseline):**
   `Machine failure == 0` and within safe nominal operational thresholds (**6,664 records, 66.64%**).

*Note: These are synthetic heuristic rules created to satisfy the 3-class assignment requirement. Evaluation against these labels measures classifier agreement with the engineered boundaries rather than validated real-world failure foresight.*

---

## Mathematical Formulation

### 1. Gower's Similarity Metric
$$d(i, j) = \frac{\sum_{f=1}^p w_f \delta_{ij}^{(f)} d_{ij}^{(f)}}{\sum_{f=1}^p w_f \delta_{ij}^{(f)}}, \quad S(i, j) = 1 - d(i, j)$$
Where:
- $\delta_{ij}^{(f)} = 1.0$ if both values are valid; $0.0$ if either is `None`, empty, or `NaN`.
- Component distances $d_{ij}^{(f)} \in [0, 1]$.
- Zero-range numeric features return $d_f = 0.0, \delta_f = 1.0$.

### 2. Distance-Weighted $k$-NN Classifier
For query $\mathbf{x}_q$ and its $k$ closest neighbors $\mathcal{N}_k(q)$ (with self-exclusion):
$$w_n = \frac{1}{d(q, n) + 10^{-5}}, \quad \hat{y}_q = \arg\max_{c} \sum_{n \in \mathcal{N}_k(q), y_n = c} w_n$$

---

## Project Structure

```
dwm_cnc-machine-similarity/
├── README.md                      # Academic documentation, math, team details, execution guide
├── data/
│   └── ai4i2020.csv               # AI4I 2020 dataset (10,000 records)
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Pure CSV loader, attribute categorization, 3-tier label generator
│   ├── gower.py                   # Gower distance & similarity metric implementation
│   ├── knn.py                     # Custom k-NN classifier (majority & distance-weighted voting)
│   └── pipeline.py                # Interactive CLI tool
├── tests/
│   ├── __init__.py
│   ├── test_gower.py              # Unit tests for Gower metric components, missing values, & axioms
│   └── test_knn.py                # Unit tests for neighbor ordering, self-exclusion, & leakage prevention
└── report/
    └── report_evidence.md         # Full tables, confusion matrix, and report write-up
```

---

## Execution Guide

### 1. Run All Unit Tests
Executes 15 comprehensive tests covering metric axioms, missing values, zero-range features, self-exclusion, and target leakage prevention:
```bash
python -m unittest discover -s tests -v
```

### 2. Run Automated 3-Scenario Demonstration
Runs Healthy Baseline, Incipient Wear, and Heavy Load scenarios showing top-5 neighbor retrieval and similarity breakdown:
```bash
python -m src.pipeline --demo
```

### 3. Classify a Custom Machine Query
Query operational parameters directly:
```bash
python -m src.pipeline --type M --air-temp 298.5 --proc-temp 308.9 --speed 1450 --torque 42.0 --wear 190.0 -k 5
```

### 4. Run Model Evaluation
Compute accuracy, confusion matrix, precision, recall, and F1-score on a stratified test sample:
```bash
python -m src.pipeline --evaluate --sample-eval-size 300 -k 5
```

---

## Verified Evaluation Results

Evaluated on stratified held-out test split ($k=5$, ranges fit on train split only):
- **Overall Accuracy:** **88.59%** (298 test instances)
- **Normal:** Precision 92.61%, Recall 94.47%, F1 93.53%
- **Warning:** Precision 80.43%, Recall 83.15%, F1 81.77%
- **Critical:** Precision 66.67%, Recall 20.00%, F1 30.77%

Full confusion matrices and report evidence are documented in [report/report_evidence.md](file:///c:/Users/athar/Desktop/Coding/dwm_cnc-machine-similarity/report/report_evidence.md).

---

## References
1. Gower, J. C. (1971). *A General Coefficient of Similarity and Some of Its Properties*. Biometrics, 27(4), 857–871.
2. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann. Chapter 2.4.4.
3. Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. Third International Conference on Artificial Intelligence for Industries (AI4I), IEEE.
