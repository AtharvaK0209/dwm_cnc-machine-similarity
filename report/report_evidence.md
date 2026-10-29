# DWM Lab Assignment Report Evidence: CNC Machine Similarity & Condition Classification

**Academic Term:** Third Year (TE), Computer Engineering  
**Course:** Data Warehousing and Mining (DWM) Lab  
**Group Number:** 4  

### Team Members
1. **Atharva Khandagale** — Roll No: `124A1061`
2. **Surya Kaushik** — Roll No: `124A1055`
3. **Prakash Biradar** — Roll No: `124A1013`
4. **Rakesh Biradar** — Roll No: `124A1014`

---

## 1. Problem Statement & Objectives
Milling machines in high-throughput manufacturing operate under varying rotational speeds, cutting torque, and thermal dissipation constraints.
1. **Mixed-Attribute Proximity:** Formulate and execute **Gower's General Distance Metric** across nominal, binary, ordinal, and continuous interval-scaled attributes without arbitrary numeric coercion.
2. **Similar Machine Identification:** Retrieve top-$k$ nearest machines with detailed component distance breakdowns.
3. **3-Tier Condition Classification:** Classify target machines into **Normal**, **Warning**, and **Critical** operational states using a distance-weighted $k$-Nearest Neighbors ($k$-NN) classifier.
4. **Methodological Rigor:** Prevent target leakage, eliminate failure flags from predictor inputs, handle missing and zero-range values gracefully, and evaluate strictly with training-fitted normalization ranges.

---

## 2. Dataset & Attribute Taxonomy

**Source:** Stephan Matzka, *AI4I 2020 Predictive Maintenance Dataset*, UCI Machine Learning Repository.  
**Data Nature:** **Synthetic physical simulation model** (Matzka, 2020) reflecting realistic milling machine tool dynamics.  
**Size:** 10,000 data rows, 14 raw columns, 0 missing values.

### Attribute Mapping
| Attribute Name | Raw Type | DWM Category | Mapping / Range $R_f$ | Distance Component $d_{ij}^{(f)}$ |
|---|---|---|---|---|
| `Type` | String (`L`, `M`, `H`) | **Nominal** (Default) | Categorical equality | $0.0$ if equal, $1.0$ if unequal |
| `Operational_High_Load` | Integer ($0$ or $1$) | **Binary** (Symmetric) | $\text{Torque} \ge 40.0\text{ Nm}$ (Operational setting) | $0.0$ if equal, $1.0$ if unequal |
| `Air temperature [K]` | Float | **Numeric Interval** | Min: $295.3$, Max: $304.5$ ($R_f = 9.2\text{ K}$) | $\frac{\|x_i - x_j\|}{R_f}$ |
| `Process temperature [K]` | Float | **Numeric Interval** | Min: $305.7$, Max: $313.8$ ($R_f = 8.1\text{ K}$) | $\frac{\|x_i - x_j\|}{R_f}$ |
| `Rotational speed [rpm]` | Float | **Numeric Ratio** | Min: $1168$, Max: $2886$ ($R_f = 1718\text{ rpm}$) | $\frac{\|x_i - x_j\|}{R_f}$ |
| `Torque [Nm]` | Float | **Numeric Ratio** | Min: $3.8$, Max: $76.6$ ($R_f = 72.8\text{ Nm}$) | $\frac{\|x_i - x_j\|}{R_f}$ |
| `Tool wear [min]` | Float | **Numeric Ratio** | Min: $0.0$, Max: $253.0$ ($R_f = 253.0\text{ min}$) | $\frac{\|x_i - x_j\|}{R_f}$ |

### Target Leakage Prevention
To guarantee zero target leakage:
- **Excluded Identifiers:** `UDI`, `Product ID` (non-predictive row indices).
- **Excluded Failure-Mode Flags:** `Machine failure`, `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), and `RNF` (Random Failure).
- *Rationale:* Since machine condition is derived from breakdown state, including failure flags as predictor features would allow the classifier to cheat (target leakage).
- *Operational Binary Flag:* `Operational_High_Load` is strictly an operating regime flag ($\text{Torque} \ge 40.0\text{ Nm}$), completely independent of failure modes.
- *Ordinal Attributes:* Supported modularly by Gower distance ($z = \frac{r-1}{M-1}$), validated in unit tests with ordered states, but `Type` is kept nominal by default to avoid ungrounded ordinal assumptions.

---

## 3. Label Engineering: 3-Tier Operational Health

The raw AI4I 2020 dataset provides only binary `Machine failure` ($0$ or $1$). The 3-tier condition classes are engineered using Stephan Matzka's documented failure physics:

### Rules & Overlap Precedence
1. **Critical (Precedence 1 - Highest):**
   $$\text{Machine failure} == 1$$
   *Count:* **339 records (3.39%)**.  
   *Rationale:* A physical breakdown occurred. It takes absolute precedence regardless of other sensor readings.
2. **Warning (Precedence 2):**
   $$\text{Machine failure} == 0 \quad \text{AND} \quad \Big(\text{Wear} \ge 180.0\text{ min} \lor \Delta T \le 9.0\text{ K} \lor P \notin [3800.0, 8500.0]\text{ W}\Big)$$
   *Count:* **2,997 records (29.97%)**.  
   *Rationale:* Machine is operating near physical breakdown regimes (Tool Wear failure regime begins ~200-240 min, Heat Dissipation failure regime at $\Delta T < 8.6\text{ K}$, Power failure regime at $<3500\text{ W}$ or $>9000\text{ W}$). Evaluated only when no breakdown has occurred.
3. **Normal (Precedence 3 - Baseline):**
   $$\text{Machine failure} == 0 \quad \text{AND safe nominal boundaries}$$
   *Count:* **6,664 records (66.64%)**.  
   *Rationale:* Healthy operational baseline.

### Methodological Limitations
- These classes are **heuristic rule-engineered labels**, NOT real-world telemetry ground truth.
- Thresholds are hard cutoffs (e.g. 179.9 min vs 180.0 min).
- Evaluation measures classifier agreement with these physical rules, rather than validated foresight on unobserved factory machines.

---

## 4. Mathematical Formulation

### 4.1 Gower's Proximity Metric
For records $\mathbf{x}_i, \mathbf{x}_j$ over $p$ attributes:
$$d(i, j) = \frac{\sum_{f=1}^p w_f \delta_{ij}^{(f)} d_{ij}^{(f)}}{\sum_{f=1}^p w_f \delta_{ij}^{(f)}}, \quad S(i, j) = 1 - d(i, j)$$
Where:
- $\delta_{ij}^{(f)} = 1.0$ if both $x_{if}, x_{jf}$ are present and non-missing; $\delta_{ij}^{(f)} = 0.0$ if either is `None`, empty, or `NaN`.
- Zero-range features ($R_f \le 0$) safely return $d_f = 0.0, \delta_f = 1.0$.
- Query records are excluded from their own neighbor pool (`exclude_self=True`).

### 4.2 Distance-Weighted $k$-NN Classifier
For query $\mathbf{x}_q$ and its $k$ closest neighbors $\mathcal{N}_k(q)$:
$$w_n = \frac{1}{d(q, n) + 10^{-5}}, \quad \hat{y}_q = \arg\max_{c \in \{\text{Normal, Warning, Critical}\}} \sum_{n \in \mathcal{N}_k(q), y_n = c} w_n$$

---

## 5. Empirical Evaluation Results

Evaluated on a stratified held-out test split ($k = 5$ neighbors) with normalization ranges fitted **strictly on the training split**:

### 5.1 Verified Performance
- **Test Sample Size:** 298 instances
- **Overall Accuracy:** **88.59%**

### 5.2 Confusion Matrix
| Actual \ Predicted | Normal | Warning | Critical | Total Actual |
|---|:---:|:---:|:---:|:---:|
| **Normal** | **188** | 11 | 0 | 199 |
| **Warning** | 14 | **74** | 1 | 89 |
| **Critical** | 1 | 7 | **2** | 10 |
| **Total Predicted** | 203 | 92 | 3 | 298 |

### 5.3 Per-Class Classification Report
| Condition Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
|---|:---:|:---:|:---:|:---:|
| **Normal** | 92.61% | 94.47% | **93.53%** | 199 |
| **Warning** | 80.43% | 83.15% | **81.77%** | 89 |
| **Critical** | 66.67% | 20.00% | **30.77%** | 10 |

---

## 6. Sample Similarity Search Demonstration

### Query Scenario: Severely Worn Machine (Tool Wear Warning Regime)
`Type=L, AirTemp=302.5 K, ProcTemp=311.2 K, Speed=1350 rpm, Torque=58.0 Nm, ToolWear=215 min, Operational_High_Load=1`

### Top-5 Retrieved Similar Machines
| Rank | UDI | Product ID | Type | Gower Similarity | Condition | Dominant Difference Factors |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | 5735 | L52914 | L | **97.72%** | Critical | Process temp diff = 0.07, Tool wear diff = 0.03 |
| **2** | 3529 | L50708 | L | **97.33%** | Critical | Torque diff = 0.07, Air temp diff = 0.05 |
| **3** | 3945 | L51124 | L | **97.14%** | Warning | Torque diff = 0.12, Air temp diff = 0.03 |
| **4** | 3941 | L51120 | L | **97.08%** | Warning | Tool wear diff = 0.07, Torque diff = 0.05 |
| **5** | 3940 | L51119 | L | **97.02%** | Warning | Tool wear diff = 0.08, Torque diff = 0.05 |

**Decision Output:** `Warning: 55.79%`, `Critical: 44.21%`, `Normal: 0.00%`  
**Diagnosis:** **WARNING** (High wear approaching tool exhaustion boundary).

---

## 7. Viva & Lab Report Discussion Points
1. **Target Leakage Remediation:** By purging all failure indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), the classifier relies solely on physical operating parameters.
2. **Preprocessing Integrity:** Numeric ranges ($R_f$) fitted exclusively on training data prevent optimistic bias from test data leakage.
3. **Transparent Interpretability:** Distance breakdown highlights which sensor readings caused similarity or divergence.
4. **Rare Event Realism:** The low recall on Critical (20.0%) realistically reflects the severe class imbalance (3.4% base rate) when binary failure shortcuts are eliminated.
