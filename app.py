"""Streamlit dashboard for CNC machine similarity search and condition classification.

Builds directly on the existing algorithm modules (src.dataset, src.gower, src.knn):
no distance or classification logic is reimplemented here.

Run from the repository root:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.dataset import (
    BINARY_COLUMNS,
    CONDITION_CLASSES,
    HIGH_LOAD_TORQUE_THRESHOLD,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    CNCDataset,
    load_dataset,
    make_query_record,
    train_test_split,
)
from src.knn import GowerKNNClassifier

DATA_PATH = Path(__file__).parent / "data" / "ai4i2020.csv"
FEATURE_COLUMNS = NOMINAL_COLUMNS + NUMERIC_COLUMNS + BINARY_COLUMNS
CONDITION_COLORS = {"Normal": "#2e7d32", "Warning": "#f9a825", "Critical": "#c62828"}

TEAM = [
    ("Atharva Khandagale", "124A1061"),
    ("Surya Kaushik", "124A1055"),
    ("Prakash Biradar", "124A1013"),
    ("Rakesh Biradar", "124A1014"),
]

# The three canonical scenarios used by `python -m src.pipeline --demo`
PRESETS: Dict[str, Dict[str, Any]] = {
    "Custom": {},
    "Healthy baseline": dict(machine_type="M", air_temp=298.2, process_temp=308.7, speed_rpm=1500.0, torque_nm=40.0, tool_wear=25.0),
    "Severely worn tool": dict(machine_type="L", air_temp=302.5, process_temp=311.2, speed_rpm=1350.0, torque_nm=58.0, tool_wear=215.0),
    "Narrow thermal margin + heavy load": dict(machine_type="H", air_temp=304.1, process_temp=312.0, speed_rpm=1280.0, torque_nm=68.5, tool_wear=195.0),
}


# ---------------------------------------------------------------------------
# Cached data and models
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI4I 2020 dataset...")
def get_dataset() -> CNCDataset:
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner="Fitting Gower k-NN classifier...")
def get_classifier(k: int) -> GowerKNNClassifier:
    # Same configuration as the CLI demo/query mode: fitted on all records,
    # numeric ranges taken from the dataset the classifier was fitted on.
    return GowerKNNClassifier(k=k, weighting="distance").fit(get_dataset())


@st.cache_data(show_spinner=False)
def get_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(get_dataset().records)
    df = df.rename(columns={"udi": "UDI", "product_id": "Product ID", "condition": "Condition", "machine_failure": "Machine failure"})
    return df[["UDI", "Product ID"] + FEATURE_COLUMNS + ["Condition", "Machine failure"]]


@st.cache_data(show_spinner=False)
def run_evaluation(k: int, sample_size: int) -> Dict[str, Any]:
    # Mirrors src.pipeline.run_evaluation: stratified split, ranges fitted on train only.
    train_data, test_data = train_test_split(get_dataset(), test_ratio=0.2, seed=42)
    if 0 < sample_size < len(test_data):
        test_data, _ = train_test_split(test_data, test_ratio=1.0 - (sample_size / len(test_data)), seed=42)
    clf = GowerKNNClassifier(k=k, weighting="distance").fit(train_data)
    result = clf.evaluate(test_data, k=k)
    result["train_size"] = len(train_data)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def neighbors_to_frame(neighbors: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rank, n in enumerate(neighbors, 1):
        rec = n["record"]
        row = {
            "Rank": rank,
            "UDI": rec.get("udi"),
            "Product ID": rec.get("product_id"),
            "Condition": n["condition"],
            "Gower similarity": round(n["similarity"], 4),
            "Gower distance": round(n["distance"], 4),
        }
        if "vote_weight" in n:
            row["Vote weight"] = round(n["vote_weight"], 2)
        row.update({col: rec.get(col) for col in FEATURE_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def breakdown_frame(neighbors: List[Dict[str, Any]]) -> pd.DataFrame:
    """Per-attribute Gower component distances d_f (rows = neighbours)."""
    data = {f"#{i} (UDI {n['record'].get('udi')})": n["breakdown"] for i, n in enumerate(neighbors, 1)}
    return pd.DataFrame(data).T


def style_condition(val: Any) -> str:
    color = CONDITION_COLORS.get(val)
    return f"color: {color}; font-weight: 600" if color else ""


def show_condition_badge(condition: str) -> None:
    color = CONDITION_COLORS[condition]
    st.markdown(
        f"<div style='padding:0.8rem 1rem;border-radius:0.5rem;background:{color};color:white;"
        f"font-size:1.4rem;font-weight:700;text-align:center'>Predicted condition: {condition.upper()}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(df: pd.DataFrame) -> None:
    st.header("Overview")
    counts = df["Condition"].value_counts().reindex(CONDITION_CLASSES, fill_value=0)

    cols = st.columns(4)
    cols[0].metric("Machines", f"{len(df):,}")
    for col, cls in zip(cols[1:], CONDITION_CLASSES):
        col.metric(cls, f"{counts[cls]:,}", f"{counts[cls] / len(df) * 100:.2f}%", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Condition distribution")
        st.bar_chart(counts.rename("Machines"), color="#455a64")
    with right:
        st.subheader("Machine type by condition")
        type_cond = pd.crosstab(df["Type"], df["Condition"]).reindex(columns=CONDITION_CLASSES, fill_value=0)
        st.bar_chart(type_cond, color=[CONDITION_COLORS[c] for c in CONDITION_CLASSES])

    st.subheader("Operating envelope: torque vs rotational speed")
    st.scatter_chart(
        df.sort_values("Condition", key=lambda s: s.map(CONDITION_CLASSES.index)),
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="Condition",
        size=12,
    )

    st.subheader("Tool wear distribution by condition")
    bins = pd.cut(df["Tool wear [min]"], bins=range(0, 270, 20), right=False)
    wear = pd.crosstab(bins.astype(str), df["Condition"]).reindex(columns=CONDITION_CLASSES, fill_value=0)
    wear.index.name = "Tool wear bin [min]"
    st.bar_chart(wear, color=[CONDITION_COLORS[c] for c in CONDITION_CLASSES])

    with st.expander("How the Normal / Warning / Critical labels are derived", expanded=False):
        st.markdown(
            """
The AI4I 2020 dataset only provides a binary `Machine failure` flag. The three condition classes are
**engineered labels** (not original AI4I labels), assigned with this precedence:

1. **Critical** — `Machine failure == 1` (an actual simulated breakdown).
2. **Warning** — no failure, and tool wear ≥ 180 min, **or** process − air temperature ≤ 9.0 K,
   **or** mechanical power outside [3800, 8500] W.
3. **Normal** — no failure and none of the warning criteria met.

Because Warning is derived from the same sensor attributes the classifier uses, evaluation accuracy measures
agreement with these rules, not validated real-world failure prediction.
"""
        )
    st.caption(
        "Dataset: S. Matzka, *AI4I 2020 Predictive Maintenance Dataset*, UCI Machine Learning Repository — "
        "a **synthetic** dataset generated from a physical simulation of a milling machine."
    )


def page_explorer(df: pd.DataFrame) -> None:
    st.header("Dataset Explorer")
    c1, c2, c3 = st.columns(3)
    types = c1.multiselect("Machine type", ["L", "M", "H"], default=["L", "M", "H"])
    conds = c2.multiselect("Condition", CONDITION_CLASSES, default=CONDITION_CLASSES)
    wear_min, wear_max = int(df["Tool wear [min]"].min()), int(df["Tool wear [min]"].max())
    wear_range = c3.slider("Tool wear [min]", wear_min, wear_max, (wear_min, wear_max))

    filtered = df[
        df["Type"].isin(types)
        & df["Condition"].isin(conds)
        & df["Tool wear [min]"].between(*wear_range)
    ]
    st.write(f"Showing **{len(filtered):,}** of {len(df):,} machines.")
    if filtered.empty:
        st.info("No machines match the selected filters.")
        return
    st.dataframe(filtered.style.map(style_condition, subset=["Condition"]), use_container_width=True, hide_index=True, height=480)
    st.caption(
        "`Machine failure` is shown for audit only. It (and UDI, Product ID, TWF, HDF, PWF, OSF, RNF) is excluded "
        "from the predictor features to prevent target leakage."
    )

    st.subheader("Summary statistics (numeric predictors)")
    st.dataframe(filtered[NUMERIC_COLUMNS].describe().T.round(2), use_container_width=True)


def page_similarity(df: pd.DataFrame, k: int) -> None:
    st.header("Machine Similarity Search")
    st.write("Select a machine to find its most similar machines using Gower's mixed-attribute distance.")

    dataset = get_dataset()
    by_udi = {rec["udi"]: rec for rec in dataset.records}

    c1, c2 = st.columns([2, 1])
    udi = c1.number_input("Machine UDI", min_value=int(df["UDI"].min()), max_value=int(df["UDI"].max()), value=1, step=1)
    top_n = c2.slider("Number of similar machines", 1, 10, 3)

    query = by_udi.get(int(udi))
    if query is None:
        st.error(f"No machine with UDI {udi} found.")
        return

    st.subheader("Selected machine")
    st.dataframe(df[df["UDI"] == int(udi)].style.map(style_condition, subset=["Condition"]), use_container_width=True, hide_index=True)

    neighbors = get_classifier(k).find_k_nearest(query, k=top_n, exclude_self=True)
    if not neighbors:
        st.warning("No neighbours could be computed for this machine.")
        return

    st.subheader(f"Top {top_n} most similar machines")
    st.dataframe(neighbors_to_frame(neighbors).style.map(style_condition, subset=["Condition"]), use_container_width=True, hide_index=True)
    st.caption("The selected machine itself is excluded from its own neighbour list (exclude_self=True).")

    st.subheader("Per-attribute Gower component distances d_f")
    st.write("0 = identical on that attribute, 1 = maximally different (numeric differences divided by the attribute range).")
    bd = breakdown_frame(neighbors)
    st.dataframe(bd.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(bd.T)


def page_prediction(k: int) -> None:
    st.header("Condition Prediction")
    st.write("Enter machine operating characteristics to classify them as Normal, Warning or Critical with the Gower k-NN classifier.")

    dataset = get_dataset()
    preset_name = st.selectbox("Load a scenario", list(PRESETS.keys()), index=1)
    preset = PRESETS[preset_name] or PRESETS["Healthy baseline"]

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        machine_type = c1.selectbox("Machine type (nominal)", ["L", "M", "H"], index=["L", "M", "H"].index(preset["machine_type"]))
        air_temp = c2.number_input("Air temperature [K]", min_value=0.0, value=preset["air_temp"], step=0.1, format="%.1f")
        process_temp = c3.number_input("Process temperature [K]", min_value=0.0, value=preset["process_temp"], step=0.1, format="%.1f")
        c4, c5, c6 = st.columns(3)
        speed = c4.number_input("Rotational speed [rpm]", min_value=0.0, value=preset["speed_rpm"], step=10.0, format="%.0f")
        torque = c5.number_input("Torque [Nm]", min_value=0.0, value=preset["torque_nm"], step=0.5, format="%.1f")
        wear = c6.number_input("Tool wear [min]", min_value=0.0, value=preset["tool_wear"], step=1.0, format="%.0f")
        load_choice = st.radio(
            "Operational high-load flag (binary)",
            ["Auto (derive from torque)", "0", "1"],
            horizontal=True,
            help=f"Auto sets the flag to 1 when torque ≥ {HIGH_LOAD_TORQUE_THRESHOLD:g} Nm, matching how the dataset feature is built.",
        )
        submitted = st.form_submit_button("Predict condition", type="primary")

    if not submitted:
        st.info("Adjust the inputs and click **Predict condition**.")
        return

    try:
        query = make_query_record(
            machine_type, air_temp, process_temp, speed, torque, wear,
            high_load=None if load_choice.startswith("Auto") else int(load_choice),
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return

    # Warn when inputs fall outside the fitted ranges (Gower clips component distances at 1)
    out_of_range = [
        f"{col} = {query[col]:g} (training range {lo:g}–{hi:g})"
        for col, (lo, hi, _) in dataset.ranges.items()
        if not lo <= query[col] <= hi
    ]
    if out_of_range:
        st.warning("Some inputs are outside the observed data range; similarity on these attributes is capped:\n\n- " + "\n- ".join(out_of_range))

    try:
        result = get_classifier(k).predict(query, k=k, exclude_self=True)
    except ValueError as exc:
        st.error(f"Prediction unavailable: {exc}")
        return

    show_condition_badge(result["predicted_condition"])

    left, right = st.columns(2)
    with left:
        st.subheader("Class probabilities")
        probs = pd.Series(result["probabilities"]).reindex(CONDITION_CLASSES).rename("Probability")
        st.bar_chart(probs, color="#455a64")
    with right:
        st.subheader("Voting result")
        votes = pd.DataFrame(
            {
                "Weighted score": pd.Series(result["class_scores"]).reindex(CONDITION_CLASSES).round(2),
                "Neighbours": pd.Series([n["condition"] for n in result["neighbors"]]).value_counts().reindex(CONDITION_CLASSES, fill_value=0),
                "Probability": probs.map(lambda p: f"{p * 100:.2f}%"),
            }
        )
        st.dataframe(votes, use_container_width=True)
        st.caption("Distance-weighted vote: w = 1 / (d + 1e-5). Ties are broken in the order Normal, Warning, Critical.")

    st.subheader(f"Nearest {k} neighbours")
    st.dataframe(neighbors_to_frame(result["neighbors"]).style.map(style_condition, subset=["Condition"]), use_container_width=True, hide_index=True)
    with st.expander("Per-attribute Gower component distances"):
        st.dataframe(breakdown_frame(result["neighbors"]).style.format("{:.3f}"), use_container_width=True)


def page_evaluation(k: int) -> None:
    st.header("Model Evaluation")
    st.write(
        "Stratified 80/20 train/test split (seed 42). Numeric normalization ranges are fitted **only on the training split**, "
        "and the label rules and k were fixed in advance — the test set is not used for tuning."
    )
    sample = st.select_slider("Test sample size", options=[100, 300, 500, 1000, 0], value=300, format_func=lambda v: "Full test split (slow)" if v == 0 else str(v))
    st.caption("The pure-Python Gower distance compares each test machine against all ~8,000 training machines, so larger samples take longer.")

    if not st.button("Run evaluation", type="primary"):
        return

    with st.spinner("Evaluating classifier..."):
        res = run_evaluation(k, sample)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{res['accuracy'] * 100:.2f}%")
    c2.metric("Test instances", res["total_tested"])
    c3.metric("Training instances", res["train_size"])

    st.subheader("Confusion matrix (rows = actual, columns = predicted)")
    cm = pd.DataFrame(res["confusion_matrix"]).T.reindex(index=CONDITION_CLASSES, columns=CONDITION_CLASSES)
    st.dataframe(cm, use_container_width=True)

    st.subheader("Per-class metrics")
    metrics = pd.DataFrame(res["per_class_metrics"]).T.reindex(CONDITION_CLASSES)
    for col in ("precision", "recall", "f1_score"):
        metrics[col] = (metrics[col] * 100).round(2)
    metrics["support"] = metrics["support"].astype(int)
    st.dataframe(metrics.rename(columns={"precision": "Precision (%)", "recall": "Recall (%)", "f1_score": "F1 (%)", "support": "Support"}), use_container_width=True)

    st.warning(
        "Interpretation: Warning/Normal labels are rule-derived from the same sensor attributes used as features, so accuracy "
        "mostly measures how well k-NN reconstructs those rule boundaries. Critical (real breakdowns, 3.4% of data) is the "
        "only class not defined by the features, and its recall is low; with small test samples the Critical estimate is also "
        "based on very few cases."
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="CNC Machine Similarity", page_icon="⚙️", layout="wide")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Make sure `data/ai4i2020.csv` is present and run from the repository root.")
        st.stop()

    with st.sidebar:
        st.title("⚙️ CNC Machine Similarity")
        st.caption("Gower distance + k-NN condition classifier")
        page = st.radio(
            "Page",
            ["Overview", "Dataset Explorer", "Machine Similarity", "Condition Prediction", "Model Evaluation"],
        )
        k = st.slider("k (neighbours for classification)", 1, 15, 5, help="Default k=5 matches the CLI and the reported evaluation.")
        st.divider()
        st.markdown("**DWM Lab — Group 4**  \nThird Year (TE), Computer Engineering")
        st.markdown("\n".join(f"- {name} — `{roll}`" for name, roll in TEAM))

    try:
        df = get_dataframe()
    except Exception as exc:  # malformed CSV etc.
        st.error(f"Could not load the dataset: {exc}")
        st.stop()

    if page == "Overview":
        page_overview(df)
    elif page == "Dataset Explorer":
        page_explorer(df)
    elif page == "Machine Similarity":
        page_similarity(df, k)
    elif page == "Condition Prediction":
        page_prediction(k)
    else:
        page_evaluation(k)


main()
