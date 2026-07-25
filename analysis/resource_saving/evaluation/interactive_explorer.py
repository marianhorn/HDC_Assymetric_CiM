#!/usr/bin/env python3
"""Small Streamlit explorer for sparse resource-saving sweep results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "extracted" / "combined.parquet"
WEIGHTED_DATASET = "weighted_over_datasets"
DATASET_ORDER = ["dataset_0", "dataset_1", "dataset_2", "dataset_3", WEIGHTED_DATASET, "overall"]
PHASE_ORDER = ["preopt", "postopt"]
SPLIT_ORDER = ["validation", "test"]
METRIC_DEFAULT = "overall_accuracy"
LOWER_IS_BETTER_METRICS = {"duration_sec", "not_correct", "transition_error"}
PROOF_CHARTS = {
    "Reference Comparison",
    "Resource Saving Frontier",
    "Missing Run Advisor",
    "Pattern Dominance",
}
CONFIG_COLUMNS = [
    "num_features",
    "binning_mode",
    "bipolar_mode",
    "precomputed_item_memory",
    "use_genetic_item_memory",
    "ga_selection_mode",
    "ga_mutation_rate",
    "n_gram_size",
    "window",
    "downsample",
    "validation_ratio",
]
PROOF_GROUP_BASE_COLUMNS = ["variant", "num_levels", "vector_dimension"]
PROOF_LABEL_COLUMNS = [
    "variant",
    "num_levels",
    "vector_dimension",
    "num_features",
    "binning_mode_name",
    "bipolar_mode",
    "precomputed_item_memory",
    "use_genetic_item_memory",
    "n_gram_size",
    "window",
    "downsample",
]


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in ["num_levels", "vector_dimension", "seed"]:
        frame[column] = frame[column].astype(int)
    return add_weighted_dataset_rows(frame)


def add_weighted_dataset_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a weighted-across-datasets view using sum(correct) / sum(total)."""
    if "correct" not in frame.columns or "total" not in frame.columns:
        return frame

    if WEIGHTED_DATASET in set(frame["dataset"].dropna().unique()):
        return frame

    dataset_rows = frame[
        (frame["scope"] == "dataset") & frame["dataset"].astype(str).str.startswith("dataset_")
    ].copy()
    if dataset_rows.empty:
        return frame

    metric_columns = [
        column
        for column in [
            "overall_accuracy",
            "class_average_accuracy",
            "class_vector_similarity",
            "correct",
            "not_correct",
            "transition_error",
            "total",
            "duration_sec",
        ]
        if column in frame.columns
    ]
    group_columns = [
        column
        for column in frame.columns
        if column not in metric_columns and column not in {"scope", "dataset", "info"}
    ]

    weighted = dataset_rows.groupby(group_columns, dropna=False, as_index=False).agg(
        correct=("correct", "sum"),
        not_correct=("not_correct", "sum"),
        transition_error=("transition_error", "sum"),
        total=("total", "sum"),
        overall_accuracy=("overall_accuracy", "mean"),
        class_average_accuracy=("class_average_accuracy", "mean"),
        class_vector_similarity=("class_vector_similarity", "mean"),
        duration_sec=("duration_sec", "max"),
    )
    weighted["overall_accuracy"] = weighted["correct"] / weighted["total"]
    weighted["scope"] = "weighted"
    weighted["dataset"] = WEIGHTED_DATASET
    weighted["info"] = (
        "scope=weighted,phase="
        + weighted["phase"].astype(str)
        + "-"
        + weighted["split"].astype(str)
    )
    weighted = weighted[frame.columns]
    return pd.concat([frame, weighted], ignore_index=True)


def ordered_values(values, preferred_order=None):
    present = list(pd.Series(values).dropna().unique())
    if preferred_order:
        ordered = [value for value in preferred_order if value in present]
        ordered.extend(sorted(value for value in present if value not in ordered))
        return ordered
    return sorted(present)


def aggregate_frame(frame: pd.DataFrame, metric: str, seed_mode: str) -> pd.DataFrame:
    group_columns = ["variant", "num_levels", "vector_dimension"]
    if seed_mode == "mean":
        return frame.groupby(group_columns, as_index=False)[metric].mean()
    if seed_mode == "std":
        return frame.groupby(group_columns, as_index=False)[metric].std()
    if seed_mode == "min":
        return frame.groupby(group_columns, as_index=False)[metric].min()
    if seed_mode == "max":
        return frame.groupby(group_columns, as_index=False)[metric].max()
    return frame[group_columns + [metric]].copy()


def proof_group_columns(frame: pd.DataFrame) -> list[str]:
    return PROOF_GROUP_BASE_COLUMNS + [column for column in CONFIG_COLUMNS if column in frame.columns]


def proof_label(row: pd.Series) -> str:
    parts = []
    for column in PROOF_LABEL_COLUMNS:
        if column in row.index:
            parts.append(f"{column}={row[column]}")
    return " | ".join(parts)


def better_is_higher(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER_METRICS


def summarize_proof_groups(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_columns = proof_group_columns(frame)
    if frame.empty:
        return pd.DataFrame(columns=group_columns)

    summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            mean_metric=(metric, "mean"),
            std_metric=(metric, "std"),
            min_metric=(metric, "min"),
            max_metric=(metric, "max"),
            rows=(metric, "size"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary["label"] = summary.apply(proof_label, axis=1)
    return summary


def select_reference_group(frame: pd.DataFrame, metric: str, key_prefix: str) -> tuple[pd.Series | None, pd.DataFrame]:
    st.subheader("Reference")
    variants = ordered_values(frame["variant"])
    if not variants:
        st.warning("No reference data available for the selected phase/split/dataset.")
        return None, pd.DataFrame()

    reference_variant = st.selectbox("Reference variant", variants, key=f"{key_prefix}_reference_variant")
    reference_frame = frame[frame["variant"] == reference_variant].copy()
    reference_summary = summarize_proof_groups(reference_frame, metric)
    if reference_summary.empty:
        st.warning("No reference configurations are available.")
        return None, reference_summary

    dimensions = sorted(reference_summary["vector_dimension"].dropna().unique().astype(int))
    reference_dimension = st.selectbox(
        "Reference vector dimension",
        dimensions,
        key=f"{key_prefix}_reference_dimension",
    )
    level_frame = reference_summary[reference_summary["vector_dimension"] == reference_dimension]
    levels = sorted(level_frame["num_levels"].dropna().unique().astype(int))
    reference_levels = st.selectbox(
        "Reference num levels",
        levels,
        key=f"{key_prefix}_reference_levels",
    )

    selected = level_frame[level_frame["num_levels"] == reference_levels].copy()
    if selected.empty:
        st.warning("No reference row matches the selected dimension and level count.")
        return None, reference_summary

    if len(selected) > 1:
        st.warning(
            "The selected variant/dimension/level maps to multiple full configurations. "
            "Using the best metric row; check that the remaining config fields are truly constant."
        )

    selected = selected.sort_values(
        "mean_metric",
        ascending=not better_is_higher(metric),
    ).reset_index(drop=True)
    reference_row = selected.iloc[0]
    with st.expander("Selected full reference configuration"):
        st.dataframe(
            selected[[column for column in PROOF_LABEL_COLUMNS + ["mean_metric", "seeds", "rows"] if column in selected.columns]],
            width="stretch",
            hide_index=True,
        )
    st.caption(
        f"Reference mean {metric}: {reference_row['mean_metric']:.6g} "
        f"from {int(reference_row['seeds'])} seed(s), {int(reference_row['rows'])} row(s)."
    )
    return reference_row, reference_summary


def rows_for_group(frame: pd.DataFrame, group_row: pd.Series) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column in proof_group_columns(frame):
        value = group_row[column]
        if pd.isna(value):
            mask &= frame[column].isna()
        else:
            mask &= frame[column] == value
    return frame[mask].copy()


def paired_candidate_summary(
    frame: pd.DataFrame,
    reference_row: pd.Series,
    candidate_variants: list[str],
    metric: str,
    tolerance: float,
) -> pd.DataFrame:
    group_columns = proof_group_columns(frame)
    reference_rows = rows_for_group(frame, reference_row)
    reference_by_seed = reference_rows.groupby("seed", as_index=True)[metric].mean().rename("reference")
    expected_seeds = set(reference_by_seed.index)

    candidate_frame = frame[frame["variant"].isin(candidate_variants)].copy()
    summaries = []
    for key, candidate_rows in candidate_frame.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        candidate_record = dict(zip(group_columns, key))
        candidate_by_seed = candidate_rows.groupby("seed", as_index=True)[metric].mean().rename("candidate")
        paired = pd.concat([reference_by_seed, candidate_by_seed], axis=1, join="inner").dropna()
        present_seeds = set(candidate_by_seed.index)
        missing_seeds = sorted(expected_seeds - present_seeds)
        n = len(paired)
        if n == 0:
            continue

        delta = paired["candidate"] - paired["reference"]
        delta_std = float(delta.std()) if n > 1 else 0.0
        ci_radius = 1.96 * delta_std / (n**0.5) if n > 1 else 0.0
        delta_mean = float(delta.mean())
        delta_min = float(delta.min())
        candidate_mean = float(candidate_by_seed.mean())
        reference_mean = float(reference_by_seed.mean())
        resource_saving = 1.0 - (
            float(candidate_record["vector_dimension"]) / float(reference_row["vector_dimension"])
        )

        if better_is_higher(metric):
            pass_mean = delta_mean >= -tolerance
            pass_worst = delta_min >= -tolerance
            margin_to_threshold = delta_mean + tolerance
        else:
            pass_mean = delta_mean <= tolerance
            pass_worst = float(delta.max()) <= tolerance
            margin_to_threshold = tolerance - delta_mean

        complete = not missing_seeds
        if complete and pass_mean and pass_worst:
            status = "passes worst-seed"
        elif complete and pass_mean:
            status = "passes mean"
        elif pass_mean:
            status = "unproven missing seeds"
        else:
            status = "fails"

        record = candidate_record
        record.update(
            {
                "mean_metric": float(candidate_rows[metric].mean()),
                "std_metric": float(candidate_rows[metric].std()) if len(candidate_rows) > 1 else 0.0,
                "min_metric": float(candidate_rows[metric].min()),
                "max_metric": float(candidate_rows[metric].max()),
                "rows": len(candidate_rows),
                "seeds": int(candidate_rows["seed"].nunique()),
                "label": proof_label(pd.Series(candidate_record)),
                "reference_mean": reference_mean,
                "candidate_mean": candidate_mean,
                "delta_mean": delta_mean,
                "delta_min": delta_min,
                "delta_std": delta_std,
                "delta_ci95_low": delta_mean - ci_radius,
                "delta_ci95_high": delta_mean + ci_radius,
                "paired_seeds": n,
                "missing_seeds": ", ".join(str(seed) for seed in missing_seeds),
                "complete": complete,
                "resource_saving": resource_saving,
                "pass_mean": pass_mean,
                "pass_worst_seed": pass_worst,
                "margin_to_threshold": margin_to_threshold,
                "status": status,
            }
        )
        summaries.append(record)

    if not summaries:
        return pd.DataFrame()
    result = pd.DataFrame(summaries)
    status_order = {
        "passes worst-seed": 0,
        "passes mean": 1,
        "unproven missing seeds": 2,
        "fails": 3,
    }
    result["_status_order"] = result["status"].map(status_order).fillna(99)
    result = result.sort_values(
        ["_status_order", "resource_saving", "delta_mean"],
        ascending=[True, False, not better_is_higher(metric)],
    ).drop(columns=["_status_order"])
    return result


def show_reference_comparison(frame: pd.DataFrame, metric: str, selected_variants: list[str]) -> None:
    reference_row, _ = select_reference_group(frame, metric, "reference_comparison")
    if reference_row is None:
        return

    tolerance_pp = st.slider(
        "Allowed metric loss / increase in percentage points",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        help="For accuracy metrics this is allowed loss. For lower-is-better metrics this is allowed increase.",
    )
    tolerance = tolerance_pp / 100.0

    available_variants = ordered_values(frame["variant"])
    default_candidates = [variant for variant in selected_variants if variant != reference_row["variant"]]
    if not default_candidates:
        default_candidates = [variant for variant in available_variants if variant != reference_row["variant"]]
    candidate_variants = st.multiselect(
        "Candidate variants",
        available_variants,
        default=default_candidates,
        key="reference_comparison_candidates",
    )
    if not candidate_variants:
        st.warning("Select at least one candidate variant.")
        return

    comparison = paired_candidate_summary(frame, reference_row, candidate_variants, metric, tolerance)
    if comparison.empty:
        st.warning("No candidate rows can be paired with the selected reference seeds.")
        return

    passing = comparison[comparison["pass_mean"] & comparison["complete"]].copy()
    if not passing.empty:
        best = passing.sort_values(
            ["vector_dimension", "resource_saving", "delta_mean"],
            ascending=[True, False, not better_is_higher(metric)],
        ).iloc[0]
        st.success(
            "Smallest complete passing candidate: "
            f"{best['variant']} at D={int(best['vector_dimension'])}, "
            f"L={int(best['num_levels'])}, "
            f"mean delta={best['delta_mean'] * 100.0:+.3f} pp, "
            f"resource saving={best['resource_saving'] * 100.0:.1f}%."
        )
    else:
        st.warning("No complete candidate passes the selected non-inferiority threshold.")

    display_columns = [
        "status",
        "variant",
        "num_levels",
        "vector_dimension",
        "candidate_mean",
        "reference_mean",
        "delta_mean",
        "delta_min",
        "delta_ci95_low",
        "delta_ci95_high",
        "resource_saving",
        "paired_seeds",
        "missing_seeds",
        "binning_mode_name",
        "num_features",
        "n_gram_size",
        "window",
        "downsample",
        "bipolar_mode",
        "precomputed_item_memory",
        "use_genetic_item_memory",
    ]
    display_columns = [column for column in display_columns if column in comparison.columns]
    st.dataframe(comparison[display_columns], width="stretch", hide_index=True)


def pattern_delta_frame(
    frame: pd.DataFrame,
    reference_variant: str,
    candidate_variant: str,
    metric: str,
) -> pd.DataFrame:
    index_columns = ["num_levels", "vector_dimension", "seed"]
    reference = (
        frame[frame["variant"] == reference_variant]
        .groupby(index_columns, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "reference_metric"})
    )
    candidate = (
        frame[frame["variant"] == candidate_variant]
        .groupby(index_columns, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "candidate_metric"})
    )
    paired = reference.merge(candidate, on=index_columns, how="inner")
    if paired.empty:
        return paired

    if better_is_higher(metric):
        paired["delta"] = paired["candidate_metric"] - paired["reference_metric"]
    else:
        paired["delta"] = paired["reference_metric"] - paired["candidate_metric"]
    paired["delta_pp"] = paired["delta"] * 100.0
    return paired


def show_pattern_dominance(frame: pd.DataFrame, metric: str, selected_variants: list[str]) -> None:
    st.subheader("Pattern Dominance")
    st.caption(
        "Compares variants over every shared `(num_levels, vector_dimension, seed)` point. "
        "Positive delta means the candidate is better than the reference for the selected metric."
    )

    variants = ordered_values(frame["variant"])
    if len(variants) < 2:
        st.warning("Need at least two variants for pattern dominance.")
        return

    reference_variant = st.selectbox(
        "Reference variant",
        variants,
        key="pattern_reference_variant",
    )
    default_candidates = [variant for variant in selected_variants if variant != reference_variant]
    if not default_candidates:
        default_candidates = [variant for variant in variants if variant != reference_variant]
    candidate_variants = st.multiselect(
        "Candidate variants",
        variants,
        default=default_candidates,
        key="pattern_candidate_variants",
    )
    candidate_variants = [variant for variant in candidate_variants if variant != reference_variant]
    if not candidate_variants:
        st.warning("Select at least one candidate variant different from the reference.")
        return

    tolerance_pp = st.slider(
        "Allowed loss / increase in percentage points",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        key="pattern_tolerance",
        help="0.0 tests strict non-worse behavior over the shared grid.",
    )

    reference_points = (
        frame[frame["variant"] == reference_variant][["num_levels", "vector_dimension"]]
        .drop_duplicates()
        .shape[0]
    )
    summaries = []
    detail_frames: dict[str, pd.DataFrame] = {}
    for candidate_variant in candidate_variants:
        paired = pattern_delta_frame(frame, reference_variant, candidate_variant, metric)
        if paired.empty:
            continue
        point_summary = (
            paired.groupby(["num_levels", "vector_dimension"], as_index=False)
            .agg(
                reference_mean=("reference_metric", "mean"),
                candidate_mean=("candidate_metric", "mean"),
                mean_delta_pp=("delta_pp", "mean"),
                worst_seed_delta_pp=("delta_pp", "min"),
                std_delta_pp=("delta_pp", "std"),
                paired_seeds=("seed", "nunique"),
            )
        )
        point_summary["passes_mean"] = point_summary["mean_delta_pp"] >= -tolerance_pp
        point_summary["passes_every_seed"] = point_summary["worst_seed_delta_pp"] >= -tolerance_pp
        detail_frames[candidate_variant] = point_summary

        compared_points = len(point_summary)
        compared_seed_pairs = len(paired)
        mean_passes = int(point_summary["passes_mean"].sum())
        every_seed_passes = int(point_summary["passes_every_seed"].sum())
        if compared_points and every_seed_passes == compared_points:
            status = "dominates every shared point/seed"
        elif compared_points and mean_passes == compared_points:
            status = "dominates every shared point mean"
        else:
            status = "has failures"

        candidate_points = (
            frame[frame["variant"] == candidate_variant][["num_levels", "vector_dimension"]]
            .drop_duplicates()
            .shape[0]
        )
        summaries.append(
            {
                "status": status,
                "candidate_variant": candidate_variant,
                "reference_variant": reference_variant,
                "shared_points": compared_points,
                "reference_points": reference_points,
                "candidate_points": candidate_points,
                "missing_reference_points": max(reference_points - compared_points, 0),
                "paired_seed_rows": compared_seed_pairs,
                "points_passing_mean": mean_passes,
                "points_passing_every_seed": every_seed_passes,
                "mean_pass_rate": mean_passes / compared_points if compared_points else 0.0,
                "every_seed_pass_rate": every_seed_passes / compared_points if compared_points else 0.0,
                "mean_delta_pp": float(point_summary["mean_delta_pp"].mean()),
                "worst_mean_delta_pp": float(point_summary["mean_delta_pp"].min()),
                "worst_seed_delta_pp": float(point_summary["worst_seed_delta_pp"].min()),
            }
        )

    if not summaries:
        st.warning("No shared grid points between the selected reference and candidates.")
        return

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["every_seed_pass_rate", "mean_pass_rate", "worst_seed_delta_pp"],
        ascending=[False, False, False],
    )
    st.dataframe(summary_frame, width="stretch", hide_index=True)

    detail_variant = st.selectbox(
        "Candidate detail",
        summary_frame["candidate_variant"].tolist(),
        key="pattern_detail_variant",
    )
    details = detail_frames[detail_variant].copy()
    failing = details[
        (~details["passes_mean"]) | (~details["passes_every_seed"])
    ].sort_values(["worst_seed_delta_pp", "mean_delta_pp"])

    left, right = st.columns(2)
    with left:
        st.metric("Shared grid points", len(details))
    with right:
        st.metric("Failing grid points", len(failing))

    pivot = details.pivot_table(
        index="num_levels",
        columns="vector_dimension",
        values="mean_delta_pp",
        aggfunc="mean",
    )
    if not pivot.empty:
        max_abs_delta = max(abs(float(details["mean_delta_pp"].min())), abs(float(details["mean_delta_pp"].max())))
        if max_abs_delta == 0.0:
            max_abs_delta = 1.0
        fig = px.imshow(
            pivot.sort_index(ascending=False),
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-max_abs_delta,
            zmax=max_abs_delta,
            labels={
                "x": "vector_dimension",
                "y": "num_levels",
                "color": f"{metric} delta [pp]",
            },
            title=f"{detail_variant} minus {reference_variant}: mean delta over seeds",
        )
        st.plotly_chart(fig, width="stretch")

    if failing.empty:
        st.success("No failing shared grid points for the selected tolerance.")
    else:
        st.subheader("Failing Grid Points")
        st.dataframe(
            failing[
                [
                    "num_levels",
                    "vector_dimension",
                    "reference_mean",
                    "candidate_mean",
                    "mean_delta_pp",
                    "worst_seed_delta_pp",
                    "std_delta_pp",
                    "paired_seeds",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


def show_resource_saving_frontier(frame: pd.DataFrame, metric: str, selected_variants: list[str]) -> None:
    reference_row, _ = select_reference_group(frame, metric, "frontier")
    if reference_row is None:
        return

    objective = st.selectbox(
        "Frontier objective",
        [
            "Save dimensions at fixed num_levels",
            "Save levels at fixed vector_dimension",
        ],
        key="frontier_objective",
    )
    if objective == "Save dimensions at fixed num_levels":
        fixed_column = "num_levels"
        fixed_value = int(reference_row["num_levels"])
        x_column = "vector_dimension"
        saving_column = "dimension_saving"
        smallest_label = "Smallest Passing Dimension Per Variant"
        no_pass_label = "No dimension passes the threshold at this fixed num_levels."
        x_label = "vector_dimension"
    else:
        fixed_column = "vector_dimension"
        fixed_value = int(reference_row["vector_dimension"])
        x_column = "num_levels"
        saving_column = "level_saving"
        smallest_label = "Smallest Passing Level Count Per Variant"
        no_pass_label = "No level count passes the threshold at this fixed vector_dimension."
        x_label = "num_levels"

    st.caption(
        f"Fixed `{fixed_column}` = {fixed_value}. "
        f"The frontier varies only `{x_column}` and does not mix the fixed axis."
    )

    tolerance_pp = st.slider(
        "Allowed metric loss / increase in percentage points",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        key="frontier_tolerance",
    )
    tolerance = tolerance_pp / 100.0

    candidate_variants = st.multiselect(
        "Frontier variants",
        ordered_values(frame["variant"]),
        default=selected_variants,
        key="frontier_variants",
    )
    if not candidate_variants:
        st.warning("Select at least one variant.")
        return

    summary = summarize_proof_groups(frame[frame["variant"].isin(candidate_variants)], metric)
    if summary.empty:
        st.warning("No rows match the frontier selection.")
        return

    summary = summary[pd.to_numeric(summary[fixed_column], errors="coerce") == fixed_value].copy()
    if summary.empty:
        st.warning(f"No rows match `{fixed_column}` = {fixed_value}.")
        return

    ascending = not better_is_higher(metric)
    best_per_step = (
        summary.sort_values(["variant", x_column, "mean_metric"], ascending=[True, True, ascending])
        .groupby(["variant", x_column], as_index=False)
        .first()
    )
    reference_x = float(reference_row[x_column])
    best_per_step[saving_column] = 1.0 - (best_per_step[x_column].astype(float) / reference_x)
    reference_mean = float(reference_row["mean_metric"])
    threshold = reference_mean - tolerance if better_is_higher(metric) else reference_mean + tolerance
    if better_is_higher(metric):
        best_per_step["passes_threshold"] = best_per_step["mean_metric"] >= threshold
    else:
        best_per_step["passes_threshold"] = best_per_step["mean_metric"] <= threshold

    fig = px.line(
        best_per_step,
        x=x_column,
        y="mean_metric",
        color="variant",
        markers=True,
        hover_data=[
            column
            for column in [
                "num_levels",
                "vector_dimension",
                "mean_metric",
                "std_metric",
                "passes_threshold",
                saving_column,
                "binning_mode_name",
                "n_gram_size",
                "window",
                "downsample",
            ]
            if column in best_per_step.columns
        ],
        title=f"Best observed {metric} vs {x_label} at fixed {fixed_column}={fixed_value}",
    )
    fig.add_hline(y=reference_mean, line_dash="solid", annotation_text="reference")
    fig.add_hline(y=threshold, line_dash="dash", annotation_text="threshold")
    st.plotly_chart(fig, width="stretch")

    passing = best_per_step[best_per_step["passes_threshold"]].copy()
    if passing.empty:
        st.warning(no_pass_label)
    else:
        smallest = (
            passing.sort_values(["variant", x_column])
            .groupby("variant", as_index=False)
            .first()
        )
        st.subheader(smallest_label)
        display_columns = [
            "variant",
            "vector_dimension",
            "num_levels",
            "mean_metric",
            "std_metric",
            saving_column,
            "binning_mode_name",
            "num_features",
            "n_gram_size",
            "window",
            "downsample",
            "bipolar_mode",
            "precomputed_item_memory",
            "use_genetic_item_memory",
        ]
        display_columns = [column for column in display_columns if column in smallest.columns]
        st.dataframe(smallest[display_columns], width="stretch", hide_index=True)


def show_missing_run_advisor(frame: pd.DataFrame, metric: str, selected_variants: list[str]) -> None:
    st.subheader("Coverage")
    expected_seed_values = sorted(frame["seed"].dropna().unique().astype(int))
    expected_seeds = set(
        st.multiselect(
            "Expected seeds",
            expected_seed_values,
            default=expected_seed_values,
            key="advisor_expected_seeds",
        )
    )
    if not expected_seeds:
        st.warning("Select at least one expected seed.")
        return

    candidate_frame = frame[frame["variant"].isin(selected_variants)].copy()
    if candidate_frame.empty:
        st.warning("No rows match the selected variants.")
        return

    group_columns = proof_group_columns(candidate_frame)
    coverage = (
        candidate_frame.groupby(group_columns, dropna=False)
        .agg(
            present_seed_count=("seed", "nunique"),
            mean_metric=(metric, "mean"),
            std_metric=(metric, "std"),
        )
        .reset_index()
    )
    seed_sets = candidate_frame.groupby(group_columns, dropna=False)["seed"].apply(
        lambda values: set(int(value) for value in values.dropna().unique())
    )
    coverage["present_seeds"] = [", ".join(str(seed) for seed in sorted(seed_sets.iloc[i])) for i in range(len(seed_sets))]
    coverage["missing_seeds"] = [
        ", ".join(str(seed) for seed in sorted(expected_seeds - seed_sets.iloc[i]))
        for i in range(len(seed_sets))
    ]
    coverage["coverage"] = coverage["present_seed_count"] / max(len(expected_seeds), 1)
    incomplete = coverage[coverage["missing_seeds"] != ""].copy()

    st.metric("Configuration groups", len(coverage))
    st.metric("Incomplete groups", len(incomplete))

    if incomplete.empty:
        st.success("No missing seeds for the selected phase/split/dataset/variants.")
    else:
        st.warning("Some configuration groups are missing expected seeds.")
        display_columns = [
            "variant",
            "num_levels",
            "vector_dimension",
            "present_seed_count",
            "present_seeds",
            "missing_seeds",
            "mean_metric",
            "std_metric",
            "binning_mode_name",
            "n_gram_size",
            "window",
            "downsample",
        ]
        display_columns = [column for column in display_columns if column in incomplete.columns]
        st.dataframe(
            incomplete.sort_values(["coverage", "variant", "vector_dimension"])[display_columns],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Boundary Runs")
    reference_row, _ = select_reference_group(frame, metric, "advisor")
    if reference_row is None:
        return
    tolerance_pp = st.slider(
        "Boundary tolerance in percentage points",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
        key="advisor_tolerance",
    )
    comparison = paired_candidate_summary(
        frame,
        reference_row,
        [variant for variant in selected_variants if variant != reference_row["variant"]],
        metric,
        tolerance_pp / 100.0,
    )
    if comparison.empty:
        st.info("No candidate groups can be paired with the selected reference.")
        return

    boundary = comparison.sort_values("margin_to_threshold", ascending=False).copy()
    boundary["recommendation"] = "lower priority"
    boundary.loc[boundary["status"] == "unproven missing seeds", "recommendation"] = "complete missing seeds"
    boundary.loc[
        (boundary["status"] == "fails") & (boundary["margin_to_threshold"] > -0.01),
        "recommendation",
    ] = "near threshold; run neighboring dimensions/seeds"
    boundary.loc[
        (boundary["status"].str.startswith("passes")) & (boundary["delta_std"] > 0.005),
        "recommendation",
    ] = "passes but variance is high; consider more seeds"

    display_columns = [
        "recommendation",
        "status",
        "variant",
        "num_levels",
        "vector_dimension",
        "candidate_mean",
        "delta_mean",
        "delta_std",
        "margin_to_threshold",
        "resource_saving",
        "paired_seeds",
        "missing_seeds",
        "binning_mode_name",
        "n_gram_size",
        "window",
        "downsample",
    ]
    display_columns = [column for column in display_columns if column in boundary.columns]
    st.dataframe(boundary.head(100)[display_columns], width="stretch", hide_index=True)


def metric_range(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = frame[metric].dropna()
    if values.empty:
        return 0.0, 1.0
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        padding = abs(min_value) * 0.05 if min_value != 0.0 else 1.0
        return min_value - padding, max_value + padding
    return min_value, max_value


def axis_ranges(frame: pd.DataFrame) -> tuple[list[int], list[int]]:
    levels = sorted(frame["num_levels"].dropna().unique().astype(int))
    dimensions = sorted(frame["vector_dimension"].dropna().unique().astype(int))
    return levels, dimensions


def make_heatmap(
    frame: pd.DataFrame,
    metric: str,
    variant: str,
    levels: list[int],
    dimensions: list[int],
    color_range: tuple[float, float],
):
    variant_frame = frame[frame["variant"] == variant]
    if variant_frame.empty:
        return None

    heatmap = variant_frame.pivot_table(
        index="num_levels",
        columns="vector_dimension",
        values=metric,
        aggfunc="mean",
    ).reindex(index=list(reversed(levels)), columns=dimensions)

    if heatmap.empty:
        return None

    return px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale="Viridis",
        zmin=color_range[0],
        zmax=color_range[1],
        labels={"x": "vector_dimension", "y": "num_levels", "color": metric},
        title=variant,
    )


def make_dimension_line(
    frame: pd.DataFrame,
    metric: str,
    selected_level: int,
    dimensions: list[int],
    value_range: tuple[float, float],
):
    line_frame = frame[frame["num_levels"] == selected_level].sort_values("vector_dimension")
    if line_frame.empty:
        return None
    fig = px.line(
        line_frame,
        x="vector_dimension",
        y=metric,
        color="variant",
        markers=True,
        title=f"{metric} vs vector_dimension at NUM_LEVELS={selected_level}",
    )
    fig.update_xaxes(range=[min(dimensions), max(dimensions)])
    fig.update_yaxes(range=list(value_range))
    return fig


def make_level_line(
    frame: pd.DataFrame,
    metric: str,
    selected_dimension: int,
    levels: list[int],
    value_range: tuple[float, float],
):
    line_frame = frame[frame["vector_dimension"] == selected_dimension].sort_values("num_levels")
    if line_frame.empty:
        return None
    fig = px.line(
        line_frame,
        x="num_levels",
        y=metric,
        color="variant",
        markers=True,
        title=f"{metric} vs num_levels at VECTOR_DIMENSION={selected_dimension}",
    )
    fig.update_xaxes(range=[min(levels), max(levels)])
    fig.update_yaxes(range=list(value_range))
    return fig


def plot_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch")


def show_heatmaps(
    frame: pd.DataFrame,
    metric: str,
    selected_variants: list[str],
    levels: list[int],
    dimensions: list[int],
    value_range: tuple[float, float],
) -> None:
    st.subheader("Heatmaps")
    st.caption(
        f"Shared heatmap color scale: {metric} from {value_range[0]:.6g} to {value_range[1]:.6g}."
    )
    columns_per_row = st.slider("Heatmaps per row", 1, 4, 2)
    for start in range(0, len(selected_variants), columns_per_row):
        cols = st.columns(columns_per_row)
        for col, variant in zip(cols, selected_variants[start : start + columns_per_row]):
            fig = make_heatmap(frame, metric, variant, levels, dimensions, value_range)
            with col:
                if fig is None:
                    st.info(f"No data for {variant}")
                else:
                    plot_chart(fig)


def show_slices(
    frame: pd.DataFrame,
    metric: str,
    levels: list[int],
    dimensions: list[int],
    value_range: tuple[float, float],
) -> None:
    st.subheader("1D Slices")
    st.caption(
        f"Shared y-axis: {metric} from {value_range[0]:.6g} to {value_range[1]:.6g}."
    )
    left, right = st.columns(2)
    with left:
        selected_level = st.select_slider("Fixed NUM_LEVELS", options=levels, value=levels[len(levels) // 2])
        fig = make_dimension_line(frame, metric, selected_level, dimensions, value_range)
        if fig:
            plot_chart(fig)
    with right:
        selected_dimension = st.select_slider(
            "Fixed VECTOR_DIMENSION",
            options=dimensions,
            value=dimensions[len(dimensions) // 2],
        )
        fig = make_level_line(frame, metric, selected_dimension, levels, value_range)
        if fig:
            plot_chart(fig)


def show_ranking(frame: pd.DataFrame, metric: str) -> None:
    st.subheader("Best Configurations")
    lower_is_better = metric in LOWER_IS_BETTER_METRICS
    top_n = st.slider("Rows", 5, 100, 25)
    ranking = frame.sort_values(metric, ascending=lower_is_better).head(top_n)
    st.dataframe(ranking, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Resource Saving Explorer", layout="wide")
    st.title("Resource Saving Explorer")

    data_path = st.sidebar.text_input("Parquet file", str(DEFAULT_DATA_PATH))
    path = Path(data_path)
    if not path.exists():
        st.error(f"Missing data file: {path}")
        st.info("Run extract_resource_saving.py first.")
        return

    data = load_data(str(path))

    metric_columns = [
        column
        for column in [
            "overall_accuracy",
            "class_average_accuracy",
            "class_vector_similarity",
            "correct",
            "not_correct",
            "transition_error",
            "total",
            "duration_sec",
        ]
        if column in data.columns
    ]

    metric = st.sidebar.selectbox(
        "Metric",
        metric_columns,
        index=metric_columns.index(METRIC_DEFAULT) if METRIC_DEFAULT in metric_columns else 0,
    )
    chart_type = st.sidebar.selectbox(
        "Chart type",
        [
            "Heatmaps",
            "Slices",
            "Best Configs",
            "Data",
            "Reference Comparison",
            "Resource Saving Frontier",
            "Missing Run Advisor",
            "Pattern Dominance",
        ],
    )
    split = st.sidebar.selectbox("Split", ordered_values(data["split"], SPLIT_ORDER), index=0)
    phase = st.sidebar.selectbox("Phase", ordered_values(data["phase"], PHASE_ORDER), index=0)
    dataset = st.sidebar.selectbox("Dataset", ordered_values(data["dataset"], DATASET_ORDER), index=0)

    seed_values = sorted(data["seed"].dropna().unique().astype(int))
    seed_choice = st.sidebar.selectbox(
        "Seed handling",
        ["mean", "std", "min", "max"] + [f"seed_{seed}" for seed in seed_values],
    )

    variants = ordered_values(data["variant"])
    selected_variants = st.sidebar.multiselect("Variants", variants, default=variants)
    if not selected_variants:
        st.warning("Select at least one variant.")
        return

    filtered = data[
        (data["split"] == split)
        & (data["phase"] == phase)
        & (data["dataset"] == dataset)
        & (data["variant"].isin(selected_variants))
    ].copy()

    if seed_choice.startswith("seed_"):
        selected_seed = int(seed_choice.removeprefix("seed_"))
        filtered = filtered[filtered["seed"] == selected_seed]
        seed_mode = "single"
    else:
        seed_mode = seed_choice

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return

    if chart_type in PROOF_CHARTS:
        st.caption(
            "Proof views compare complete configurations against a selected reference. "
            f"`{WEIGHTED_DATASET}` uses sum(correct) / sum(total) across dataset rows."
        )
        if chart_type == "Reference Comparison":
            show_reference_comparison(filtered, metric, selected_variants)
        elif chart_type == "Resource Saving Frontier":
            show_resource_saving_frontier(filtered, metric, selected_variants)
        elif chart_type == "Missing Run Advisor":
            show_missing_run_advisor(filtered, metric, selected_variants)
        else:
            show_pattern_dominance(filtered, metric, selected_variants)
        return

    aggregated = aggregate_frame(filtered, metric, seed_mode)
    levels, dimensions = axis_ranges(aggregated)
    value_range = metric_range(aggregated, metric)

    st.caption(
        f"Rows after filtering: {len(filtered):,}. "
        "Missing configurations stay empty; no interpolation or zero-fill is applied."
    )

    if chart_type == "Heatmaps":
        show_heatmaps(aggregated, metric, selected_variants, levels, dimensions, value_range)
    elif chart_type == "Slices":
        show_slices(aggregated, metric, levels, dimensions, value_range)
    elif chart_type == "Best Configs":
        show_ranking(aggregated, metric)
    else:
        st.subheader("Filtered Long Table")
        st.dataframe(filtered, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
