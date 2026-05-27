#!/usr/bin/env python3
"""Small Streamlit explorer for sparse resource-saving sweep results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "extracted" / "combined.parquet"
DATASET_ORDER = ["dataset_0", "dataset_1", "dataset_2", "dataset_3", "overall"]
PHASE_ORDER = ["preopt", "postopt"]
SPLIT_ORDER = ["validation", "test"]
METRIC_DEFAULT = "overall_accuracy"
LOWER_IS_BETTER_METRICS = {"duration_sec", "not_correct", "transition_error"}


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in ["num_levels", "vector_dimension", "seed"]:
        frame[column] = frame[column].astype(int)
    return frame


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
        ["Heatmaps", "Slices", "Best Configs", "Data"],
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
