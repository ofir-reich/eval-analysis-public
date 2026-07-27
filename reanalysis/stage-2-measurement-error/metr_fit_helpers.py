"""Shared fit primitives for the Stage 2 measurement-error analyses (a)/(c)/(d).

Thin wrappers around METR's own logistic fit and frontier / doubling-time
construction, so every sub-analysis perturbs the x-axis and then runs the
*identical* downstream pipeline. Imported by the analysis notebooks; not itself
a notebook.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import yaml

from horizon.utils.logistic import get_x_for_quantile, logistic_regression
from horizon.wrangle.out_of_sample_extrapolation import get_frontier_agents

HERE = Path(__file__).parent
REPO = HERE / ".." / ".."

# Which public suite to analyse: "1-0" (paper's suite, default) or "1-1" (§7 replication).
# Set DATASET_VERSION=1-1 in the environment to run the whole Stage-2 chain on v1.1.
DATASET_VERSION = os.environ.get("DATASET_VERSION", "1-0")
# suffix appended to every input/output filename so v1.0 and v1.1 artifacts coexist
# ("" for the default v1.0, "_v1_1" for v1.1). Matches the Stage-1 derived-CSV naming.
VERSION_SUFFIX = "" if DATASET_VERSION == "1-0" else "_v" + DATASET_VERSION.replace("-", "_")
# short tag for figure titles (blank for v1.0 to keep the headline figures unchanged)
VERSION_TITLE_TAG = "" if DATASET_VERSION == "1-0" else f" · v{DATASET_VERSION}"
REPORT = REPO / "reports" / f"time-horizon-{DATASET_VERSION}"

REGULARIZATION = 1e-5          # headline value from reports/time-horizon-*/fig_params
WEIGHT_COLUMN = "invsqrt_task_weight"
DEFAULT_QUANTILES = (0.5, 0.8)
# Round floats on CSV write so re-running a script is byte-stable: BLAS summation order
# varies run to run and perturbs the last couple of digits, which otherwise shows up as
# spurious diffs in every committed data file.
CSV_FLOAT_FORMAT = "%.10g"


def derived_human_runs_csv() -> Path:
    """Path to the Stage-1 δ-corrected per-run dataset for the active version."""
    return (
        HERE / ".." / "stage-1-export-archaeology" / "data"
        / f"human_runs_derived{VERSION_SUFFIX}.csv"
    )


def load_runs() -> pd.DataFrame:
    runs = pd.read_json(REPORT / "data" / "raw" / "runs.jsonl", lines=True)
    return runs.rename(columns={"alias": "agent"})


def official_human_minutes_by_task(runs: pd.DataFrame) -> pd.Series:
    return runs.groupby("task_id")["human_minutes"].first()


def fit_horizons_per_quantile_by_agent(
    runs_sample: pd.DataFrame,
    human_minutes_by_task: pd.Series,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[float, dict[str, float]]:
    """METR's weighted logistic of success on log2(human_minutes), per agent.
    Returns {quantile: {agent: horizon_minutes}}."""
    log2_minutes = np.log2(
        runs_sample["task_id"].map(human_minutes_by_task).to_numpy()
    )
    horizon_per_quantile_by_agent = {quantile: {} for quantile in quantiles}
    for agent, indices in runs_sample.groupby("agent").indices.items():
        is_success = runs_sample["score_binarized"].to_numpy()[indices]
        if len(np.unique(is_success)) < 2:
            continue
        model = logistic_regression(
            log2_minutes[indices].reshape(-1, 1), is_success,
            sample_weight=runs_sample[WEIGHT_COLUMN].to_numpy()[indices],
            regularization=REGULARIZATION,
            ensure_weights_sum_to_1=False,
        )
        for quantile in quantiles:
            horizon_per_quantile_by_agent[quantile][agent] = float(
                np.exp2(get_x_for_quantile(model, quantile))
            )
    return horizon_per_quantile_by_agent


def load_frontier_agents_and_dates() -> tuple[list[str], dict, pd.DataFrame]:
    release_dates = yaml.safe_load(
        (REPO / "data" / "external" / "release_dates.yaml").read_text()
    )
    official_fits_by_agent = pd.read_csv(
        REPORT / "data" / "wrangled" / "logistic_fits" / "headline.csv"
    )
    # METR returns a set; sort it so downstream row order (and hence every CSV we write)
    # is reproducible — set iteration order varies between processes under hash
    # randomisation. Nothing downstream depends on the ordering itself.
    frontier_agents = sorted(get_frontier_agents(official_fits_by_agent, release_dates, 50))
    return frontier_agents, release_dates, official_fits_by_agent


def years_since_first_release_by_agent(
    frontier_agents: list[str], release_dates: dict
) -> dict[str, float]:
    release_date_by_agent = {
        agent: pd.Timestamp(date) for agent, date in release_dates["date"].items()
        if agent in frontier_agents and date is not None
    }
    first_release_date = min(release_date_by_agent.values())
    return {
        agent: (date - first_release_date).days / 365.25
        for agent, date in release_date_by_agent.items()
    }


def fit_doubling_days(
    horizon_by_agent: dict[str, float] | pd.Series,
    years_since_release_by_agent: dict[str, float],
) -> float:
    """OLS of log2(horizon) on years-since-first-release over frontier agents;
    doubling time in days = 365.25 / slope. NaN if <5 agents have finite horizons."""
    horizon_by_agent = dict(horizon_by_agent)
    points = [
        (years, np.log2(horizon_by_agent[agent]))
        for agent, years in years_since_release_by_agent.items()
        if np.isfinite(horizon_by_agent.get(agent, np.nan))
    ]
    if len(points) < 5:
        return np.nan
    years, log2_horizon = map(np.array, zip(*points))
    slope = np.polyfit(years, log2_horizon, 1)[0]
    return 365.25 / slope


def dodged_confidence_interval_figure(
    ci_by_agent_condition: pd.DataFrame,
    agents_sorted: list[str],
    condition_order: list[str],
    title: str,
    row_offset: float = 0.26,
) -> "px.scatter":
    """Horizontal interval plot, one row per agent, with the conditions **vertically
    offset** so their intervals sit side by side instead of on top of each other, and
    intervals drawn as bare capless lines (the style METR uses in the paper).

    Expects columns agent / condition / p50_median / ci_lower / ci_upper.
    """
    y_position_by_agent = {agent: index for index, agent in enumerate(agents_sorted)}
    offset_by_condition = {
        condition: (position - (len(condition_order) - 1) / 2) * row_offset
        for position, condition in enumerate(condition_order)
    }
    plot_data = ci_by_agent_condition[
        ci_by_agent_condition["agent"].isin(y_position_by_agent)
    ].copy()
    plot_data["y_position"] = (
        plot_data["agent"].map(y_position_by_agent)
        + plot_data["condition"].map(offset_by_condition)
    )
    plot_data["error_upper"] = plot_data["ci_upper"] - plot_data["p50_median"]
    plot_data["error_lower"] = plot_data["p50_median"] - plot_data["ci_lower"]

    figure = px.scatter(
        plot_data, x="p50_median", y="y_position", color="condition",
        error_x="error_upper", error_x_minus="error_lower", log_x=True,
        category_orders={"condition": condition_order}, hover_name="agent",
        title=title,
        labels={"p50_median": "50%-success time horizon (minutes)", "y_position": ""},
    )
    figure.update_traces(marker=dict(size=6), error_x=dict(width=0, thickness=1.6))
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(y_position_by_agent.values()),
        ticktext=list(y_position_by_agent.keys()),
        range=[-0.75, len(agents_sorted) - 0.25],
    )
    figure.update_layout(legend=dict(orientation="h", y=1.04, x=0))
    return figure
