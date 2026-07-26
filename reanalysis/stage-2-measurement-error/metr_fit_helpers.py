"""Shared fit primitives for the Stage 2 measurement-error analyses (a)/(c)/(d).

Thin wrappers around METR's own logistic fit and frontier / doubling-time
construction, so every sub-analysis perturbs the x-axis and then runs the
*identical* downstream pipeline. Imported by the analysis notebooks; not itself
a notebook.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from horizon.utils.logistic import get_x_for_quantile, logistic_regression
from horizon.wrangle.out_of_sample_extrapolation import get_frontier_agents

HERE = Path(__file__).parent
REPO = HERE / ".." / ".."
REPORT = REPO / "reports" / "time-horizon-1-0"

REGULARIZATION = 1e-5          # headline value from reports/time-horizon-1-0/fig_params
WEIGHT_COLUMN = "invsqrt_task_weight"
DEFAULT_QUANTILES = (0.5, 0.8)


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
    frontier_agents = get_frontier_agents(official_fits_by_agent, release_dates, 50)
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
