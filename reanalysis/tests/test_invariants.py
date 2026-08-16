"""Invariant tests for the reanalysis deliverables.

Two groups:

1. Stage-1 deliverable invariants, checked directly on the committed per-run CSVs
   (`human_runs_derived*.csv`) — run identity, the δ range, and the property the whole
   δ reconstruction rests on: gmean(minutes_derived over successes) == published
   `human_minutes`.
2. Fit reproduction — that `metr_fit_helpers` still reproduces METR's own checked-in
   p50/p80 values, which every Stage-2/3 scenario is measured against.

The analysis scripts themselves are jupytext notebooks that execute on import, so
nothing here imports them; `metr_fit_helpers` is a plain module and is imported.

Run: .venv/bin/python -m pytest reanalysis/tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

REANALYSIS = Path(__file__).resolve().parent.parent
STAGE_1_DATA = REANALYSIS / "stage-1-export-archaeology" / "data"
STAGE_2 = REANALYSIS / "stage-2-measurement-error"

VERSION_SUFFIXES = ["", "_v1_1"]   # v1.0 (paper suite) and v1.1
DERIVATION_CATEGORIES = {
    "delta_corrected",         # HCAST/RE-Bench run with a task-level δ solved for
    "delta_imputed_failure",   # failed run, δ borrowed from the task's successes
    "swaa_exact",              # SWAA: exact seconds, no flooring to undo
    "uncorrected_rebench",     # RE-Bench: published human_minutes is not a gmean
    "uncorrected_no_delta",    # no δ solvable (single-run tasks, estimates)
}


@pytest.fixture(scope="module", params=VERSION_SUFFIXES, ids=["v1_0", "v1_1"])
def human_runs_derived(request) -> pd.DataFrame:
    return pd.read_csv(STAGE_1_DATA / f"human_runs_derived{request.param}.csv")


# --- group 1: Stage-1 deliverable invariants ---------------------------------------

def test_run_id_present_non_null_and_unique(human_runs_derived: pd.DataFrame) -> None:
    assert "run_id" in human_runs_derived.columns
    assert human_runs_derived["run_id"].notna().all()
    assert human_runs_derived["run_id"].is_unique


def test_delta_task_in_unit_interval(human_runs_derived: pd.DataFrame) -> None:
    """δ is a fractional part recovered from flooring, so 0 <= δ < 1 where present."""
    delta_task = human_runs_derived["delta_task"].dropna()
    assert len(delta_task) > 0
    assert (delta_task >= 0).all()
    assert (delta_task < 1).all()


def test_derivation_categories_are_the_documented_set(
    human_runs_derived: pd.DataFrame,
) -> None:
    assert set(human_runs_derived["derivation"]) == DERIVATION_CATEGORIES


def test_delta_corrected_gmean_reproduces_human_minutes(
    human_runs_derived: pd.DataFrame,
) -> None:
    """On every HCAST task whose successful runs are all δ-corrected, the geometric mean
    of the derived per-run times must return the published human_minutes exactly."""
    successful_hcast_baseline_runs = human_runs_derived.query(
        "score_binarized == 1 and human_source == 'baseline' and task_source == 'HCAST'"
    )
    fully_delta_corrected_runs = successful_hcast_baseline_runs.groupby("task_id").filter(
        lambda task_runs: (task_runs["derivation"] == "delta_corrected").all()
    )
    reconstruction_by_task = fully_delta_corrected_runs.groupby("task_id").agg(
        gmean_minutes_derived=("minutes_derived", stats.gmean),
        human_minutes=("human_minutes", "first"),
    )
    assert len(reconstruction_by_task) > 50   # v1.0: 81 tasks, v1.1: 94
    relative_error = (
        reconstruction_by_task["gmean_minutes_derived"]
        / reconstruction_by_task["human_minutes"] - 1
    ).abs()
    assert relative_error.max() < 1e-9, (
        f"worst task: {relative_error.idxmax()} at {relative_error.max():.2e}"
    )


# --- group 2: fit reproduction ------------------------------------------------------

@pytest.fixture(scope="module")
def metr_helpers():
    """`metr_fit_helpers` lives beside the Stage-2 scripts and is importable on its own
    (unlike the analysis scripts, which execute on import by design)."""
    sys.path.insert(0, str(STAGE_2))
    import metr_fit_helpers

    return metr_fit_helpers


@pytest.mark.parametrize("quantile,official_column", [(0.5, "p50"), (0.8, "p80")])
def test_helper_reproduces_official_fits(
    metr_helpers, quantile: float, official_column: str
) -> None:
    """The shared fit machinery must reproduce METR's checked-in headline horizons on
    the unperturbed x-axis — that is what makes every scenario ratio meaningful.
    v1.0 only; the helper reads DATASET_VERSION, which defaults to 1-0."""
    runs = metr_helpers.load_runs()
    official_human_minutes_by_task = metr_helpers.official_human_minutes_by_task(runs)
    horizon_by_agent = metr_helpers.fit_horizons_per_quantile_by_agent(
        runs, official_human_minutes_by_task, quantiles=(quantile,)
    )[quantile]

    _, _, official_fits_by_agent = metr_helpers.load_frontier_agents_and_dates()
    official_horizon_by_agent = official_fits_by_agent.set_index("agent")[official_column]

    our_horizon = pd.Series(horizon_by_agent)
    official_horizon = official_horizon_by_agent.reindex(our_horizon.index)
    assert official_horizon.notna().all()
    assert len(our_horizon) > 30   # v1.0 has 34 agents in the headline fits

    relative_error = (our_horizon / official_horizon - 1).abs()
    assert relative_error.max() < 1e-4, (
        f"worst agent: {relative_error.idxmax()} at {relative_error.max():.2e}"
    )


def test_frontier_agents_are_a_subset_of_the_fitted_agents(metr_helpers) -> None:
    frontier_agents, _, official_fits_by_agent = (
        metr_helpers.load_frontier_agents_and_dates()
    )
    assert len(frontier_agents) > 0
    assert set(frontier_agents) <= set(official_fits_by_agent["agent"])
    assert np.isfinite(official_fits_by_agent["p50"]).all()
