# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stage 2(a) — Parametric x-bootstrap with per-task σ
#
# [(a0)](a0_nonparametric_xboot.py) resampled *which baseline runs* enter each gmean —
# assumption-free, but it could only perturb the 126 tasks with ≥2 successful runs, and
# **froze the rest** (21 single-baseline tasks, researcher estimates, RE-Bench), which
# are disproportionately the long, frontier-driving tasks. That made it a *lower bound*.
#
# Here we lift the bound with the per-task σ from [(b)](b_sigma_task.py): in each
# replicate we draw every task's `human_minutes` from
# LogNormal(ln `human_minutes`, SEM²_task), with **SEM_task = σ̃_task / √n** for
# baselined tasks and METR's σ = 1.05 for researcher estimates — so *every* task carries
# x-uncertainty, including the frozen ones. Same three conditions and METR fit code as
# (a0), so the two are directly comparable.
#
# | condition | agent-run resampling (METR's f→t→r) | parametric x draw |
# |---|---|---|
# | `metr` | ✔ | – |
# | `metr+x_param` | ✔ | ✔ (all tasks) |
# | `x_param_only` | – | ✔ (all tasks) |

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from joblib import Parallel, delayed

from horizon.wrangle.bootstrap import bootstrap_sample
import metr_fit_helpers as metr

plotly.io.templates.default = "plotly_white"
pd.set_option("display.max_columns", None, "display.width", 200)

HERE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
FIGURES = HERE / "figures"
DATA_OUT = HERE / "data"
FIGURES.mkdir(exist_ok=True), DATA_OUT.mkdir(exist_ok=True)
SUFFIX = metr.VERSION_SUFFIX   # "" for v1.0, "_v1_1" for v1.1
print(f"dataset version: {metr.DATASET_VERSION}")

try:
    get_ipython()  # type: ignore[name-defined]
    IS_INTERACTIVE = True
except NameError:
    IS_INTERACTIVE = False

def show(fig):
    if IS_INTERACTIVE:
        fig.show()

N_BOOT = int(os.environ.get("N_BOOT", 500))
CATEGORIES = ["task_family", "task_id", "run_id"]   # METR's "ftr"

# %%
runs = metr.load_runs()
official_human_minutes_by_task = metr.official_human_minutes_by_task(runs)
sigma_by_task = pd.read_csv(DATA_OUT / f"b_sigma_by_task{SUFFIX}.csv", index_col=0)

log_human_minutes_by_task = np.log(official_human_minutes_by_task)
sem_log_by_task = sigma_by_task["sem_log"].reindex(official_human_minutes_by_task.index)
assert sem_log_by_task.notna().all(), "every task must carry a SEM from (b)"
print(f"{len(runs):,} runs | all {len(sem_log_by_task)} tasks carry x-uncertainty "
      f"(vs 126/170 in a0)")
print("SEM_log quantiles:", sem_log_by_task.quantile([.1, .5, .9]).round(3).to_dict())


def draw_human_minutes(rng: np.random.Generator) -> pd.Series:
    """Parametric x draw: every task's ln(human_minutes) jittered by N(0, SEM²_task)."""
    return np.exp(
        log_human_minutes_by_task + rng.normal(0, sem_log_by_task)
    )


# %% [markdown]
# ## Bootstrap — METR's fit, three conditions (p50 and p80)

# %%
def one_replicate(replicate_index: int, condition: str) -> dict:
    rng = np.random.default_rng(42 + replicate_index)
    runs_sample = (
        bootstrap_sample(runs, CATEGORIES, rng)
        if condition in ("metr", "metr+x_param") else runs
    )
    human_minutes_by_task = (
        draw_human_minutes(rng)
        if condition in ("metr+x_param", "x_param_only")
        else official_human_minutes_by_task
    )
    horizon_per_quantile_by_agent = metr.fit_horizons_per_quantile_by_agent(
        runs_sample, human_minutes_by_task
    )
    # flatten to {f"{agent}@p50": horizon, ...} for a tidy per-replicate row
    return {
        f"{agent}@p{int(quantile * 100)}": horizon
        for quantile, horizon_by_agent in horizon_per_quantile_by_agent.items()
        for agent, horizon in horizon_by_agent.items()
    }


horizons_by_replicate_by_condition = {}
for condition in ["metr", "metr+x_param", "x_param_only"]:
    cache_path = DATA_OUT / f"a_horizons_{condition.replace('+', '_')}{SUFFIX}.csv"
    if cache_path.exists():
        horizons_by_replicate_by_condition[condition] = pd.read_csv(cache_path)
        print(f"{condition}: loaded {len(horizons_by_replicate_by_condition[condition])} cached")
        continue
    replicate_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(one_replicate)(i, condition) for i in range(N_BOOT)
    )
    horizons_by_replicate_by_condition[condition] = pd.DataFrame(replicate_results)
    horizons_by_replicate_by_condition[condition].to_csv(cache_path, index=False)
    print(f"{condition}: {len(horizons_by_replicate_by_condition[condition])} replicates")

# %% [markdown]
# ## Effect on per-agent p50 CIs — parametric (this) vs nonparametric (a0)

# %%
frontier_agents, release_dates, official_fits_by_agent = (
    metr.load_frontier_agents_and_dates()
)
years_since_release_by_agent = metr.years_since_first_release_by_agent(
    frontier_agents, release_dates
)

ci_rows = []
for condition, horizons_by_replicate in horizons_by_replicate_by_condition.items():
    for agent in frontier_agents:
        column = f"{agent}@p50"
        if column not in horizons_by_replicate:
            continue
        p50_draws = horizons_by_replicate[column].dropna()
        p50_draws = p50_draws[np.isfinite(p50_draws)]
        if len(p50_draws) < N_BOOT * 0.9:
            continue
        ci_lower, ci_upper = p50_draws.quantile([0.025, 0.975])
        ci_rows.append({
            "agent": agent, "condition": condition, "p50_median": p50_draws.median(),
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "ci_logwidth": np.log(ci_upper / ci_lower),
        })
ci_by_agent_condition = pd.DataFrame(ci_rows)
ci_by_agent_condition.to_csv(DATA_OUT / f"a_ci_by_agent{SUFFIX}.csv", index=False)

ci_logwidth_per_condition_by_agent = ci_by_agent_condition.pivot(
    index="agent", columns="condition", values="ci_logwidth"
)
ci_logwidth_per_condition_by_agent["widening_pct"] = 100 * (
    ci_logwidth_per_condition_by_agent["metr+x_param"]
    / ci_logwidth_per_condition_by_agent["metr"] - 1
)
# pull in a0's nonparametric widening for the same agents, if present
a0_ci_path = DATA_OUT / f"a0_ci_by_agent{SUFFIX}.csv"
if a0_ci_path.exists():
    a0_ci = pd.read_csv(a0_ci_path)
    a0_logwidth = a0_ci.pivot(index="agent", columns="condition", values="ci_logwidth")
    ci_logwidth_per_condition_by_agent["a0_widening_pct"] = 100 * (
        a0_logwidth["metr+x"] / a0_logwidth["metr"] - 1
    )
print(ci_logwidth_per_condition_by_agent.round(1).sort_values(
    "widening_pct", ascending=False).to_string())
print(f"\nmedian p50 CI widening — parametric (all tasks): "
      f"{ci_logwidth_per_condition_by_agent['widening_pct'].median():.1f}%")
if "a0_widening_pct" in ci_logwidth_per_condition_by_agent:
    print(f"median p50 CI widening — a0 nonparametric (floor): "
          f"{ci_logwidth_per_condition_by_agent['a0_widening_pct'].median():.1f}%")

# %%
agents_sorted_by_official_p50 = (
    official_fits_by_agent.set_index("agent")["p50"].sort_values().index
)
agents_sorted = [
    agent for agent in agents_sorted_by_official_p50 if agent in frontier_agents
]
fig = metr.dodged_confidence_interval_figure(
    ci_by_agent_condition, agents_sorted, ["metr", "metr+x_param", "x_param_only"],
    title="50%-horizon 95% interval — parametric per-task x-noise vs METR baseline",
)
fig.write_image(FIGURES / f"a_ci_comparison{SUFFIX}.png", width=850, height=620, scale=2)
show(fig)

# %% [markdown]
# ## Effect on the doubling time

# %%
def fit_doubling_days_from_row(horizon_row: pd.Series, quantile_label: str) -> float:
    horizon_by_agent = {
        agent: horizon_row.get(f"{agent}@{quantile_label}", np.nan)
        for agent in years_since_release_by_agent
    }
    return metr.fit_doubling_days(horizon_by_agent, years_since_release_by_agent)

doubling_days_per_condition_by_replicate = pd.DataFrame({
    condition: horizons_by_replicate.apply(
        fit_doubling_days_from_row, axis=1, quantile_label="p50")
    for condition, horizons_by_replicate in horizons_by_replicate_by_condition.items()
})
doubling_days_summary_by_condition = (
    doubling_days_per_condition_by_replicate
    .describe(percentiles=[0.025, 0.5, 0.975]).T[["2.5%", "50%", "97.5%"]]
)
doubling_days_summary_by_condition["ci_width_days"] = (
    doubling_days_summary_by_condition["97.5%"] - doubling_days_summary_by_condition["2.5%"]
)
print(doubling_days_summary_by_condition.round(1).to_string())
doubling_days_per_condition_by_replicate.to_csv(
    DATA_OUT / f"a_doubling_times{SUFFIX}.csv", index=False)

fig = px.ecdf(
    doubling_days_per_condition_by_replicate.melt(
        var_name="condition", value_name="doubling_days"),
    x="doubling_days", color="condition",
    title=f"Doubling-time bootstrap distribution — parametric per-task x-noise "
          f"({N_BOOT} reps)",
    labels={"doubling_days": "doubling time (days)"},
)
fig.write_image(FIGURES / f"a_doubling_time{SUFFIX}.png", width=800, height=450, scale=2)
show(fig)

# %% [markdown]
# ## Takeaway
#
# Extending x-noise to *all* tasks parametrically (vs (a0)'s resampleable-only floor)
# raises the per-agent CI widening and the doubling-time CI width — the printed numbers
# above quantify by how much. This is the honest parametric estimate of the independent
# x-axis contribution; the *systematic* counterpart is [(c)](c_coherent_shift.py), and
# the horizon-level (not CI-width) correction is SIMEX in [(d)](d_simex.py).
