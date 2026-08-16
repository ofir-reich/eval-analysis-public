# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stage 2 — Parametric x-bootstrap
#
# METR's bootstrap resamples task families → tasks → agent runs, but every replicate
# carries the same fixed `human_minutes` per task. The x-axis — itself a geometric mean
# of a handful of noisy human baseline runs — contributes **zero** to their confidence
# intervals. This analysis adds the missing level, in two passes:
#
# 1. **An assumption-free floor**: resample *which successful baseline runs* enter each
#    task's geometric mean (δ-corrected times from
#    [Stage 1](../stage-1-export-archaeology/)). No distributional assumptions — but
#    only the tasks with ≥2 successful runs can vary, so it is a *lower bound*.
# 2. **The parametric bootstrap**: draw every task's `human_minutes` from
#    LogNormal(ln `human_minutes`, SEM²_task), with the per-task SEM from
#    [the σ analysis](sigma_task.py) — so *every* task carries x-uncertainty, including
#    the single-baseline, researcher-estimate, and RE-Bench tasks the floor must freeze.
#
# Both passes use METR's own fit code and headline parameters (weighted logistic,
# regularization 1e-5, `invsqrt_task_weight`), and both add the x-draw on top of METR's
# family→task→run resampling so the comparison is like for like:
#
# | condition | agent-run resampling (METR's f→t→r) | x draw |
# |---|---|---|
# | `metr` | ✔ | – |
# | `metr+x` / `metr+x_param` | ✔ | ✔ |
# | `x_only` / `x_param_only` | – | ✔ |

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from joblib import Parallel, delayed
from scipy import stats

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

show = metr.show   # inline display in interactive sessions; no-op headless

N_BOOT = int(os.environ.get("N_BOOT", 500))
CATEGORIES = ["task_family", "task_id", "run_id"]   # METR's "ftr"

# %%
runs = metr.load_runs()
official_human_minutes_by_task = metr.official_human_minutes_by_task(runs)
human_runs_derived = pd.read_csv(metr.derived_human_runs_csv())
frontier_agents, release_dates, official_fits_by_agent = (
    metr.load_frontier_agents_and_dates()
)
years_since_release_by_agent = metr.years_since_first_release_by_agent(
    frontier_agents, release_dates
)
print(f"{len(runs):,} runs | {runs['agent'].nunique()} agents | "
      f"{len(frontier_agents)} SOTA-at-release")


def confidence_intervals_by_agent(
    p50_per_agent_by_replicate_by_condition: dict, csv_name: str,
    agents: list | None = None,
) -> pd.DataFrame:
    """95% interval of each agent's p50 draws per condition; written to data/.
    `agents` fixes the iteration (and hence CSV row) order; None = column order."""
    ci_rows = []
    for condition, p50_per_agent_by_replicate in (
        p50_per_agent_by_replicate_by_condition.items()
    ):
        for agent in (agents if agents is not None
                      else p50_per_agent_by_replicate.columns):
            if agent not in p50_per_agent_by_replicate.columns:
                continue
            p50_draws = p50_per_agent_by_replicate[agent].dropna()
            p50_draws = p50_draws[np.isfinite(p50_draws)]
            if len(p50_draws) < len(p50_per_agent_by_replicate) * 0.9:
                continue
            ci_lower, ci_upper = p50_draws.quantile([0.025, 0.975])
            ci_rows.append({
                "agent": agent, "condition": condition,
                "p50_median": p50_draws.median(),
                "ci_lower": ci_lower, "ci_upper": ci_upper,
                "ci_logwidth": np.log(ci_upper / ci_lower),
            })
    ci_by_agent_condition = pd.DataFrame(ci_rows)
    ci_by_agent_condition.to_csv(
        DATA_OUT / csv_name, index=False, float_format=metr.CSV_FLOAT_FORMAT
    )
    return ci_by_agent_condition


def doubling_time_summary(
    p50_per_agent_by_replicate_by_condition: dict, csv_name: str
) -> pd.DataFrame:
    """Doubling time per replicate and condition; summary printed, draws written."""
    doubling_days_per_condition_by_replicate = pd.DataFrame({
        condition: p50_per_agent_by_replicate.apply(
            metr.fit_doubling_days, axis=1,
            years_since_release_by_agent=years_since_release_by_agent,
        )
        for condition, p50_per_agent_by_replicate
        in p50_per_agent_by_replicate_by_condition.items()
    })
    summary = (
        doubling_days_per_condition_by_replicate
        .describe(percentiles=[0.025, 0.5, 0.975]).T[["2.5%", "50%", "97.5%"]]
    )
    summary["ci_width_days"] = summary["97.5%"] - summary["2.5%"]
    print(summary.round(1).to_string())
    doubling_days_per_condition_by_replicate.to_csv(
        DATA_OUT / csv_name, index=False, float_format=metr.CSV_FLOAT_FORMAT
    )
    return doubling_days_per_condition_by_replicate


agents_sorted_by_official_p50 = (
    official_fits_by_agent.set_index("agent")["p50"].sort_values().index
)
agents_sorted = [
    agent for agent in agents_sorted_by_official_p50 if agent in frontier_agents
]

# %% [markdown]
# ## Pass 1 — the assumption-free floor: nonparametric x-resampling
#
# Per task: the δ-corrected times of its successful baseline runs. Only tasks with n≥2
# such runs and a gmean-consistent `human_minutes` (HCAST `delta_corrected`, SWAA
# `swaa_exact`) vary across replicates; everything else is frozen:
#
# - tasks with a single baseline run get zero x-variance (resampling 1 of 1);
# - researcher-**estimate** tasks have no runs at all — zero x-variance, though they are
#   disproportionately the *long* tasks;
# - RE-Bench `human_minutes` is not a gmean of raw elapsed times (different convention),
#   so it is held fixed;
# - the bootstrap of n=2–3 samples understates variance generically (~factor (n−1)/n).

# %%
resampleable_successful_runs = human_runs_derived.query(
    "score_binarized == 1 and derivation in ['delta_corrected', 'swaa_exact']"
)
successful_times_by_task = {
    task: grp["minutes_derived"].to_numpy()
    for task, grp in resampleable_successful_runs.groupby("task_id")
    if len(grp) >= 2
}
n_tasks = runs["task_id"].nunique()
print(f"x-resampled tasks: {len(successful_times_by_task)} / {n_tasks} "
      f"(rest fixed: n=1, estimates, RE-Bench)")

# HCAST reproduces exactly (δ-corrected); SWAA to ~2e-4 (its timestamps are ms-rounded)
pool_gmean_to_official_ratio = np.array([
    stats.gmean(times) / official_human_minutes_by_task[task]
    for task, times in successful_times_by_task.items()
])
assert np.abs(pool_gmean_to_official_ratio - 1).max() < 1e-3, \
    "pool gmeans should reproduce official human_minutes"


def draw_human_minutes_nonparam(rng: np.random.Generator) -> pd.Series:
    """One bootstrap draw of the x-axis: resample each task's baseline runs, re-gmean."""
    human_minutes_by_task = official_human_minutes_by_task.copy()
    for task, times in successful_times_by_task.items():
        human_minutes_by_task[task] = stats.gmean(
            rng.choice(times, size=len(times), replace=True)
        )
    return human_minutes_by_task


def one_replicate_nonparam(replicate_index: int, condition: str) -> dict:
    rng = np.random.default_rng(42 + replicate_index)
    runs_sample = (
        bootstrap_sample(runs, CATEGORIES, rng) if condition in ("metr", "metr+x")
        else runs
    )
    human_minutes_by_task = (
        draw_human_minutes_nonparam(rng) if condition in ("metr+x", "x_only")
        else official_human_minutes_by_task
    )
    return metr.fit_horizons_per_quantile_by_agent(
        runs_sample, human_minutes_by_task, quantiles=(0.5,)
    )[0.5]


# Raw per-replicate horizons are cached to data/ — delete those files to force a
# recompute (seeds are fixed, so a recompute reproduces them exactly).
nonparam_p50_by_replicate_by_condition = {}
for condition in ["metr", "metr+x", "x_only"]:
    cache_path = (
        DATA_OUT / f"xboot_nonparam_horizons_{condition.replace('+', '_')}{SUFFIX}.csv"
    )
    if cache_path.exists():
        # one row = one replicate, so the cache is only usable at the current N_BOOT
        cached_replicates = pd.read_csv(cache_path)
        if len(cached_replicates) == N_BOOT:
            nonparam_p50_by_replicate_by_condition[condition] = cached_replicates
            print(f"{condition}: loaded {len(cached_replicates)} cached replicates")
            continue
        print(f"{condition}: cache has {len(cached_replicates)} replicates but "
              f"N_BOOT={N_BOOT} — recomputing and overwriting it")
    replicate_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(one_replicate_nonparam)(i, condition) for i in range(N_BOOT)
    )
    nonparam_p50_by_replicate_by_condition[condition] = pd.DataFrame(replicate_results)
    nonparam_p50_by_replicate_by_condition[condition].to_csv(
        cache_path, index=False, float_format=metr.CSV_FLOAT_FORMAT
    )
    print(f"{condition}: {len(nonparam_p50_by_replicate_by_condition[condition])} replicates")

# %% [markdown]
# ### Floor result: per-agent horizon CIs

# %%
nonparam_ci_by_agent_condition = confidence_intervals_by_agent(
    nonparam_p50_by_replicate_by_condition, f"xboot_nonparam_ci_by_agent{SUFFIX}.csv"
)
nonparam_logwidth_per_condition_by_agent = nonparam_ci_by_agent_condition.pivot(
    index="agent", columns="condition", values="ci_logwidth"
)
nonparam_logwidth_per_condition_by_agent["widening_pct"] = 100 * (
    nonparam_logwidth_per_condition_by_agent["metr+x"]
    / nonparam_logwidth_per_condition_by_agent["metr"] - 1
)
nonparam_logwidth_per_condition_by_agent["x_share_pct"] = 100 * (
    nonparam_logwidth_per_condition_by_agent["x_only"]
    / nonparam_logwidth_per_condition_by_agent["metr+x"]
) ** 2   # variance-share heuristic
nonparam_widening_by_frontier_agent = nonparam_logwidth_per_condition_by_agent.loc[
    nonparam_logwidth_per_condition_by_agent.index.isin(frontier_agents)
].sort_values("widening_pct", ascending=False)
print(nonparam_widening_by_frontier_agent.round(1).to_string())
print(f"\nmedian p50 interval widening, nonparametric floor (SOTA-at-release agents): "
      f"{nonparam_widening_by_frontier_agent['widening_pct'].median():.1f}%")

fig = metr.dodged_confidence_interval_figure(
    nonparam_ci_by_agent_condition, agents_sorted, ["metr", "metr+x", "x_only"],
    title="50%-horizon 95% bootstrap interval — with vs without x-axis resampling",
)
fig.write_image(FIGURES / f"xboot_nonparam_ci_comparison{SUFFIX}.png",
                width=850, height=620, scale=2)
show(fig)

# %% [markdown]
# ### Floor result: doubling time

# %%
nonparam_doubling_by_replicate = doubling_time_summary(
    nonparam_p50_by_replicate_by_condition, f"xboot_nonparam_doubling_times{SUFFIX}.csv"
)
fig = px.ecdf(
    nonparam_doubling_by_replicate.melt(
        var_name="condition", value_name="doubling_days"
    ),
    x="doubling_days", color="condition",
    title=f"Bootstrap distribution of the doubling time (frontier agents, {N_BOOT} reps)",
    labels={"doubling_days": "doubling time (days)"},
)
fig.write_image(FIGURES / f"xboot_nonparam_doubling_time{SUFFIX}.png",
                width=800, height=450, scale=2)
show(fig)

# %% [markdown]
# The *resampleable* component of x-axis noise barely moves the headline results. What
# this floor cannot see is exactly where the remaining risk lives: single-baseline
# tasks, researcher estimates (the long tasks driving the top-end horizons), small-n
# variance understatement, and *systematic* (correlated) errors. The parametric pass
# below lifts the first three; the systematic component is
# [the coherent shift](coherent_shift.py).

# %% [markdown]
# ## Pass 2 — the parametric bootstrap with per-task σ
#
# In each replicate, draw **every** task's `human_minutes` from
# LogNormal(ln `human_minutes`, SEM²_task), with SEM_task = σ̃_task / √n for baselined
# tasks and METR's σ = 1.05 for researcher estimates — from
# [the σ analysis](sigma_task.py). Same conditions and fit code as the floor, so the
# two are directly comparable; p80 is fitted alongside p50 here.

# %%
sigma_by_task = pd.read_csv(DATA_OUT / f"sigma_by_task{SUFFIX}.csv", index_col=0)
log_human_minutes_by_task = np.log(official_human_minutes_by_task)
sem_log_by_task = sigma_by_task["sem_log"].reindex(official_human_minutes_by_task.index)
assert sem_log_by_task.notna().all(), "every task must carry a SEM from the σ analysis"
print(f"all {len(sem_log_by_task)} tasks carry x-uncertainty "
      f"(vs {len(successful_times_by_task)}/{n_tasks} in the floor)")
print("SEM_log quantiles:", sem_log_by_task.quantile([.1, .5, .9]).round(3).to_dict())


def draw_human_minutes_param(rng: np.random.Generator) -> pd.Series:
    """Parametric x draw: every task's ln(human_minutes) jittered by N(0, SEM²_task)."""
    return np.exp(
        log_human_minutes_by_task + rng.normal(0, sem_log_by_task)
    )


def one_replicate_param(replicate_index: int, condition: str) -> dict:
    rng = np.random.default_rng(42 + replicate_index)
    runs_sample = (
        bootstrap_sample(runs, CATEGORIES, rng)
        if condition in ("metr", "metr+x_param") else runs
    )
    human_minutes_by_task = (
        draw_human_minutes_param(rng)
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


param_horizons_by_replicate_by_condition = {}
for condition in ["metr", "metr+x_param", "x_param_only"]:
    cache_path = (
        DATA_OUT / f"xboot_param_horizons_{condition.replace('+', '_')}{SUFFIX}.csv"
    )
    if cache_path.exists():
        # one row = one replicate, so the cache is only usable at the current N_BOOT
        cached_replicates = pd.read_csv(cache_path)
        if len(cached_replicates) == N_BOOT:
            param_horizons_by_replicate_by_condition[condition] = cached_replicates
            print(f"{condition}: loaded {len(cached_replicates)} cached")
            continue
        print(f"{condition}: cache has {len(cached_replicates)} replicates but "
              f"N_BOOT={N_BOOT} — recomputing and overwriting it")
    replicate_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(one_replicate_param)(i, condition) for i in range(N_BOOT)
    )
    param_horizons_by_replicate_by_condition[condition] = pd.DataFrame(replicate_results)
    param_horizons_by_replicate_by_condition[condition].to_csv(
        cache_path, index=False, float_format=metr.CSV_FLOAT_FORMAT
    )
    print(f"{condition}: {len(param_horizons_by_replicate_by_condition[condition])} replicates")

# %% [markdown]
# ### Effect on per-agent p50 CIs — parametric vs the nonparametric floor

# %%
param_p50_by_replicate_by_condition = {
    condition: horizons_by_replicate[
        [column for column in horizons_by_replicate.columns if column.endswith("@p50")]
    ].rename(columns=lambda column: column.removesuffix("@p50"))
    for condition, horizons_by_replicate
    in param_horizons_by_replicate_by_condition.items()
}
param_ci_by_agent_condition = confidence_intervals_by_agent(
    param_p50_by_replicate_by_condition, f"xboot_param_ci_by_agent{SUFFIX}.csv",
    agents=frontier_agents,
)
param_logwidth_per_condition_by_agent = param_ci_by_agent_condition.pivot(
    index="agent", columns="condition", values="ci_logwidth"
)
param_logwidth_per_condition_by_agent["widening_pct"] = 100 * (
    param_logwidth_per_condition_by_agent["metr+x_param"]
    / param_logwidth_per_condition_by_agent["metr"] - 1
)
param_logwidth_per_condition_by_agent["nonparam_widening_pct"] = (
    nonparam_logwidth_per_condition_by_agent["widening_pct"]
)
print(param_logwidth_per_condition_by_agent.round(1).sort_values(
    "widening_pct", ascending=False).to_string())
print(f"\nmedian p50 CI widening — parametric (all tasks): "
      f"{param_logwidth_per_condition_by_agent['widening_pct'].median():.1f}%")
print(f"median p50 CI widening — nonparametric floor: "
      f"{param_logwidth_per_condition_by_agent['nonparam_widening_pct'].median():.1f}%")

# %%
fig = metr.dodged_confidence_interval_figure(
    param_ci_by_agent_condition, agents_sorted, ["metr", "metr+x_param", "x_param_only"],
    title="50%-horizon 95% interval — parametric per-task x-noise vs METR baseline",
)
fig.write_image(FIGURES / f"xboot_param_ci_comparison{SUFFIX}.png",
                width=850, height=620, scale=2)
show(fig)

# %% [markdown]
# ### Effect on the doubling time

# %%
param_doubling_by_replicate = doubling_time_summary(
    param_p50_by_replicate_by_condition, f"xboot_param_doubling_times{SUFFIX}.csv"
)
fig = px.ecdf(
    param_doubling_by_replicate.melt(
        var_name="condition", value_name="doubling_days"),
    x="doubling_days", color="condition",
    title=f"Doubling-time bootstrap distribution — parametric per-task x-noise "
          f"({N_BOOT} reps)",
    labels={"doubling_days": "doubling time (days)"},
)
fig.write_image(FIGURES / f"xboot_param_doubling_time{SUFFIX}.png",
                width=800, height=450, scale=2)
show(fig)

# %% [markdown]
# ## Takeaway
#
# Extending x-noise to *all* tasks parametrically (vs the resampleable-only floor)
# raises the per-agent CI widening and the doubling-time CI width — the printed numbers
# above quantify by how much. This is the honest parametric estimate of the independent
# x-axis contribution; the *systematic* counterpart is
# [the coherent shift](coherent_shift.py), and the horizon-level (not CI-width)
# correction is [SIMEX](simex.py).
#
# ## Caveats
#
# 1. The floor's `x_only` share of total variance is a heuristic (variances don't add
#    exactly across the two resampling schemes).
# 2. Doubling-time construction: single OLS over all SOTA-at-release agents, not METR's
#    windowed variants; the *comparison across conditions* is the point, not the
#    absolute number.
# 3. The parametric draw treats x-noise as independent across tasks — correlated error
#    is explored separately, as a scenario, in [the coherent shift](coherent_shift.py).
