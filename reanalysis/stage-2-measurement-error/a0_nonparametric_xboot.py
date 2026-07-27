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
# # Stage 2(a0) — Nonparametric bootstrap of the x-axis
#
# METR's bootstrap resamples task families → tasks → agent runs, but every replicate
# carries the same fixed `human_minutes` per task. The x-axis — itself a geometric mean
# of a handful of noisy human baseline runs — contributes **zero** to their confidence
# intervals.
#
# Here we add the missing level with **no distributional assumptions**: in each bootstrap
# replicate we also resample *which successful human baseline runs* enter each task's
# `human_minutes` (using the δ-corrected per-run times from
# [Stage 1](../stage-1-export-archaeology/)), and recompute the geometric mean.
#
# **This is a lower bound on x-axis uncertainty:**
# - tasks with a single baseline run (n=1) get zero x-variance (resampling 1 of 1);
# - researcher-**estimate** tasks have no runs at all — zero x-variance, though they are
#   disproportionately the *long* tasks;
# - RE-Bench `human_minutes` is not a gmean of raw elapsed times (different convention),
#   so it is held fixed;
# - the bootstrap of n=2–3 samples understates variance generically (~factor (n−1)/n).
#
# Three conditions, each 500 replicates with METR's own fit code and headline parameters
# (regularization 1e-5, `invsqrt_task_weight`):
#
# | condition | agent-run resampling (METR's f→t→r) | x resampling |
# |---|---|---|
# | `metr` | ✔ | – |
# | `metr+x` | ✔ | ✔ |
# | `x_only` | – | ✔ |

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from joblib import Parallel, delayed
from scipy import stats

from horizon.utils.logistic import get_x_for_quantile, logistic_regression
from horizon.wrangle.bootstrap import bootstrap_sample

import metr_fit_helpers as metr   # shared figure/fit helpers (see (a)/(c)/(d))

plotly.io.templates.default = "plotly_white"
pd.set_option("display.max_columns", None, "display.width", 200)

HERE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO = HERE / ".." / ".."
REPORT = REPO / "reports" / "time-horizon-1-0"
FIGURES = HERE / "figures"
DATA_OUT = HERE / "data"
FIGURES.mkdir(exist_ok=True), DATA_OUT.mkdir(exist_ok=True)

try:  # running under a Jupyter/VS Code kernel?
    get_ipython()  # type: ignore[name-defined]
    IS_INTERACTIVE = True
except NameError:
    IS_INTERACTIVE = False

def show(fig):
    """Display inline in interactive sessions; no-op in headless script runs
    (plotly's fallback there opens browser windows — figures are on disk anyway)."""
    if IS_INTERACTIVE:
        fig.show()

N_BOOT = int(os.environ.get("N_BOOT", 500))
REGULARIZATION = 1e-5          # headline value from reports/time-horizon-1-0/fig_params
WEIGHT_COLUMN = "invsqrt_task_weight"
CATEGORIES = ["task_family", "task_id", "run_id"]   # METR's "ftr"

runs = pd.read_json(REPORT / "data" / "raw" / "runs.jsonl", lines=True)
runs = runs.rename(columns={"alias": "agent"})
human_runs_derived = pd.read_csv(
    HERE / ".." / "stage-1-export-archaeology" / "data" / "human_runs_derived.csv"
)
print(f"{len(runs):,} runs | {runs['agent'].nunique()} agents")

# %% [markdown]
# ## The resampling pool
#
# Per task: the δ-corrected times of its successful baseline runs. Only tasks with n≥2
# such runs and a gmean-consistent `human_minutes` (HCAST `delta_corrected`, SWAA
# `swaa_exact`) vary across replicates.

# %%
resampleable_successful_runs = human_runs_derived.query(
    "score_binarized == 1 and derivation in ['delta_corrected', 'swaa_exact']"
)
successful_times_by_task = {
    task: grp["minutes_derived"].to_numpy()
    for task, grp in resampleable_successful_runs.groupby("task_id")
    if len(grp) >= 2
}
official_human_minutes_by_task = runs.groupby("task_id")["human_minutes"].first()
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


def draw_human_minutes(rng: np.random.Generator) -> pd.Series:
    """One bootstrap draw of the x-axis: resample each task's baseline runs, re-gmean."""
    human_minutes_by_task = official_human_minutes_by_task.copy()
    for task, times in successful_times_by_task.items():
        human_minutes_by_task[task] = stats.gmean(
            rng.choice(times, size=len(times), replace=True)
        )
    return human_minutes_by_task


# %% [markdown]
# ## Bootstrap machinery — METR's fit, three conditions

# %%
def fit_horizons(runs_sample: pd.DataFrame, human_minutes_by_task: pd.Series) -> dict:
    """Fit METR's weighted logistic per agent on log2(human_minutes); return p50s."""
    log2_minutes = np.log2(
        runs_sample["task_id"].map(human_minutes_by_task).to_numpy()
    )
    p50_by_agent = {}
    for agent, indices in runs_sample.groupby("agent").indices.items():
        y = runs_sample["score_binarized"].to_numpy()[indices]
        if len(np.unique(y)) < 2:
            continue
        model = logistic_regression(
            log2_minutes[indices].reshape(-1, 1), y,
            sample_weight=runs_sample[WEIGHT_COLUMN].to_numpy()[indices],
            regularization=REGULARIZATION,
            ensure_weights_sum_to_1=False,
        )
        p50_by_agent[agent] = float(np.exp2(get_x_for_quantile(model, 0.5)))
    return p50_by_agent


def one_replicate(replicate_index: int, condition: str) -> dict:
    rng = np.random.default_rng(42 + replicate_index)
    runs_sample = (
        bootstrap_sample(runs, CATEGORIES, rng) if condition in ("metr", "metr+x")
        else runs
    )
    human_minutes_by_task = (
        draw_human_minutes(rng) if condition in ("metr+x", "x_only")
        else official_human_minutes_by_task
    )
    return fit_horizons(runs_sample, human_minutes_by_task)


# Raw per-replicate horizons are cached to data/ — delete those files to force a
# recompute (seeds are fixed, so a recompute reproduces them exactly).
p50_per_agent_by_replicate_by_condition = {}
for condition in ["metr", "metr+x", "x_only"]:
    cache_path = DATA_OUT / f"a0_horizons_{condition.replace('+', '_')}.csv"
    if cache_path.exists():
        p50_per_agent_by_replicate_by_condition[condition] = pd.read_csv(cache_path)
        print(f"{condition}: loaded "
              f"{len(p50_per_agent_by_replicate_by_condition[condition])} cached replicates")
        continue
    replicate_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(one_replicate)(i, condition) for i in range(N_BOOT)
    )
    p50_per_agent_by_replicate_by_condition[condition] = pd.DataFrame(replicate_results)
    p50_per_agent_by_replicate_by_condition[condition].to_csv(cache_path, index=False, float_format=metr.CSV_FLOAT_FORMAT)
    print(f"{condition}: {len(p50_per_agent_by_replicate_by_condition[condition])} replicates")

# %% [markdown]
# ## Effect on per-agent horizon CIs

# %%
frontier_agents, release_dates, official_fits_by_agent = (
    metr.load_frontier_agents_and_dates()   # sorted, so CSV row order is reproducible
)
print(f"SOTA-at-release agents ({len(frontier_agents)}):", frontier_agents)

ci_rows = []
for condition, p50_per_agent_by_replicate in p50_per_agent_by_replicate_by_condition.items():
    for agent in p50_per_agent_by_replicate.columns:
        p50_draws = p50_per_agent_by_replicate[agent].dropna()
        p50_draws = p50_draws[np.isfinite(p50_draws)]
        if len(p50_draws) < N_BOOT * 0.9:
            continue
        ci_lower, ci_upper = p50_draws.quantile([0.025, 0.975])
        ci_rows.append({
            "agent": agent, "condition": condition,
            "p50_median": p50_draws.median(), "ci_lower": ci_lower, "ci_upper": ci_upper,
            "ci_logwidth": np.log(ci_upper / ci_lower),
        })
ci_by_agent_condition = pd.DataFrame(ci_rows)
ci_by_agent_condition.to_csv(DATA_OUT / "a0_ci_by_agent.csv", index=False, float_format=metr.CSV_FLOAT_FORMAT)

ci_logwidth_per_condition_by_agent = ci_by_agent_condition.pivot(
    index="agent", columns="condition", values="ci_logwidth"
)
ci_logwidth_per_condition_by_agent["widening_pct"] = 100 * (
    ci_logwidth_per_condition_by_agent["metr+x"]
    / ci_logwidth_per_condition_by_agent["metr"] - 1
)
ci_logwidth_per_condition_by_agent["x_share_pct"] = 100 * (
    ci_logwidth_per_condition_by_agent["x_only"]
    / ci_logwidth_per_condition_by_agent["metr+x"]
) ** 2   # variance-share heuristic
ci_widening_by_frontier_agent = ci_logwidth_per_condition_by_agent.loc[
    ci_logwidth_per_condition_by_agent.index.isin(frontier_agents)
].sort_values("widening_pct", ascending=False)
print(ci_widening_by_frontier_agent.round(1).to_string())
print(f"\nmedian p50 interval widening (SOTA-at-release agents): "
      f"{ci_widening_by_frontier_agent['widening_pct'].median():.1f}%")

agents_sorted_by_official_p50 = official_fits_by_agent.set_index("agent")["p50"].sort_values().index
agents_sorted = [
    agent for agent in agents_sorted_by_official_p50 if agent in frontier_agents
]
fig = metr.dodged_confidence_interval_figure(
    ci_by_agent_condition, agents_sorted, ["metr", "metr+x", "x_only"],
    title="50%-horizon 95% bootstrap interval — with vs without x-axis resampling",
)
fig.write_image(FIGURES / "a0_ci_comparison.png", width=850, height=620, scale=2)
show(fig)

# %% [markdown]
# ## Effect on the doubling time
#
# Per replicate: OLS of log2(p50) on release date over the frontier agents (paper's
# headline construction); doubling time = 1/slope.

# %%
release_date_by_agent = {
    agent: pd.Timestamp(date) for agent, date in release_dates["date"].items()
    if agent in frontier_agents and date is not None
}
first_release_date = min(release_date_by_agent.values())
years_since_first_release_by_agent = {
    agent: (date - first_release_date).days / 365.25
    for agent, date in release_date_by_agent.items()
}

def fit_doubling_days(p50_by_agent: pd.Series) -> float:
    points = [
        (years_since_first_release_by_agent[agent], np.log2(p50_by_agent[agent]))
        for agent in years_since_first_release_by_agent
        if agent in p50_by_agent and np.isfinite(p50_by_agent.get(agent, np.nan))
    ]
    if len(points) < 5:
        return np.nan
    x, y = map(np.array, zip(*points))
    slope = np.polyfit(x, y, 1)[0]
    return 365.25 / slope

doubling_days_per_condition_by_replicate = pd.DataFrame({
    condition: p50_per_agent_by_replicate.apply(fit_doubling_days, axis=1)
    for condition, p50_per_agent_by_replicate
    in p50_per_agent_by_replicate_by_condition.items()
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
    DATA_OUT / "a0_doubling_times.csv", index=False, float_format=metr.CSV_FLOAT_FORMAT
)

fig = px.ecdf(
    doubling_days_per_condition_by_replicate.melt(
        var_name="condition", value_name="doubling_days"
    ),
    x="doubling_days", color="condition",
    title=f"Bootstrap distribution of the doubling time (frontier agents, {N_BOOT} reps)",
    labels={"doubling_days": "doubling time (days)"},
)
fig.write_image(FIGURES / "a0_doubling_time.png", width=800, height=450, scale=2)
show(fig)

# %% [markdown]
# ## Caveats
#
# 1. **Lower bound**: n=1 tasks (21), estimate tasks, and RE-Bench contribute zero
#    x-variance here; small-n bootstrap additionally understates within-task variance.
#    Stage 2(b) replaces this with a hierarchical parametric model.
# 2. `x_only` isolates the pure x-axis contribution; its share of total variance is a
#    heuristic (variances don't add exactly across the two resampling schemes).
# 3. Doubling-time construction: single OLS over all frontier agents (2019→2026 window),
#    not METR's windowed variants; the *comparison across conditions* is the point, not
#    the absolute number.
