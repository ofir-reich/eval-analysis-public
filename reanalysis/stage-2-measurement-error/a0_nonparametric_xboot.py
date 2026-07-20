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
import yaml
from joblib import Parallel, delayed
from scipy import stats

from horizon.utils.logistic import get_x_for_quantile, logistic_regression
from horizon.wrangle.bootstrap import bootstrap_sample
from horizon.wrangle.out_of_sample_extrapolation import get_frontier_agents

plotly.io.templates.default = "plotly_white"
pd.set_option("display.max_columns", None, "display.width", 200)

HERE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO = HERE / ".." / ".."
REPORT = REPO / "reports" / "time-horizon-1-0"
FIGURES = HERE / "figures"
DATA_OUT = HERE / "data"
FIGURES.mkdir(exist_ok=True), DATA_OUT.mkdir(exist_ok=True)

N_BOOT = int(os.environ.get("N_BOOT", 500))
REGULARIZATION = 1e-5          # headline value from reports/time-horizon-1-0/fig_params
WEIGHT_COL = "invsqrt_task_weight"
CATEGORIES = ["task_family", "task_id", "run_id"]   # METR's "ftr"

runs = pd.read_json(REPORT / "data" / "raw" / "runs.jsonl", lines=True)
runs = runs.rename(columns={"alias": "agent"})
derived = pd.read_csv(HERE / ".." / "stage-1-export-archaeology" / "data" / "human_runs_derived.csv")
print(f"{len(runs):,} runs | {runs['agent'].nunique()} agents")

# %% [markdown]
# ## The resampling pool
#
# Per task: the δ-corrected times of its successful baseline runs. Only tasks with n≥2
# such runs and a gmean-consistent `human_minutes` (HCAST `delta_corrected`, SWAA
# `swaa_exact`) vary across replicates.

# %%
pool_df = derived.query(
    "score_binarized == 1 and derivation in ['delta_corrected', 'swaa_exact']"
)
pool = {
    task: grp["minutes_derived"].to_numpy()
    for task, grp in pool_df.groupby("task_id")
    if len(grp) >= 2
}
official_hm = runs.groupby("task_id")["human_minutes"].first()
n_tasks = runs["task_id"].nunique()
print(f"x-resampled tasks: {len(pool)} / {n_tasks} "
      f"(rest fixed: n=1, estimates, RE-Bench)")

# HCAST reproduces exactly (δ-corrected); SWAA to ~2e-4 (its timestamps are ms-rounded)
sanity = np.array([stats.gmean(v) / official_hm[t] for t, v in pool.items()])
assert np.abs(sanity - 1).max() < 1e-3, "pool gmeans should reproduce official human_minutes"


def draw_human_minutes(rng: np.random.Generator) -> pd.Series:
    """One bootstrap draw of the x-axis: resample each task's baseline runs, re-gmean."""
    hm = official_hm.copy()
    for task, times in pool.items():
        hm[task] = stats.gmean(rng.choice(times, size=len(times), replace=True))
    return hm


# %% [markdown]
# ## Bootstrap machinery — METR's fit, three conditions

# %%
def fit_horizons(df: pd.DataFrame, hm: pd.Series) -> dict:
    """Fit METR's weighted logistic per agent on log2(human_minutes); return p50s."""
    x_all = np.log2(df["task_id"].map(hm).to_numpy())
    out = {}
    for agent, idx in df.groupby("agent").indices.items():
        y = df["score_binarized"].to_numpy()[idx]
        if len(np.unique(y)) < 2:
            continue
        model = logistic_regression(
            x_all[idx].reshape(-1, 1), y,
            sample_weight=df[WEIGHT_COL].to_numpy()[idx],
            regularization=REGULARIZATION,
            ensure_weights_sum_to_1=False,
        )
        out[agent] = float(np.exp2(get_x_for_quantile(model, 0.5)))
    return out


def one_replicate(i: int, condition: str) -> dict:
    rng = np.random.default_rng(42 + i)
    df = bootstrap_sample(runs, CATEGORIES, rng) if condition in ("metr", "metr+x") else runs
    hm = draw_human_minutes(rng) if condition in ("metr+x", "x_only") else official_hm
    return fit_horizons(df, hm)


# Raw per-replicate horizons are cached to data/ — delete those files to force a
# recompute (seeds are fixed, so a recompute reproduces them exactly).
results = {}
for condition in ["metr", "metr+x", "x_only"]:
    cache = DATA_OUT / f"a0_horizons_{condition.replace('+', '_')}.csv"
    if cache.exists():
        results[condition] = pd.read_csv(cache)
        print(f"{condition}: loaded {len(results[condition])} cached replicates")
        continue
    reps = Parallel(n_jobs=-1, verbose=0)(
        delayed(one_replicate)(i, condition) for i in range(N_BOOT)
    )
    results[condition] = pd.DataFrame(reps)
    results[condition].to_csv(cache, index=False)
    print(f"{condition}: {len(results[condition])} replicates")

# %% [markdown]
# ## Effect on per-agent horizon CIs

# %%
release_dates = yaml.safe_load((REPO / "data" / "external" / "release_dates.yaml").read_text())
fits_official = pd.read_csv(REPORT / "data" / "wrangled" / "logistic_fits" / "headline.csv")
frontier = get_frontier_agents(fits_official, release_dates, 50)
print(f"frontier agents ({len(frontier)}):", sorted(frontier))

rows = []
for condition, df in results.items():
    for agent in df.columns:
        p50 = df[agent].dropna()
        p50 = p50[np.isfinite(p50)]
        if len(p50) < N_BOOT * 0.9:
            continue
        lo, hi = p50.quantile([0.025, 0.975])
        rows.append({
            "agent": agent, "condition": condition,
            "p50_median": p50.median(), "ci_lo": lo, "ci_hi": hi,
            "ci_logwidth": np.log(hi / lo),
        })
ci = pd.DataFrame(rows)
ci.to_csv(DATA_OUT / "a0_ci_by_agent.csv", index=False)

wide = ci.pivot(index="agent", columns="condition", values="ci_logwidth")
wide["widening_pct"] = 100 * (wide["metr+x"] / wide["metr"] - 1)
wide["x_share_pct"] = 100 * (wide["x_only"] / wide["metr+x"]) ** 2   # variance-share heuristic
summary = wide.loc[wide.index.isin(frontier)].sort_values("widening_pct", ascending=False)
print(summary.round(1).to_string())
print(f"\nmedian CI log-width widening (frontier agents): {summary['widening_pct'].median():.1f}%")

order = fits_official.set_index("agent")["p50"].sort_values().index
plot_df = ci[ci["agent"].isin(frontier)].copy()
plot_df["err_plus"] = plot_df["ci_hi"] - plot_df["p50_median"]
plot_df["err_minus"] = plot_df["p50_median"] - plot_df["ci_lo"]
plot_df["agent"] = pd.Categorical(plot_df["agent"],
                                  [a for a in order if a in frontier], ordered=True)
fig = px.scatter(
    plot_df.sort_values("agent"), y="agent", x="p50_median", color="condition",
    error_x="err_plus", error_x_minus="err_minus",
    log_x=True,
    title="p50 horizon, 95% bootstrap CI — with vs without x-axis resampling (frontier agents)",
    labels={"p50_median": "p50 horizon (minutes)", "agent": ""},
)
fig.update_layout(height=600)
fig.write_image(FIGURES / "a0_ci_comparison.png", width=850, height=600, scale=2)
fig.show()

# %% [markdown]
# ## Effect on the doubling time
#
# Per replicate: OLS of log2(p50) on release date over the frontier agents (paper's
# headline construction); doubling time = 1/slope.

# %%
date_lookup = {a: pd.Timestamp(d) for a, d in release_dates["date"].items()
               if a in frontier and d is not None}
t0 = min(date_lookup.values())
years = {a: (d - t0).days / 365.25 for a, d in date_lookup.items()}

def doubling_days(row: pd.Series) -> float:
    pts = [(years[a], np.log2(row[a])) for a in years
           if a in row and np.isfinite(row.get(a, np.nan))]
    if len(pts) < 5:
        return np.nan
    x, y = map(np.array, zip(*pts))
    slope = np.polyfit(x, y, 1)[0]
    return 365.25 / slope

dt = pd.DataFrame({
    condition: df.apply(doubling_days, axis=1) for condition, df in results.items()
})
dt_summary = dt.describe(percentiles=[0.025, 0.5, 0.975]).T[["2.5%", "50%", "97.5%"]]
dt_summary["ci_width_days"] = dt_summary["97.5%"] - dt_summary["2.5%"]
print(dt_summary.round(1).to_string())
dt.to_csv(DATA_OUT / "a0_doubling_times.csv", index=False)

fig = px.ecdf(
    dt.melt(var_name="condition", value_name="doubling_days"),
    x="doubling_days", color="condition",
    title=f"Bootstrap distribution of the doubling time (frontier agents, {N_BOOT} reps)",
    labels={"doubling_days": "doubling time (days)"},
)
fig.write_image(FIGURES / "a0_doubling_time.png", width=800, height=450, scale=2)
fig.show()

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
