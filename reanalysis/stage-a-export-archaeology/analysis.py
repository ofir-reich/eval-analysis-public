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
# # Stage A — Export archaeology: reconstructing per-run human baseline times
#
# **Finding:** in the public METR export (`runs.jsonl`), the durations of the human
# baseline runs are not the originally measured times. For HCAST (and RE-Bench) they are
# **floored to the whole minute**, and are in fact *derived from* the official per-task
# aggregate `human_minutes` — the aggregation ran on unfloored internal data, then
# per-run times were reconstructed and truncated for release. SWAA runs are stored to
# the millisecond and are unaffected.
#
# **Consequences:** anyone re-deriving `human_minutes = gmean(successful baseline times)`
# from the export gets systematic misses, concentrated on short tasks (3.7 min → 3.0 is
# a 19% error; 877.02 → 877 is nothing).
#
# **Contribution:** we (1) demonstrate the flooring and its derived-ness, (2) verify the
# flooring bracket `gmean(t) ≤ human_minutes < gmean(t+1)` holds for every HCAST
# baselined task, (3) solve for a per-task sub-minute offset δ_task ∈ [0,1) such that
# `gmean(t + δ_task) = human_minutes` exactly, and (4) publish the resulting corrected
# per-run dataset (`data/human_runs_derived.csv`) as the basis for downstream analyses.
#
# Runs on **Time Horizon v1.0** (the paper's suite). See `../../README.md` for context.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import brentq

plotly.io.templates.default = "plotly_white"
pd.set_option("display.max_columns", None, "display.width", 200)
pd.options.display.precision = 3

HERE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO = HERE / ".." / ".."
FIGURES = HERE / "figures"
DATA_OUT = HERE / "data"

runs = pd.read_json(
    REPO / "reports" / "time-horizon-1-0" / "data" / "raw" / "runs.jsonl", lines=True
)
human = runs.query("alias == 'human'").copy()
human["minutes_floored"] = (human["completed_at"] - human["started_at"]) / 60_000
print(f"{len(runs):,} runs | {len(human)} human baseline runs | "
      f"{human['task_id'].nunique()} tasks with human runs")


# %% [markdown]
# ## 1. The stored durations are floored — and derived from `human_minutes`
#
# Two observations establish this:
#
# **(a) Granularity differs by source.** Every HCAST and RE-Bench human duration is a
# whole number of minutes; SWAA durations are stored to the millisecond.
#
# **(b) Single-baseliner tasks give the smoking gun.** When exactly one successful
# baseline exists, `human_minutes` *is* that run's time — and the stored duration equals
# `floor(human_minutes)` **exactly, for all such tasks**. The export's per-run time was
# computed from the aggregate, not vice versa.

# %%
frac = human["minutes_floored"] % 1
granularity = (
    human.assign(is_whole_minute=np.isclose(frac, 0))
    .groupby("task_source")["is_whole_minute"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "share_whole_minute", "count": "n_runs"})
)
print(granularity)

# single-successful-baseliner tasks: stored duration == floor(human_minutes) exactly?
succ = human.query("score_binarized == 1 and human_source == 'baseline'")
single = succ.groupby("task_id").filter(lambda g: len(g) == 1)
exact_floor = np.isclose(single["minutes_floored"], np.floor(single["human_minutes"]))
print(f"\nsingle-baseliner tasks: {len(single)} | "
      f"stored == floor(human_minutes) exactly: {exact_floor.mean():.0%}")

# %% [markdown]
# ## 2. The flooring bracket holds for every task
#
# If each true time t* is floored to t, then t ≤ t* < t+1, so for every task:
# `gmean(t) ≤ human_minutes < gmean(t+1)`. We verify this on all HCAST tasks with
# baselines (RE-Bench uses a different time convention — its `human_minutes` is not the
# gmean of raw elapsed times — and SWAA needs no bracket; both are excluded here and
# handled in §4).
#
# Note we include the three successful runs stored as **0 minutes** (flooring artifact —
# dropping them collapses the geometric mean and was the one apparent "violation" in
# early attempts).

# %%
hcast = succ.query("task_source == 'HCAST'")
bracket = (
    hcast.groupby("task_id")
    .apply(
        lambda d: pd.Series({
            "lo": stats.gmean(d["minutes_floored"]),
            "hi": stats.gmean(d["minutes_floored"] + 1),
            "human_minutes": d["human_minutes"].iloc[0],
            "n_success": len(d),
        }),
        include_groups=False,
    )
    .reset_index()
)
bracket["in_bracket"] = (bracket["human_minutes"] >= bracket["lo"] * (1 - 1e-9)) & (
    bracket["human_minutes"] < bracket["hi"] * (1 + 1e-9)
)
print(f"bracket holds: {bracket['in_bracket'].sum()} / {len(bracket)} HCAST baselined tasks")
assert bracket["in_bracket"].all(), "bracket violated — flooring model wrong for some task"

fig = px.scatter(
    bracket, x="human_minutes", y="lo",
    error_y=bracket["hi"] - bracket["lo"], error_y_minus=[0] * len(bracket),
    hover_data=["task_id", "n_success"], log_x=True, log_y=True,
    labels={"lo": "gmean of floored times (bar: +1 min ceiling)",
            "human_minutes": "official human_minutes"},
    title="Flooring bracket [gmean(t), gmean(t+1)] vs official human_minutes — y=x crosses every bar",
)
mn, mx = 0.5, 2500
fig.add_trace(go.Scatter(x=(mn, mx), y=(mn, mx), mode="lines", name="y=x",
                         line=dict(color="gray", dash="dash", width=2), opacity=0.5))
fig.write_image(FIGURES / "bracket.png", width=750, height=550, scale=2)
fig.show()

# %% [markdown]
# ## 3. Solving for δ_task
#
# The bracket guarantees (by continuity and monotonicity of δ ↦ gmean(t+δ)) a **unique**
# δ_task ∈ [0,1) with `gmean(t + δ_task) = human_minutes`. For single-baseliner tasks it
# is simply the fractional part of `human_minutes`.
#
# δ_task is the *shared per-task* sub-minute offset: the true per-run residuals are
# unrecoverable, and constant-δ is one of infinitely many completions consistent with the
# official aggregate. It is, however, the canonical minimal-assumption imputation, and it
# reproduces every official `human_minutes` **exactly**.

# %%
def solve_delta(times_floored: np.ndarray, target: float) -> float:
    f = lambda d: np.log(stats.gmean(times_floored + d)) - np.log(target)
    if f(1e-9) >= 0:          # already exact (within float noise) at delta=0
        return 0.0
    return brentq(f, 1e-9, 1.0, xtol=1e-12)

delta = (
    hcast.groupby("task_id")
    .apply(
        lambda d: solve_delta(d["minutes_floored"].to_numpy(), d["human_minutes"].iloc[0]),
        include_groups=False,
    )
    .rename("delta_task")
    .reset_index()
)
check = hcast.merge(delta, on="task_id")
resid = (
    check.groupby("task_id")
    .apply(
        lambda d: stats.gmean(d["minutes_floored"] + d["delta_task"]) / d["human_minutes"].iloc[0],
        include_groups=False,
    )
)
print(f"δ solved for {len(delta)} tasks | max |gmean(t+δ)/human_minutes − 1| = "
      f"{np.abs(resid - 1).max():.2e}")

fig = px.histogram(
    delta, x="delta_task", nbins=40,
    title=f"Distribution of δ_task across {len(delta)} HCAST tasks (uniform-ish, as expected)",
    labels={"delta_task": "δ_task (minutes)"},
)
fig.write_image(FIGURES / "delta_distribution.png", width=750, height=450, scale=2)
fig.show()

# %% [markdown]
# ## 4. The corrected per-run dataset
#
# One row per human baseline run, with `minutes_derived` and a `derivation` flag:
#
# | derivation | meaning |
# |---|---|
# | `swaa_exact` | SWAA: stored to the ms, no correction needed (δ=0) |
# | `delta_corrected` | HCAST successful run: floored + δ_task (aggregate-exact) |
# | `delta_imputed_failure` | HCAST failed run: floored + δ_task as best guess (δ not identified from failures; true value in [t, t+1)) |
# | `uncorrected_rebench` | RE-Bench: different time convention, no δ exists; floored value shipped as-is |
#
# Failed runs matter downstream as *censoring times* (Stage B), where a sub-minute offset
# is immaterial — but the flag makes the epistemic status explicit.

# %%
out = human.merge(delta, on="task_id", how="left")
is_swaa = out["task_source"] == "SWAA"
is_rebench = out["task_source"] == "RE-Bench"
is_hcast = out["task_source"] == "HCAST"

out["delta_task"] = np.where(is_swaa, 0.0, out["delta_task"])
out["minutes_derived"] = out["minutes_floored"] + out["delta_task"].fillna(0)
out["derivation"] = np.select(
    [is_swaa,
     is_hcast & out["delta_task"].notna() & (out["score_binarized"] == 1),
     is_hcast & out["delta_task"].notna() & (out["score_binarized"] == 0),
     is_rebench],
    ["swaa_exact", "delta_corrected", "delta_imputed_failure", "uncorrected_rebench"],
    default="uncorrected_no_delta",   # HCAST runs on tasks with no successful baseline
)
print(out["derivation"].value_counts().to_string())

cols = ["task_id", "task_family", "task_source", "human_source", "score_binarized",
        "score_cont", "human_minutes", "minutes_floored", "delta_task",
        "minutes_derived", "derivation"]
out[cols].to_csv(DATA_OUT / "human_runs_derived.csv", index=False)
print(f"\nwrote {DATA_OUT / 'human_runs_derived.csv'}  ({len(out)} rows)")

# %% [markdown]
# ## 5. Before/after: reconstruction error of `human_minutes`
#
# The point of the correction: re-deriving `human_minutes` from the export now succeeds
# exactly on every δ-corrected task, where the naive floored reconstruction missed by up
# to ~20% on short tasks.

# %%
naive = hcast.groupby("task_id").apply(
    lambda d: stats.gmean(d["minutes_floored"].clip(lower=1e-9)), include_groups=False)
corrected = check.groupby("task_id").apply(
    lambda d: stats.gmean(d["minutes_floored"] + d["delta_task"]), include_groups=False)
official = hcast.groupby("task_id")["human_minutes"].first()

cmp = pd.DataFrame({
    "naive (floored)": naive / official,
    "δ-corrected": corrected / official,
}).reset_index().melt(id_vars="task_id", var_name="reconstruction", value_name="ratio")
fig = px.ecdf(
    cmp, x="ratio", color="reconstruction",
    title="Reconstructed / official human_minutes — before and after δ correction",
    labels={"ratio": "reconstructed ÷ official human_minutes"},
)
fig.update_xaxes(range=[0.75, 1.05])
fig.write_image(FIGURES / "before_after_ecdf.png", width=750, height=450, scale=2)
fig.show()

# %% [markdown]
# ## Caveats
#
# 1. **δ is a per-task constant by construction** — per-run sub-minute residuals are lost;
#    any downstream analysis needing within-task time *spread* at sub-minute resolution
#    should treat δ-corrected times as interval data `[t, t+1)`.
# 2. **Failures are best-guess** (`delta_imputed_failure`): δ is identified from
#    successes only.
# 3. **RE-Bench is uncorrected**: its `human_minutes` is not the gmean of raw elapsed
#    times (8h-capped sessions; threshold = mean of 7–9h runs), so no δ ∈ [0,1) exists.
# 4. This documents the **public export**, not METR's internal analysis — the official
#    `human_minutes` has full precision throughout; the information loss is only in the
#    shipped per-run times.
