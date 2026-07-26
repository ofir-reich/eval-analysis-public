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
# # Stage 1 — Export archaeology: reconstructing per-run human baseline times
#
# **Finding:** in the public METR export (`runs.jsonl`), the durations of the human
# baseline runs are not the originally measured times. For HCAST (and RE-Bench) they were
# **floored to the whole minute** before public release. The official per-task aggregate
# `human_minutes` was computed upstream from the *unfloored* times and is published at
# full precision — so the floored per-run times cannot exactly reproduce it. SWAA runs
# are stored to the millisecond and are unaffected.
#
# **Consequences:** anyone re-deriving `human_minutes = gmean(successful baseline times)`
# from the export gets systematic misses, concentrated on short tasks (3.7 min → 3.0 is
# a 19% error; 877.02 → 877 is nothing).
#
# **Contribution:** we (1) show the flooring and its derived-ness, (2) verify the
# flooring bracket `gmean(t) ≤ human_minutes < gmean(t+1)` holds for every HCAST
# baselined task, (3) solve for a per-task sub-minute offset δ_task ∈ [0,1) such that
# `gmean(t + δ_task) = human_minutes` exactly, and (4) publish the resulting per-run
# dataset (`data/human_runs_derived*.csv`) as the basis for downstream analyses.
#
# δ_task is an **imputation, not a recovery**: the individual sub-minute residuals are
# gone for good, and every run on a task receives the same offset. What it buys is that
# the task's aggregate comes out exactly right instead of biased low.
#
# Everything runs on **both public suites**: Time Horizon **v1.0** (the paper's suite,
# used for the narrative below and for downstream stages) and **v1.1** (§7 replication).
# See `../../README.md` for context.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import brentq

plotly.io.templates.default = "plotly_white"
pd.set_option("display.max_columns", None, "display.width", 200)
pd.options.display.precision = 3

HERE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO = HERE / ".." / ".."
FIGURES = HERE / "figures"
DATA_OUT = HERE / "data"

MAIN_DATASET_VERSION = "1-0"   # the paper's suite; v1.1 replication in §7
TASK_SOURCE_ORDER = ["HCAST", "RE-Bench", "SWAA"]
TASK_SOURCE_COLORS = {"HCAST": "#2a78d6", "RE-Bench": "#eb6834", "SWAA": "#1baf7a"}
TASK_SOURCE_SYMBOLS = {"HCAST": "circle", "RE-Bench": "square", "SWAA": "diamond"}

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

def load_human_runs(dataset_version: str) -> pd.DataFrame:
    """Human baseline/estimate runs of one public suite, with stored duration in minutes."""
    runs = pd.read_json(
        REPO / "reports" / f"time-horizon-{dataset_version}" / "data" / "raw" / "runs.jsonl",
        lines=True,
    )
    human_runs = runs.query("alias == 'human'").copy()
    human_runs["minutes_floored"] = (
        human_runs["completed_at"] - human_runs["started_at"]
    ) / 60_000
    return human_runs

human_runs = load_human_runs(MAIN_DATASET_VERSION)
successful_baseline_runs = human_runs.query(
    "score_binarized == 1 and human_source == 'baseline'"
)
print(f"Time Horizon v{MAIN_DATASET_VERSION.replace('-', '.')}: "
      f"{len(human_runs)} human runs | {human_runs['task_id'].nunique()} tasks | "
      f"{len(successful_baseline_runs)} successful baseline runs")


# %% [markdown]
# ## 1. The symptom, by task source
#
# `human_minutes` is documented as the geometric mean of the successful human
# baseliners' completion times. So re-derive it from the export — per task, take the
# gmean of the stored durations of the successful baseline runs — and plot it against
# the official value. If the export were lossless, every point would sit on y=x.
#
# It splits cleanly by task source:
#
# * **SWAA** lands on the line (within 4×10⁻⁴ relative) — nothing to fix.
# * **HCAST** sits *below* the line, never above, by up to ~20%, and the miss grows as
#   tasks get shorter — the signature of a downward truncation of the per-run times.
# * **RE-Bench** sits *above* the line, by up to 76%: its `human_minutes` is not a gmean
#   of elapsed times at all (8h-capped sessions; §5), so it is a different animal.
#
# Only HCAST needs — and admits — the correction developed in §2–§4.

# %%
def gmean_stored_by_task(runs_of_interest: pd.DataFrame) -> pd.DataFrame:
    """Per task: gmean of stored durations vs the official human_minutes aggregate."""
    return (
        runs_of_interest.groupby(["task_source", "task_id"])
        .apply(
            lambda d: pd.Series({
                "gmean_stored": stats.gmean(d["minutes_floored"]),
                "human_minutes": d["human_minutes"].iloc[0],
                "n_success": len(d),
                "n_zero_minute_runs": int((d["minutes_floored"] == 0).sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )

gmean_per_official_by_task = gmean_stored_by_task(successful_baseline_runs)
gmean_per_official_by_task["ratio_reconstructed_to_official"] = (
    gmean_per_official_by_task["gmean_stored"]
    / gmean_per_official_by_task["human_minutes"]
)
print(
    gmean_per_official_by_task.groupby("task_source")["ratio_reconstructed_to_official"]
    .agg(n_tasks="count", worst_miss="min", median="median", largest="max")
)

# three HCAST tasks contain a run floored all the way to 0 min, collapsing the gmean to
# exactly 0 — unplottable on a log axis, so pin them to the axis floor and mark them.
AXIS_FLOOR_MINUTES = 0.008
n_collapsed = int((gmean_per_official_by_task["gmean_stored"] == 0).sum())
gmean_per_official_by_task["gmean_stored_plotted"] = gmean_per_official_by_task[
    "gmean_stored"].clip(lower=AXIS_FLOOR_MINUTES)

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.11,
    subplot_titles=("gmean of stored per-run times vs official",
                    "ratio: reconstructed ÷ official"),
)
for task_source in TASK_SOURCE_ORDER:
    tasks_of_source = gmean_per_official_by_task.query("task_source == @task_source")
    marker = dict(color=TASK_SOURCE_COLORS[task_source], size=8,
                  symbol=TASK_SOURCE_SYMBOLS[task_source],
                  line=dict(color="white", width=1))
    hover = ("%{customdata[0]}<br>official %{customdata[1]:.3f} min"
             "<br>n successful baselines %{customdata[2]:.0f}<extra></extra>")
    custom = tasks_of_source[["task_id", "human_minutes", "n_success"]]
    fig.add_trace(go.Scatter(
        x=tasks_of_source["human_minutes"], y=tasks_of_source["gmean_stored_plotted"],
        mode="markers", name=task_source, legendgroup=task_source,
        marker=marker, customdata=custom, hovertemplate=hover), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=tasks_of_source["human_minutes"],
        y=tasks_of_source["ratio_reconstructed_to_official"],
        mode="markers", name=task_source, legendgroup=task_source, showlegend=False,
        marker=marker, customdata=custom, hovertemplate=hover), row=1, col=2)

identity_line = (AXIS_FLOOR_MINUTES, 2000)
fig.add_trace(go.Scatter(x=identity_line, y=identity_line, mode="lines", name="y = x",
                         line=dict(color="gray", dash="dash", width=2), opacity=0.6),
              row=1, col=1)
fig.add_hline(y=1, line=dict(color="gray", dash="dash", width=2), opacity=0.6, row=1, col=2)
fig.update_xaxes(type="log", dtick=1, title_text="official human_minutes", row=1, col=1)
fig.update_xaxes(type="log", dtick=1, title_text="official human_minutes", row=1, col=2)
fig.update_yaxes(type="log", dtick=1, title_text="gmean of stored durations (min)",
                 row=1, col=1)
fig.update_yaxes(title_text="reconstructed ÷ official", range=[-0.05, 1.85], row=1, col=2)
fig.update_layout(
    title=(f"Re-deriving human_minutes from the v{MAIN_DATASET_VERSION.replace('-', '.')}"
           " export — SWAA lands on y=x, HCAST falls below it"
           f"<br><sup>one point per task, all {len(gmean_per_official_by_task)} tasks with "
           f"≥1 successful human baseline run; {n_collapsed} HCAST tasks whose gmean "
           "collapses to 0 are pinned to the axis floor (left panel)</sup>"),
    legend=dict(title="task source", orientation="h", y=-0.18, x=0.5, xanchor="center"),
    margin=dict(t=95),
)
fig.write_image(FIGURES / "gmean_vs_official_by_source.png", width=1000, height=520, scale=2)
show(fig)

# %% [markdown]
# ## 2. The stored durations are floored to the whole minute
#
# Two observations establish this:
#
# **(a) Granularity differs by source.** Every HCAST and RE-Bench human duration is a
# whole number of minutes; SWAA durations are stored to the millisecond.
#
# **(b) Single-baseliner tasks pin down the exact scheme.** When exactly one successful
# baseline exists, `human_minutes` *is* that run's (unfloored) time — and the stored
# duration equals `floor(human_minutes)` **exactly, for all such tasks**. So the
# truncation is a floor (not a round), and the upstream aggregate kept full precision.

# %%
minutes_fractional_part = human_runs["minutes_floored"] % 1
whole_minute_share_by_task_source = (
    human_runs.assign(is_whole_minute=np.isclose(minutes_fractional_part, 0))
    .groupby("task_source")["is_whole_minute"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "share_whole_minute", "count": "n_runs"})
)
print(whole_minute_share_by_task_source)

# single-successful-baseliner tasks: stored duration == floor(human_minutes) exactly?
successful_runs_single_baseliner = successful_baseline_runs.groupby("task_id").filter(
    lambda g: len(g) == 1
)
is_exact_floor = np.isclose(
    successful_runs_single_baseliner["minutes_floored"],
    np.floor(successful_runs_single_baseliner["human_minutes"]),
)
print(f"\nsingle-baseliner tasks: {len(successful_runs_single_baseliner)} | "
      f"stored == floor(human_minutes) exactly: {is_exact_floor.mean():.0%}")

# %% [markdown]
# ## 3. The flooring bracket holds for every HCAST task
#
# If each true time t* is floored to t, then t ≤ t* < t+1, so for every task:
# `gmean(t) ≤ human_minutes < gmean(t+1)`. We verify this on the **HCAST** tasks with
# baselines. RE-Bench and SWAA are excluded and stay excluded for the rest of the
# analysis: RE-Bench uses a different time convention (its `human_minutes` is not the
# gmean of raw elapsed times — §1 right panel, and §5), and SWAA needs no correction.
#
# Note we include the three successful runs stored as **0 minutes** (flooring artifact —
# dropping them collapses the geometric mean and was the one apparent "violation" in
# early attempts).

# %%
def bracket_by_task_of(successful_hcast_runs: pd.DataFrame) -> pd.DataFrame:
    """Per HCAST task: the flooring bracket [gmean(t), gmean(t+1)] around human_minutes."""
    bracket_by_task = (
        successful_hcast_runs.groupby("task_id")
        .apply(
            lambda d: pd.Series({
                "gmean_lower": stats.gmean(d["minutes_floored"]),
                "gmean_upper": stats.gmean(d["minutes_floored"] + 1),
                "human_minutes": d["human_minutes"].iloc[0],
                "n_success": len(d),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    bracket_by_task["in_bracket"] = (
        bracket_by_task["human_minutes"] >= bracket_by_task["gmean_lower"] * (1 - 1e-9)
    ) & (bracket_by_task["human_minutes"] < bracket_by_task["gmean_upper"] * (1 + 1e-9))
    return bracket_by_task

successful_hcast_baseline_runs = successful_baseline_runs.query("task_source == 'HCAST'")
bracket_by_task = bracket_by_task_of(successful_hcast_baseline_runs)
print(f"bracket holds: {bracket_by_task['in_bracket'].sum()} / {len(bracket_by_task)} "
      "HCAST baselined tasks")
assert bracket_by_task["in_bracket"].all(), "bracket violated — flooring model wrong for some task"

fig = px.scatter(
    bracket_by_task, x="human_minutes", y="gmean_lower",
    error_y=bracket_by_task["gmean_upper"] - bracket_by_task["gmean_lower"],
    error_y_minus=[0] * len(bracket_by_task),
    hover_data=["task_id", "n_success"], log_x=True, log_y=True,
    color_discrete_sequence=[TASK_SOURCE_COLORS["HCAST"]],
    labels={"gmean_lower": "gmean of floored times (bar: +1 min ceiling)",
            "human_minutes": "official human_minutes"},
    title=("Flooring bracket [gmean(t), gmean(t+1)] vs official human_minutes — "
           "y=x crosses every bar"
           f"<br><sup>HCAST only ({len(bracket_by_task)} tasks with ≥1 successful "
           "baseline); RE-Bench and SWAA excluded — see §1</sup>"),
)
line_start, line_end = 0.5, 2500
fig.add_trace(go.Scatter(x=(line_start, line_end), y=(line_start, line_end),
                         mode="lines", name="y=x",
                         line=dict(color="gray", dash="dash", width=2), opacity=0.5))
fig.update_xaxes(dtick=1)
fig.update_yaxes(dtick=1)
fig.update_layout(margin=dict(t=95))
fig.write_image(FIGURES / "bracket.png", width=750, height=570, scale=2)
show(fig)

# %% [markdown]
# ## 4. Solving for δ_task
#
# The bracket guarantees (by continuity and monotonicity of δ ↦ gmean(t+δ)) a **unique**
# δ_task ∈ [0,1) with `gmean(t + δ_task) = human_minutes`. For single-baseliner tasks it
# is simply the fractional part of `human_minutes`.
#
# δ_task is a *shared per-task* sub-minute offset, i.e. an imputation: the true per-run
# residuals are unrecoverable, and constant-δ is one of infinitely many completions
# consistent with the official aggregate. It is, however, the canonical
# minimal-assumption choice, and it reproduces every official `human_minutes` **exactly**.

# %%
def solve_delta(times_floored: np.ndarray, target: float) -> float:
    log_gmean_gap = lambda d: np.log(stats.gmean(times_floored + d)) - np.log(target)
    if log_gmean_gap(1e-9) >= 0:      # already exact (within float noise) at delta=0
        return 0.0
    return brentq(log_gmean_gap, 1e-9, 1.0, xtol=1e-12)

def delta_by_task_of(successful_hcast_runs: pd.DataFrame) -> pd.DataFrame:
    return (
        successful_hcast_runs.groupby("task_id")
        .apply(
            lambda d: solve_delta(d["minutes_floored"].to_numpy(), d["human_minutes"].iloc[0]),
            include_groups=False,
        )
        .rename("delta_task")
        .reset_index()
    )

delta_by_task = delta_by_task_of(successful_hcast_baseline_runs)
successful_hcast_runs_with_delta = successful_hcast_baseline_runs.merge(
    delta_by_task, on="task_id"
)
reconstruction_ratio_by_task = (
    successful_hcast_runs_with_delta.groupby("task_id")
    .apply(
        lambda d: stats.gmean(d["minutes_floored"] + d["delta_task"]) / d["human_minutes"].iloc[0],
        include_groups=False,
    )
)
print(f"δ solved for {len(delta_by_task)} tasks | max |gmean(t+δ)/human_minutes − 1| = "
      f"{np.abs(reconstruction_ratio_by_task - 1).max():.2e}")

fig = px.histogram(
    delta_by_task, x="delta_task", nbins=40,
    color_discrete_sequence=[TASK_SOURCE_COLORS["HCAST"]],
    title=(f"Distribution of δ_task across {len(delta_by_task)} tasks (uniform-ish, as expected)"
           "<br><sup>HCAST only — RE-Bench and SWAA get no δ (see §1, §5)</sup>"),
    labels={"delta_task": "δ_task (minutes)"},
)
fig.update_layout(margin=dict(t=95))
fig.write_image(FIGURES / "delta_distribution.png", width=750, height=470, scale=2)
show(fig)

# %% [markdown]
# ## 5. The per-run dataset
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
# Failed runs matter downstream as *censoring times* (Stage 3, survival analysis), where
# a sub-minute offset is immaterial — but the flag makes the epistemic status explicit.

# %%
DERIVED_OUTPUT_COLUMNS = ["task_id", "task_family", "task_source", "human_source",
                          "score_binarized", "score_cont", "human_minutes",
                          "minutes_floored", "delta_task", "minutes_derived", "derivation"]

def build_derived_runs(human_runs: pd.DataFrame, delta_by_task: pd.DataFrame) -> pd.DataFrame:
    human_runs_derived = human_runs.merge(delta_by_task, on="task_id", how="left")
    is_swaa = human_runs_derived["task_source"] == "SWAA"
    is_rebench = human_runs_derived["task_source"] == "RE-Bench"
    is_hcast = human_runs_derived["task_source"] == "HCAST"

    human_runs_derived["delta_task"] = np.where(
        is_swaa, 0.0, human_runs_derived["delta_task"]
    )
    human_runs_derived["minutes_derived"] = (
        human_runs_derived["minutes_floored"] + human_runs_derived["delta_task"].fillna(0)
    )
    human_runs_derived["derivation"] = np.select(
        [is_swaa,
         is_hcast & human_runs_derived["delta_task"].notna()
             & (human_runs_derived["score_binarized"] == 1),
         is_hcast & human_runs_derived["delta_task"].notna()
             & (human_runs_derived["score_binarized"] == 0),
         is_rebench],
        ["swaa_exact", "delta_corrected", "delta_imputed_failure", "uncorrected_rebench"],
        default="uncorrected_no_delta",   # HCAST runs on tasks with no successful baseline
    )
    return human_runs_derived[DERIVED_OUTPUT_COLUMNS]

human_runs_derived = build_derived_runs(human_runs, delta_by_task)
print(human_runs_derived["derivation"].value_counts().to_string())

derived_path = DATA_OUT / "human_runs_derived.csv"   # v1.0 — what Stage 2+ consume
human_runs_derived.to_csv(derived_path, index=False)
print(f"\nwrote {derived_path}  ({len(human_runs_derived)} rows)")

# %% [markdown]
# ## 6. Before/after: reconstruction error of `human_minutes`
#
# The point of the correction: re-deriving `human_minutes` from the export now succeeds
# exactly on every δ-corrected task, where the naive floored reconstruction missed by up
# to ~20% on short tasks. HCAST only — this is the left tail of §1's right panel, fixed.

# %%
naive_gmean_by_task = successful_hcast_baseline_runs.groupby("task_id").apply(
    lambda d: stats.gmean(d["minutes_floored"].clip(lower=1e-9)), include_groups=False)
corrected_gmean_by_task = successful_hcast_runs_with_delta.groupby("task_id").apply(
    lambda d: stats.gmean(d["minutes_floored"] + d["delta_task"]), include_groups=False)
official_human_minutes_by_task = successful_hcast_baseline_runs.groupby("task_id")[
    "human_minutes"].first()

ratio_by_task_reconstruction = pd.DataFrame({
    "naive (floored)": naive_gmean_by_task / official_human_minutes_by_task,
    "δ-corrected": corrected_gmean_by_task / official_human_minutes_by_task,
}).reset_index().melt(id_vars="task_id", var_name="reconstruction", value_name="ratio")
fig = px.ecdf(
    ratio_by_task_reconstruction, x="ratio", color="reconstruction",
    color_discrete_sequence=["#eb6834", "#2a78d6"],
    title=("Reconstructed / official human_minutes — before and after δ correction"
           f"<br><sup>HCAST only ({len(naive_gmean_by_task)} baselined tasks); "
           "SWAA already exact, RE-Bench not correctable — see §1</sup>"),
    labels={"ratio": "reconstructed ÷ official human_minutes"},
)
fig.update_xaxes(range=[0.75, 1.05])
fig.update_layout(margin=dict(t=95))
fig.write_image(FIGURES / "before_after_ecdf.png", width=750, height=470, scale=2)
show(fig)

# %% [markdown]
# ## 7. Replication on Time Horizon v1.1
#
# v1.0 is the paper's suite. METR later published **v1.1**, which re-cuts the human
# baseline data: the same 467 HCAST human runs are mapped onto a finer set of task ids
# (94 baselined tasks instead of 81), and 20 RE-Bench runs are dropped. If the flooring
# story is a fact about the export pipeline rather than an artifact of one snapshot, it
# should reproduce there untouched — and it does, on all four checks.

# %%
def run_export_archaeology(dataset_version: str) -> dict:
    """The whole §1–§5 pipeline for one suite version."""
    human_runs = load_human_runs(dataset_version)
    successful_baseline_runs = human_runs.query(
        "score_binarized == 1 and human_source == 'baseline'"
    )
    single_baseliner = successful_baseline_runs.groupby("task_id").filter(lambda g: len(g) == 1)
    successful_hcast = successful_baseline_runs.query("task_source == 'HCAST'")
    bracket = bracket_by_task_of(successful_hcast)
    delta = delta_by_task_of(successful_hcast)
    with_delta = successful_hcast.merge(delta, on="task_id")
    residual = with_delta.groupby("task_id").apply(
        lambda d: stats.gmean(d["minutes_floored"] + d["delta_task"]) / d["human_minutes"].iloc[0],
        include_groups=False,
    )
    swaa_ratio = successful_baseline_runs.query("task_source == 'SWAA'").groupby("task_id").apply(
        lambda d: stats.gmean(d["minutes_floored"]) / d["human_minutes"].iloc[0],
        include_groups=False,
    )
    return {
        "human runs": len(human_runs),
        "HCAST runs whole-minute": "{:.0%}".format(
            np.isclose(human_runs.query("task_source == 'HCAST'")["minutes_floored"] % 1, 0).mean()),
        "single-baseliner tasks": len(single_baseliner),
        "…stored == floor(human_minutes)": "{:.0f} / {:.0f}".format(
            np.isclose(single_baseliner["minutes_floored"],
                       np.floor(single_baseliner["human_minutes"])).sum(), len(single_baseliner)),
        "HCAST baselined tasks": len(bracket),
        "…bracket holds": f"{bracket['in_bracket'].sum()} / {len(bracket)}",
        "max |gmean(t+δ)/official − 1|": f"{np.abs(residual - 1).max():.1e}",
        "max SWAA |gmean/official − 1|": f"{np.abs(swaa_ratio - 1).max():.1e}",
        "_derived": build_derived_runs(human_runs, delta),
    }

results_by_dataset_version = {v: run_export_archaeology(v) for v in ("1-0", "1-1")}
checks_per_dataset_version = pd.DataFrame(
    {f"v{v.replace('-', '.')}": {k: r for k, r in result.items() if not k.startswith("_")}
     for v, result in results_by_dataset_version.items()}
)
print(checks_per_dataset_version.to_string())

derived_path_v1_1 = DATA_OUT / "human_runs_derived_v1_1.csv"
results_by_dataset_version["1-1"]["_derived"].to_csv(derived_path_v1_1, index=False)
print(f"\nwrote {derived_path_v1_1}  "
      f"({len(results_by_dataset_version['1-1']['_derived'])} rows)")
print(results_by_dataset_version["1-1"]["_derived"]["derivation"].value_counts().to_string())

# %% [markdown]
# ## Caveats
#
# 1. **δ is an imputation, not a measurement.** It is a per-task constant by
#    construction — the per-run sub-minute residuals are lost, and δ only restores the
#    task-level aggregate. Any downstream analysis needing within-task time *spread* at
#    sub-minute resolution should treat δ-corrected times as interval data `[t, t+1)`.
# 2. **Failures are best-guess** (`delta_imputed_failure`): δ is identified from
#    successes only.
# 3. **RE-Bench is uncorrected**: its `human_minutes` is not the gmean of raw elapsed
#    times (8h-capped sessions; threshold = mean of 7–9h runs), so no δ ∈ [0,1) exists.
# 4. This documents the **public export**, not METR's internal analysis — `human_minutes`
#    was computed from the unfloored times upstream and has full precision throughout;
#    the information loss is only in the shipped per-run times.
