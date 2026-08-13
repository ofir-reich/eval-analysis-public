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
# # Stage 3 — Failed human baselines as right-censored observations
#
# `human_minutes` is the geometric mean of the completion times of the baseliners who
# **succeeded**. The baseliners who *failed* are discarded — but the export still records
# how long they worked. A failure after 8 hours is not missing data: it is the
# observation **"this person's completion time exceeds 8 hours"** — a classic
# **right-censored** observation.
#
# Dropping them is complete-case analysis of censored data, and it biases the estimated
# time *downwards*, because the discarded attempts are exactly the slow ones. Nobody has
# done this correction: METR's
# [limitations note](https://metr.org/notes/2026-01-22-time-horizon-limitations/)
# explicitly lists it as open, and Hamilton's Weibull survival work is about *agent*
# runs, not human baselines.
#
# **Why this matters for the headline.** Stage 2 found the x-axis barely moves the trend,
# with one exception: the errors-in-variables bias, where SIMEX *lowers* horizons by
# 26–36%. Censoring correction pushes the **opposite** way — it *raises* `human_minutes`
# on exactly the hardest tasks. So the two corrections partly cancel, and the net effect
# is the interesting number.
#
# We work in three escalating tiers of assumption:
#
# | tier | method | assumption |
# |---|---|---|
# | 1 | **hard bounds** — gmean of censoring times is a lower bound on the true gmean | none |
# | 2 | **Kaplan–Meier** — nonparametric survival of completion time | non-informative censoring |
# | 3 | **censored-lognormal MLE** with σ̃ from [Stage 2's σ analysis](../stage-2-measurement-error/) | lognormal, σ known |
#
# then propagate the corrected `human_minutes` through METR's own fit.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from scipy import stats
from scipy.optimize import minimize_scalar

HERE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
STAGE_1 = HERE / ".." / "stage-1-export-archaeology"
STAGE_2 = HERE / ".." / "stage-2-measurement-error"
sys.path.insert(0, str(STAGE_2.resolve()))   # reuse the Stage-2 fit machinery
import metr_fit_helpers as metr  # noqa: E402

plotly.io.templates.default = "plotly_white"
pd.set_option("display.max_columns", None, "display.width", 220)

FIGURES = HERE / "figures"
DATA_OUT = HERE / "data"
FIGURES.mkdir(exist_ok=True), DATA_OUT.mkdir(exist_ok=True)
SUFFIX = metr.VERSION_SUFFIX
print(f"dataset version: {metr.DATASET_VERSION}")

show = metr.show   # inline display in interactive sessions; no-op headless

MIN_MINUTES = 10 / 60          # safety floor for log(time); see the clip note below
LENGTH_BUCKET_EDGES = [0, 1, 4, 16, 64, 256, 960, np.inf]
LENGTH_BUCKET_LABELS = ["<1m", "1-4m", "4-16m", "16-64m", "64-256m", "256-960m", ">960m"]

# %% [markdown]
# ## The human baseline runs, successes *and* failures
#
# Durations come straight from the export's `started_at`/`completed_at` on `alias ==
# "human"` rows. HCAST/RE-Bench timestamps carry the whole-minute flooring documented in
# [Stage 1](../stage-1-export-archaeology/), so we add that stage's per-task δ to put
# successes and censoring times on the same footing as `human_minutes`.

# %%
runs = metr.load_runs()
human_runs = runs.query("agent == 'human'").copy()   # metr.load_runs renames alias->agent
human_runs["minutes_floored"] = (
    human_runs["completed_at"] - human_runs["started_at"]
) / 60_000.0
# per-task δ from Stage 1 (keyed by task_id, so independent of that file's row order);
# δ applies only to the minute-floored sources — SWAA timestamps are exact to the ms.
is_minute_floored = human_runs["task_source"] != "SWAA"
delta_by_task = (
    pd.read_csv(STAGE_1 / "data" / f"human_runs_derived{SUFFIX}.csv")
    .groupby("task_id")["delta_task"].first()
)
human_runs["minutes_derived"] = human_runs["minutes_floored"] + np.where(
    is_minute_floored, human_runs["task_id"].map(delta_by_task).fillna(0.0), 0.0
)
# A minute-floored run recorded as 0 means "took under a minute". Where Stage 1 solved a
# δ the correction already makes it positive; the 10-second floor is a safety mechanism
# for the remaining runs (tasks with no solvable δ: all-failure HCAST tasks and
# RE-Bench), needed only to keep log(time) finite in the fits below. SWAA is exempt —
# its times are genuinely sub-minute and clipping them would inflate the shortest tasks.
human_runs["minutes_derived"] = human_runs["minutes_derived"].where(
    ~is_minute_floored, human_runs["minutes_derived"].clip(lower=MIN_MINUTES)
)
human_runs["is_success"] = human_runs["score_binarized"] == 1

# Validation: on tasks where every baseliner succeeded, the gmean of our independently
# derived times must reproduce the published human_minutes (that is Stage 1's result).
fully_successful_baseline_tasks = human_runs.groupby("task_id").filter(
    lambda task_runs: task_runs["is_success"].all()
    and task_runs["human_source"].iloc[0] == "baseline"
)
reconstruction_ratio_by_task = fully_successful_baseline_tasks.groupby("task_id").apply(
    lambda task_runs: stats.gmean(task_runs["minutes_derived"])
    / task_runs["human_minutes"].iloc[0], include_groups=False
)
print(f"reconstruction check on {len(reconstruction_ratio_by_task)} fully-successful "
      f"baseline tasks: max |ratio-1| = "
      f"{np.abs(reconstruction_ratio_by_task - 1).max():.2e}")
assert np.abs(reconstruction_ratio_by_task - 1).max() < 2e-3, \
    "derived times must reproduce published human_minutes on uncensored tasks"

runs_per_is_success_by_task_source = pd.crosstab(
    human_runs["task_source"], human_runs["is_success"]
)
runs_per_is_success_by_task_source["censoring_rate"] = (
    runs_per_is_success_by_task_source[False]
    / runs_per_is_success_by_task_source.sum(axis=1)
)
print("human baseline runs by source (False = failed = right-censored):")
print(runs_per_is_success_by_task_source.round(3).to_string())
print(f"\ntotal: {len(human_runs)} runs, {(~human_runs['is_success']).sum()} censored "
      f"({100 * (~human_runs['is_success']).mean():.1f}%)")

# %% [markdown]
# Censoring is not a rounding error: **38% of HCAST and 48% of RE-Bench** baseline runs
# ended in failure. And the censoring times are long — the failures are not quick
# give-ups:

# %%
censored_runs = human_runs.query("not is_success")
print("censoring-time (minutes) distribution:")
print(censored_runs["minutes_derived"].describe(
    percentiles=[0.25, 0.5, 0.75, 0.95]).round(1).to_string())
print(f"\ncensored runs beyond 8 hours: {(censored_runs['minutes_derived'] > 480).sum()}")
print("most common floored censoring times (an 8h administrative cap is visible):")
print(censored_runs["minutes_floored"].round().value_counts().head(5).to_string())
print(f"\ntotal discarded human effort: "
      f"{censored_runs['minutes_derived'].sum() / 60:,.0f} hours")

# %% [markdown]
# ## Censoring rises steeply with task length
#
# This is the crux: the tasks whose `human_minutes` is most affected are the **long**
# ones — the tasks that determine frontier agents' horizons.

# %%
task_level = human_runs.groupby("task_id").agg(
    n_runs=("is_success", "size"),
    n_successes=("is_success", "sum"),
    human_minutes=("human_minutes", "first"),
    task_source=("task_source", "first"),
    human_source=("human_source", "first"),
)
task_level["n_censored"] = task_level["n_runs"] - task_level["n_successes"]
task_level["length_bucket"] = pd.cut(
    task_level["human_minutes"], LENGTH_BUCKET_EDGES, labels=LENGTH_BUCKET_LABELS
)

success_rate_by_length_bucket = task_level.groupby(
    "length_bucket", observed=True
).apply(lambda bucket: pd.Series({
    "n_tasks": len(bucket),
    "n_runs": bucket["n_runs"].sum(),
    "human_success_rate": bucket["n_successes"].sum() / bucket["n_runs"].sum(),
}), include_groups=False)
print(success_rate_by_length_bucket.round(3).to_string())

fig = px.bar(
    success_rate_by_length_bucket.reset_index(),
    x="length_bucket", y="human_success_rate", text="n_runs",
    title="Human baseliner success rate collapses on long tasks<br>"
          "<sub>bar label = number of baseline runs</sub>",
    labels={"human_success_rate": "share of baseline runs that succeeded",
            "length_bucket": "task length (published human_minutes)"},
)
fig.update_traces(textposition="outside")
fig.update_yaxes(range=[0, 1.05])
fig.write_image(FIGURES / f"success_rate_by_length{SUFFIX}.png",
                width=850, height=470, scale=2)
show(fig)

# %% [markdown]
# ## The structural finding: "estimate" tasks *are* the all-failure tasks
#
# 16 tasks have **zero** successful baseline runs. Every one of them is exactly a task
# whose `human_minutes` carries `human_source == "estimate"` — and conversely, every
# HCAST estimate task is one of these 16. METR fell back to a researcher's estimate
# precisely when **all** baseliners failed.
#
# Two things follow. First, the estimates are not exogenous guesses — baselining was
# attempted on every one of these tasks, and the estimate is the fallback. Second, they
# stand in for tasks humans **could not finish**, and the export contains the record of
# how long people tried.

# %%
all_censored_tasks = task_level.query("n_successes == 0")
print(f"tasks with zero successful baselines: {len(all_censored_tasks)}")
print("their human_source:", all_censored_tasks["human_source"].value_counts().to_dict())
hcast_estimate_tasks = task_level.query(
    "human_source == 'estimate' and task_source == 'HCAST'"
)
print(f"HCAST estimate tasks: {len(hcast_estimate_tasks)} — "
      f"identical set: {set(all_censored_tasks.index) == set(hcast_estimate_tasks.index)}")
print("\n(the 3 RE-Bench estimate tasks DO have successful runs — "
      "their `human_minutes` is the 480-min budget, treated separately)")

# %% [markdown]
# ## Tier 1 — assumption-free lower bounds
#
# If every baseliner on a task failed, with censoring times $c_1 \dots c_n$, then each
# true completion time satisfies $T_i > c_i$, so **any** summary that is monotone in the
# $T_i$ is bounded below by the same summary of the $c_i$. In particular
# $\mathrm{gmean}(T) > \mathrm{gmean}(c)$. This needs no distributional
# model, and it can be checked directly against the published number.

# %%
censoring_bound_rows = []
for task_id, task_censored_runs in censored_runs.groupby("task_id"):
    if task_id not in all_censored_tasks.index:
        continue
    censoring_times = task_censored_runs["minutes_derived"].to_numpy()
    censoring_bound_rows.append({
        "task_id": task_id,
        "n_censored": len(censoring_times),
        "published_human_minutes": all_censored_tasks.loc[task_id, "human_minutes"],
        "gmean_censoring_lower_bound": stats.gmean(censoring_times),
        "max_censoring_time": censoring_times.max(),
    })
bound_by_all_censored_task = pd.DataFrame(censoring_bound_rows).set_index("task_id")
bound_by_all_censored_task["bound_over_published"] = (
    bound_by_all_censored_task["gmean_censoring_lower_bound"]
    / bound_by_all_censored_task["published_human_minutes"]
)
bound_by_all_censored_task = bound_by_all_censored_task.sort_values(
    "bound_over_published", ascending=False
)
print(bound_by_all_censored_task.round(2).to_string())

n_understated_by_bound = (bound_by_all_censored_task["bound_over_published"] > 1).sum()
print(f"\ntasks whose published value is below the hard lower bound: "
      f"{n_understated_by_bound} / {len(bound_by_all_censored_task)}")
print("median understatement factor among those: "
      f"{bound_by_all_censored_task.query('bound_over_published > 1')['bound_over_published'].median():.2f}×")
print("tasks where a single baseliner already worked longer than the published estimate: "
      f"{(bound_by_all_censored_task['max_censoring_time'] > bound_by_all_censored_task['published_human_minutes']).sum()}"
      f" / {len(bound_by_all_censored_task)}")

# %%
bound_plot_data = bound_by_all_censored_task.reset_index()
fig = px.scatter(
    bound_plot_data, x="published_human_minutes", y="gmean_censoring_lower_bound",
    size="n_censored", hover_name="task_id", log_x=True, log_y=True,
    title="All-failure tasks: hard lower bound vs published estimate<br>"
          "<sub>above the dashed line = the published value is below the hard lower "
          "bound</sub>",
    labels={"published_human_minutes": "published human_minutes (researcher estimate)",
            "gmean_censoring_lower_bound": "gmean of censoring times (hard lower bound)"},
)
diagonal = np.array([bound_plot_data[["published_human_minutes",
                                      "gmean_censoring_lower_bound"]].min().min() * 0.7,
                     bound_plot_data[["published_human_minutes",
                                      "gmean_censoring_lower_bound"]].max().max() * 1.4])
fig.add_scatter(x=diagonal, y=diagonal, mode="lines",
                line=dict(dash="dash", color="gray"), name="y = x", hoverinfo="skip")
fig.write_image(FIGURES / f"lower_bounds_all_censored{SUFFIX}.png",
                width=800, height=560, scale=2)
show(fig)

# %% [markdown]
# ## Tier 2 — Kaplan–Meier
#
# Pool the baseline runs within a task-length bucket and estimate the survival function
# of completion time treating failures as right-censored (Kaplan–Meier). This needs no
# distributional assumption, only that censoring is non-informative. Comparing the KM
# median against the naive geometric mean of successes gives a **model-free** measure of
# how much discarding failures understates the time — which we can then check against the
# parametric Tier-3 correction.
#
# (Caveat: a bucket pools tasks of differing true length, so these are coarse
# bucket-level statistics, not per-task estimates.)

# %%
def kaplan_meier(durations: np.ndarray, is_event: np.ndarray) -> pd.DataFrame:
    """Kaplan–Meier estimate of S(t) = P(completion time > t).
    `is_event` marks observed completions; False marks right-censored runs."""
    order = np.argsort(durations)
    durations, is_event = durations[order], is_event[order]
    n_at_risk = len(durations)
    survival, rows = 1.0, [{"time": 0.0, "survival": 1.0, "n_at_risk": n_at_risk}]
    for time in np.unique(durations[is_event]):
        n_events = int(((durations == time) & is_event).sum())
        at_risk = int((durations >= time).sum())
        survival *= 1 - n_events / at_risk
        rows.append({"time": time, "survival": survival, "n_at_risk": at_risk})
    return pd.DataFrame(rows)


kaplan_meier_curves, kaplan_meier_summary_rows = [], []
for length_bucket, bucket_runs in human_runs.assign(
    length_bucket=human_runs["task_id"].map(task_level["length_bucket"])
).groupby("length_bucket", observed=True):
    curve = kaplan_meier(
        bucket_runs["minutes_derived"].to_numpy(),
        bucket_runs["is_success"].to_numpy(),
    )
    curve["length_bucket"] = str(length_bucket)
    kaplan_meier_curves.append(curve)

    reached_median = curve.query("survival <= 0.5")
    naive_gmean = stats.gmean(bucket_runs.query("is_success")["minutes_derived"])
    kaplan_meier_summary_rows.append({
        "length_bucket": str(length_bucket),
        "n_runs": len(bucket_runs),
        "kaplan_meier_median": reached_median["time"].iloc[0] if len(reached_median) else np.nan,
        "naive_gmean_successes": naive_gmean,
        "lowest_survival": curve["survival"].min(),
    })
kaplan_meier_by_length_bucket = pd.concat(kaplan_meier_curves)
kaplan_meier_by_length_bucket["length_bucket"] = pd.Categorical(
    kaplan_meier_by_length_bucket["length_bucket"], LENGTH_BUCKET_LABELS, ordered=True
)
kaplan_meier_summary = pd.DataFrame(kaplan_meier_summary_rows).set_index("length_bucket")
kaplan_meier_summary = kaplan_meier_summary.reindex(
    [label for label in LENGTH_BUCKET_LABELS if label in kaplan_meier_summary.index]
)
kaplan_meier_summary["km_median_over_naive_gmean"] = (
    kaplan_meier_summary["kaplan_meier_median"]
    / kaplan_meier_summary["naive_gmean_successes"]
)
print("KM median vs the naive gmean of successes (model-free view of the bias):")
print(kaplan_meier_summary.round(3).to_string())
print("\nIn the well-populated middle buckets (4-16m through 64-256m) the KM median runs "
      "\n~1.5× the naive gmean — independent, distribution-free corroboration of the "
      "\nparametric correction below. The 256-960m bucket dips under 1.0 because pooling "
      "\nspans a 4× range of true task lengths there, so read these as coarse checks.")

fig = px.line(
    kaplan_meier_by_length_bucket.sort_values(["length_bucket", "time"]),
    x="time", y="survival", color="length_bucket",
    line_shape="hv", log_x=True,
    category_orders={"length_bucket": LENGTH_BUCKET_LABELS},
    title="Kaplan–Meier survival of human completion time<br>"
          "<sub>failures treated as right-censored; by task-length bucket</sub>",
    labels={"time": "minutes (log scale)",
            "survival": "P(not yet completed)", "length_bucket": "task length"},
)
fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
fig.write_image(FIGURES / f"kaplan_meier_by_length{SUFFIX}.png",
                width=880, height=520, scale=2)
show(fig)

# %% [markdown]
# ## Tier 3 — censored-lognormal MLE
#
# To get a *point estimate* per task we need one distributional assumption. Take
# completion time on task $j$ to be lognormal with location $\mu_j$ and spread
# $\sigma_j$ — and take $\sigma_j$ from **Stage 2's σ analysis**, which estimated it by
# empirical-Bayes shrinkage (HCAST complete-pools to σ̃ ≈ 0.89). That is what makes this
# tractable: with σ known, $\mu_j$ is a one-parameter fit, stable even on a task with a
# single success and several censored runs.
#
# $$\ell(\mu_j) = \sum_{\text{successes}} \log \phi\!\left(\tfrac{\ln t_i - \mu_j}{\sigma_j}\right)
#  + \sum_{\text{censored}} \log \Phi\!\left(\tfrac{\mu_j - \ln c_k}{\sigma_j}\right)$$
#
# The censored terms only ever push $\mu_j$ **up**, so the corrected `human_minutes`
# $= e^{\hat\mu_j}$ is always ≥ the naive gmean of successes. Note the identifiability
# limit: with **no** successes the likelihood increases without bound in $\mu_j$, so the
# 16 all-failure tasks have no MLE — for them Tier 1's bound is all the data supports.

# %%
sigma_by_task = pd.read_csv(
    STAGE_2 / "data" / f"sigma_by_task{SUFFIX}.csv", index_col=0
)
shrunken_sigma_by_task = sigma_by_task["log_time_std_shrunken"].fillna(
    sigma_by_task["sem_log"]
)


def censored_lognormal_location(
    success_times: np.ndarray, censoring_times: np.ndarray, sigma: float
) -> float:
    """MLE of μ for lognormal completion time with σ fixed; returns exp(μ) (the gmean).
    Requires at least one success (otherwise the likelihood is unbounded in μ)."""
    log_success, log_censor = np.log(success_times), np.log(censoring_times)

    def negative_log_likelihood(mu: float) -> float:
        total = float(np.sum(stats.norm.logpdf((log_success - mu) / sigma)))
        if len(log_censor):
            total += float(np.sum(stats.norm.logcdf((mu - log_censor) / sigma)))
        return -total

    search_low = min(log_success.min(), log_censor.min() if len(log_censor) else np.inf)
    search_high = max(log_success.max(), log_censor.max() if len(log_censor) else -np.inf)
    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(search_low - 5 * sigma, search_high + 10 * sigma),
        method="bounded",
    )
    return float(np.exp(result.x))


mle_rows = []
for task_id, task_runs in human_runs.groupby("task_id"):
    success_times = task_runs.query("is_success")["minutes_derived"].to_numpy()
    censoring_times = task_runs.query("not is_success")["minutes_derived"].to_numpy()
    sigma = shrunken_sigma_by_task.get(task_id, np.nan)
    naive_gmean = stats.gmean(success_times) if len(success_times) else np.nan
    corrected = (
        censored_lognormal_location(success_times, censoring_times, sigma)
        if len(success_times) and np.isfinite(sigma) else np.nan
    )
    mle_rows.append({
        "task_id": task_id,
        "n_successes": len(success_times),
        "n_censored": len(censoring_times),
        "sigma_used": sigma,
        "naive_gmean_successes": naive_gmean,
        "censored_mle_gmean": corrected,
        "published_human_minutes": task_runs["human_minutes"].iloc[0],
        "task_source": task_runs["task_source"].iloc[0],
        "human_source": task_runs["human_source"].iloc[0],
    })
survival_by_task = pd.DataFrame(mle_rows).set_index("task_id")
survival_by_task["mle_over_naive"] = (
    survival_by_task["censored_mle_gmean"] / survival_by_task["naive_gmean_successes"]
)

corrected_tasks = survival_by_task.query("n_censored > 0 and n_successes > 0")
print(f"tasks with a censored-MLE correction: {len(corrected_tasks)}")
print("\ncorrection factor (MLE ÷ naive gmean of successes):")
print(corrected_tasks["mle_over_naive"].describe(
    percentiles=[0.25, 0.5, 0.75, 0.95]).round(3).to_string())
print("\nby source:")
print(corrected_tasks.groupby("task_source")["mle_over_naive"]
      .agg(["size", "median", "max"]).round(3).to_string())

# %%
correction_plot_data = survival_by_task.query("n_censored > 0 and n_successes > 0").reset_index()
fig = px.scatter(
    correction_plot_data, x="published_human_minutes", y="mle_over_naive",
    color="task_source", size="n_censored", hover_name="task_id", log_x=True,
    title="Censoring correction factor per task<br>"
          "<sub>1.0 = unchanged; higher = published human_minutes more understated</sub>",
    labels={"published_human_minutes": "published human_minutes (log scale)",
            "mle_over_naive": "censored MLE ÷ naive gmean"},
)
fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
fig.write_image(FIGURES / f"correction_factor_by_task{SUFFIX}.png",
                width=880, height=520, scale=2)
show(fig)

# %% [markdown]
# The largest corrections are the most informative cases. `blackbox/acorn` carries a
# published `human_minutes` of 12.0 — the time of its **single** successful baseliner —
# while **six of its nine failed baseliners worked longer than 12 minutes** without
# solving it. The recorded attempts alone show that 12 minutes understates the typical
# completion time; the MLE's 113 minutes is one defensible answer.
#
# It also exposes this tier's main risk: σ is *fixed* at Stage 2's pooled 0.89, but a
# task where one person finishes in 12 min and others exceed 139 min plainly has a larger
# spread. With σ pinned too low, the fit can only reconcile the censored runs by pushing
# μ up. So we check how much the headline depends on that borrowed σ.

# %% [markdown]
# ## Propagating to horizons and the doubling time
#
# Six x-axis scenarios, each fed through METR's own weighted-logistic fit:
#
# | scenario | `human_minutes` used |
# |---|---|
# | `published` | METR's values (reproduces the headline exactly) |
# | `max_impute` | crude: give every failed run its task's longest observed time, re-gmean |
# | `censored_mle` | Tier 3 MLE **level** where identified; published elsewhere |
# | `censored_mle + bounds` | as above, plus all-failure tasks raised to their Tier-1 bound |
# | `censored_ratio` | published × (Tier-3 MLE ÷ naive gmean of successes) |
# | `censored_ratio + bounds` | as above, plus the Tier-1 floors |
#
# The two families answer different questions. For HCAST and SWAA they coincide, because
# the published value *is* the gmean of successful run times (Stage 1). For **RE-Bench**
# they do not: its published `human_minutes` follows a different time convention
# (§Stage 1 — not a gmean of elapsed times), so `censored_mle` substituting the
# elapsed-time MLE level there bundles the censoring correction together with a
# convention switch. `censored_ratio` keeps every task in its published convention and
# applies only the *relative* censoring effect — the cleaner "pure censoring" scenario —
# while `censored_mle` shows what a fully elapsed-time-consistent x-axis would look like.

# %%
published_human_minutes_by_task = metr.official_human_minutes_by_task(runs)

max_impute_by_task = {}
for task_id, task_runs in human_runs.groupby("task_id"):
    times = task_runs["minutes_derived"].to_numpy()
    imputed = np.where(task_runs["is_success"].to_numpy(), times, times.max())
    max_impute_by_task[task_id] = stats.gmean(imputed)
max_impute_human_minutes_by_task = pd.Series(max_impute_by_task).reindex(
    published_human_minutes_by_task.index
).fillna(published_human_minutes_by_task)

censored_mle_human_minutes_by_task = (
    survival_by_task["censored_mle_gmean"]
    .reindex(published_human_minutes_by_task.index)
    .fillna(published_human_minutes_by_task)
)
# only raise, never lower: the correction is one-directional by construction
censored_mle_human_minutes_by_task = np.maximum(
    censored_mle_human_minutes_by_task, published_human_minutes_by_task
)

censoring_ratio_by_task = (
    survival_by_task["mle_over_naive"].clip(lower=1)   # one-directional, like the MLE
    .reindex(published_human_minutes_by_task.index).fillna(1)
)
censored_ratio_human_minutes_by_task = (
    published_human_minutes_by_task * censoring_ratio_by_task
)


def raised_to_tier1_bounds(human_minutes_by_task: pd.Series) -> pd.Series:
    """All-failure tasks raised to their Tier-1 arithmetic floor."""
    with_bounds = human_minutes_by_task.copy()
    for task_id, bound_row in bound_by_all_censored_task.iterrows():
        with_bounds[task_id] = max(
            with_bounds[task_id], bound_row["gmean_censoring_lower_bound"]
        )
    return with_bounds


human_minutes_by_scenario = {
    "published": published_human_minutes_by_task,
    "max_impute": max_impute_human_minutes_by_task,
    "censored_mle": censored_mle_human_minutes_by_task,
    "censored_mle + bounds": raised_to_tier1_bounds(censored_mle_human_minutes_by_task),
    "censored_ratio": censored_ratio_human_minutes_by_task,
    "censored_ratio + bounds": raised_to_tier1_bounds(censored_ratio_human_minutes_by_task),
}
print("median ×change in human_minutes vs published, and change on long tasks:")
is_long_task = published_human_minutes_by_task > 60
for scenario, scenario_human_minutes in human_minutes_by_scenario.items():
    ratio = scenario_human_minutes / published_human_minutes_by_task
    print(f"  {scenario:24} all ×{stats.gmean(ratio):.3f} | "
          f">60min tasks ×{stats.gmean(ratio[is_long_task]):.3f}")

# %%
frontier_agents, release_dates, official_fits_by_agent = (
    metr.load_frontier_agents_and_dates()
)
years_since_release_by_agent = metr.years_since_first_release_by_agent(
    frontier_agents, release_dates
)

scenario_rows = []
for scenario, scenario_human_minutes in human_minutes_by_scenario.items():
    horizon_per_quantile_by_agent = metr.fit_horizons_per_quantile_by_agent(
        runs, scenario_human_minutes
    )
    for quantile, horizon_by_agent in horizon_per_quantile_by_agent.items():
        frontier_horizon = pd.Series(horizon_by_agent).reindex(frontier_agents).dropna()
        scenario_rows.append({
            "scenario": scenario,
            "quantile": f"p{int(quantile * 100)}",
            "gmean_frontier_horizon_min": float(np.exp(np.log(frontier_horizon).mean())),
            "newest_agent_horizon_min": horizon_by_agent.get(
                max(years_since_release_by_agent,
                    key=years_since_release_by_agent.get), np.nan),
            "doubling_days": metr.fit_doubling_days(
                horizon_by_agent, years_since_release_by_agent),
        })
scenario_summary = pd.DataFrame(scenario_rows)
published_by_quantile = scenario_summary.query("scenario == 'published'").set_index("quantile")
for column in ["gmean_frontier_horizon_min", "newest_agent_horizon_min", "doubling_days"]:
    scenario_summary[f"{column}_vs_published"] = scenario_summary.apply(
        lambda row: row[column] / published_by_quantile.loc[row["quantile"], column], axis=1
    )
print(scenario_summary.round(3).to_string(index=False))
scenario_summary.to_csv(DATA_OUT / f"scenario_summary{SUFFIX}.csv", index=False, float_format=metr.CSV_FLOAT_FORMAT)
survival_by_task.to_csv(DATA_OUT / f"survival_by_task{SUFFIX}.csv", float_format=metr.CSV_FLOAT_FORMAT)
bound_by_all_censored_task.to_csv(DATA_OUT / f"lower_bounds_all_censored{SUFFIX}.csv", float_format=metr.CSV_FLOAT_FORMAT)

# %%
SCENARIO_ORDER = ["published", "censored_ratio", "censored_ratio + bounds",
                  "censored_mle", "censored_mle + bounds", "max_impute"]

fig = px.scatter(
    scenario_summary, x="scenario", y="gmean_frontier_horizon_min_vs_published",
    color="quantile", symbol="quantile",
    category_orders={"scenario": SCENARIO_ORDER},
    title="Time horizon under censoring correction<br>"
          "<sub>1.0 = METR's published value; geometric mean over the "
          "SOTA-at-release agents</sub>",
    labels={"gmean_frontier_horizon_min_vs_published": "horizon ÷ published horizon",
            "scenario": ""},
)
fig.update_traces(marker=dict(size=11))
fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
fig.write_image(FIGURES / f"horizon_by_scenario{SUFFIX}.png", width=880, height=500, scale=2)
show(fig)

# %%
fig = px.scatter(
    scenario_summary, x="scenario", y="doubling_days",
    color="quantile", symbol="quantile",
    category_orders={"scenario": SCENARIO_ORDER},
    title="Doubling time under censoring correction<br>"
          "<sub>dotted lines mark the published values; note the narrow vertical "
          "range</sub>",
    labels={"doubling_days": "doubling time (days)", "scenario": ""},
)
fig.update_traces(marker=dict(size=11))
for quantile_label in ["p50", "p80"]:
    fig.add_hline(
        y=float(published_by_quantile.loc[quantile_label, "doubling_days"]),
        line_dash="dot", opacity=0.4,
    )
fig.write_image(FIGURES / f"doubling_by_scenario{SUFFIX}.png", width=880, height=500, scale=2)
show(fig)

# %% [markdown]
# ## How much rides on the borrowed σ?
#
# σ is the one parameter Tier 3 imports rather than estimates, so we refit everything with
# it scaled by 0.75×, 1×, and 1.5× (HCAST σ̃ 0.67 / 0.89 / 1.33).
#
# The correction turns out to *grow* with σ: a wider distribution makes each exact success
# weaker evidence about the location, so the censored observations dominate more and push
# μ higher. That matters for how to read the headline — σ̃ is estimated from **successful**
# runs only, so if failures are drawn from a heavier-tailed part of the distribution the
# true σ is *larger* than 0.89, and our central estimate is the **conservative** one.

# %%
# complete pooling in Stage 2's σ analysis gives every HCAST task the same σ̃ (= the prior s0);
# the median just reads that common value off the table for display
hcast_pooled_sigma = float(
    sigma_by_task.query("task_source == 'HCAST'")["log_time_std_shrunken"].median()
)
sigma_sensitivity_rows = []
for sigma_multiplier in [0.75, 1.0, 1.5]:
    corrected_by_task = {}
    for task_id, task_runs in human_runs.groupby("task_id"):
        success_times = task_runs.query("is_success")["minutes_derived"].to_numpy()
        censoring_times = task_runs.query("not is_success")["minutes_derived"].to_numpy()
        sigma = shrunken_sigma_by_task.get(task_id, np.nan) * sigma_multiplier
        if not len(success_times) or not np.isfinite(sigma):
            continue
        corrected_by_task[task_id] = censored_lognormal_location(
            success_times, censoring_times, sigma
        )
    corrected_human_minutes = np.maximum(
        pd.Series(corrected_by_task)
        .reindex(published_human_minutes_by_task.index)
        .fillna(published_human_minutes_by_task),
        published_human_minutes_by_task,
    )
    horizon_per_quantile_by_agent = metr.fit_horizons_per_quantile_by_agent(
        runs, corrected_human_minutes
    )
    frontier_p50 = pd.Series(horizon_per_quantile_by_agent[0.5]).reindex(
        frontier_agents).dropna()
    correction_ratio = corrected_human_minutes / published_human_minutes_by_task
    sigma_sensitivity_rows.append({
        "sigma_multiplier": sigma_multiplier,
        "hcast_sigma": hcast_pooled_sigma * sigma_multiplier,
        "median_correction_on_changed_tasks": float(
            correction_ratio[correction_ratio > 1.001].median()),
        "gmean_frontier_p50_min": float(np.exp(np.log(frontier_p50).mean())),
        "frontier_p50_vs_published": float(np.exp(np.log(frontier_p50).mean()))
        / float(published_by_quantile.loc["p50", "gmean_frontier_horizon_min"]),
    })
sigma_sensitivity = pd.DataFrame(sigma_sensitivity_rows)
print("Sensitivity of the censoring correction to the borrowed σ:")
print(sigma_sensitivity.round(3).to_string(index=False))
sigma_sensitivity.to_csv(DATA_OUT / f"sigma_sensitivity{SUFFIX}.csv", index=False, float_format=metr.CSV_FLOAT_FORMAT)

# %% [markdown]
# ## Net effect: censoring correction vs SIMEX
#
# [Stage 2's SIMEX](../stage-2-measurement-error/) found SIMEX *lowers* the newest frontier
# agent's p50 (−18.8% on v1.0 with per-task σ). Censoring correction pushes the other
# way. Composing them multiplicatively in log space gives the net revision to the
# headline horizon — the two largest known x-axis corrections, applied together for the
# first time. Both censoring variants are shown: `censored_ratio + bounds` (pure
# censoring, published conventions kept) and `censored_mle + bounds` (fully
# elapsed-time-consistent, which for RE-Bench also switches the time convention).
#
# The multiplicative composition is itself an approximation: it assumes the two
# corrections commute, whereas SIMEX rerun *on* a censoring-corrected x-axis (with σ
# re-estimated from it) would come out somewhat different. Read the net as indicative,
# not as a precise revised headline.

# %%
simex_corrections = pd.read_csv(
    STAGE_2 / "data" / f"simex_corrections{SUFFIX}.csv"
)
newest_frontier_agent = max(
    years_since_release_by_agent, key=years_since_release_by_agent.get
)
simex_per_task_p50 = simex_corrections.query(
    "noise_model == 'per_task' and quantile == 'p50' and agent == @newest_frontier_agent"
)["pct_change"].iloc[0]

print(f"newest frontier agent: {newest_frontier_agent}")
print(f"  SIMEX (Stage 2, per-task σ):              ×{1 + simex_per_task_p50 / 100:.3f} "
      f"({simex_per_task_p50:+.1f}%)")
for censoring_scenario in ["censored_ratio + bounds", "censored_mle + bounds"]:
    censoring_p50 = scenario_summary.query(
        "scenario == @censoring_scenario and quantile == 'p50'"
    )["newest_agent_horizon_min_vs_published"].iloc[0]
    net_factor = (1 + simex_per_task_p50 / 100) * censoring_p50
    print(f"  censoring, {censoring_scenario:24}: ×{censoring_p50:.3f} "
          f"({100 * (censoring_p50 - 1):+.1f}%)  →  net ×{net_factor:.3f} "
          f"({100 * (net_factor - 1):+.1f}%)")

# %% [markdown]
# ## Caveats
#
# 1. **Informative censoring.** Kaplan–Meier and the MLE both assume a baseliner's
#    decision to stop is independent of how much longer they would have needed. Real
#    give-ups are likely *not* independent — someone who senses a task is hopeless quits
#    earlier — which would make even these corrections **understate** the true times.
#    The 8-hour administrative cap (25 runs at 479 min) is genuinely non-informative;
#    voluntary give-ups are not.
# 2. **σ borrowed from successes.** Stage 2's σ̃ is estimated from *successful* runs
#    only. If failures come from a heavier-tailed part of the distribution, the true σ is
#    larger and the correction bigger.
# 3. **All-failure tasks are bounds, not estimates** — with no successes the location is
#    not identified, so `censored_mle + bounds` uses the Tier-1 arithmetic floor. The true
#    values are somewhere above it, by an unknown amount.
# 4. **The 10-second clip is inert.** A handful of runs are floored to 0 minutes
#    (v1.0: 4, of which 1 censored); where no δ exists they are clipped to 10 s purely
#    to keep log(time) finite. The one clipped *censored* run sits on an all-failure
#    task whose Tier-1 bound is ~0.07× the published value under any treatment of that
#    run — so the set of tasks understated per the Tier-1 bound does not depend on a
#    clipped value.
