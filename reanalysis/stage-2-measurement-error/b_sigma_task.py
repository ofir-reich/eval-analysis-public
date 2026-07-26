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
# # Stage 2(b) — Per-task σ via empirical-Bayes shrinkage
#
# Goal: an estimate of **σ_task** — the spread (SD) of log completion time across human
# baseliners — for *every* task, including the 21 tasks with a single successful baseline
# run, where no within-task spread is observable.
#
# **Model** (Smyth 2004, the *limma/eBayes* moderated-variance model from genomics, which
# faces the identical problem — thousands of genes, 2–3 replicates each):
# each task $j$ has a true log-time variance $\sigma_j^2$; the observed sample variance
# $s_j^2$ (from $n_j$ runs, $d_j = n_j - 1$ degrees of freedom) is scaled-$\chi^2$ noise
# around it; across tasks, $\sigma_j^2 \sim$ scaled-inverse-$\chi^2(d_0, s_0^2)$.
# The posterior mean is a precision-weighted average — the **shrinkage formula**:
#
# $$\tilde\sigma_j^2 = \frac{d_0 s_0^2 + d_j s_j^2}{d_0 + d_j}$$
#
# The prior strength $d_0$ (in pseudo-degrees-of-freedom) and location $s_0^2$ are
# **estimated from the ensemble of tasks by method of moments** on $\log s_j^2$: the
# observed dispersion of the $\log s_j^2$ decomposes into known $\chi^2$ sampling noise
# (trigamma functions of $d_j$) plus true between-task spread (trigamma of $d_0$).
# Tasks with $n_j = 1$ ($d_j = 0$) get exactly the prior $s_0^2$.
#
# Because the between-baseliner spread differs systematically by task source (HCAST
# tasks are far more variable than SWAA ones — see ECDF below), we fit the prior
# **per task source** rather than pooled.
#
# **Comparison target:** METR's [modelling-assumptions note
# (2026-03-20)](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)
# assumes a *global* natural-log noise scale — SEM² = 0.78²/n for baselined tasks,
# σ = 1.05 for researcher-estimate tasks — feeding their SIMEX correction
# (−26…−36% on Opus 4.6's p50). Here we check that global 0.78 against the per-task
# empirical values. All σ's below are in **natural log** units (divide by ln 2 ≈ 0.693
# for log2 units).

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from scipy import stats
from scipy.optimize import brentq
from scipy.special import digamma, polygamma

import metr_fit_helpers as metr

plotly.io.templates.default = "plotly_white"
pd.set_option("display.max_columns", None, "display.width", 200)

HERE = Path(__file__).parent if "__file__" in dir() else Path.cwd()
FIGURES = HERE / "figures"
DATA_OUT = HERE / "data"
FIGURES.mkdir(exist_ok=True), DATA_OUT.mkdir(exist_ok=True)
SUFFIX = metr.VERSION_SUFFIX   # "" for v1.0, "_v1_1" for v1.1
print(f"dataset version: {metr.DATASET_VERSION}")

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

METR_SIGMA_BASELINED = 0.78   # METR's global natural-log σ: SEM² = 0.78²/n
METR_SIGMA_ESTIMATE = 1.05    # METR's σ for researcher-estimate tasks

# %% [markdown]
# ## Data: per-task spread of log completion time
#
# Successful baseline runs with usable per-run times from
# [Stage 1](../stage-1-export-archaeology/): δ-corrected HCAST (`delta_corrected`),
# exact SWAA (`swaa_exact`), and RE-Bench (`uncorrected_rebench` — no δ was solvable
# because RE-Bench `human_minutes` is not a gmean of run times, but the floored times
# are hours-scale, so flooring is negligible for *spread*).

# %%
runs = pd.read_json(metr.REPORT / "data" / "raw" / "runs.jsonl", lines=True)
tasks = runs.groupby("task_id").agg(
    task_source=("task_source", "first"),
    human_source=("human_source", "first"),
    human_minutes=("human_minutes", "first"),
)
human_runs_derived = pd.read_csv(metr.derived_human_runs_csv())
successful_runs_with_derived_times = human_runs_derived.query("score_binarized == 1")
assert set(successful_runs_with_derived_times["derivation"]) == {
    "delta_corrected", "swaa_exact", "uncorrected_rebench"
}

log_time_spread_by_task = successful_runs_with_derived_times.groupby("task_id").agg(
    n_successful_runs=("minutes_derived", "size"),
    log_time_std=("minutes_derived", lambda times: np.log(times).std(ddof=1)),
)
sigma_by_task = tasks.join(log_time_spread_by_task)
sigma_by_task["n_successful_runs"] = (
    sigma_by_task["n_successful_runs"].fillna(0).astype(int)
)
sigma_by_task["degrees_of_freedom"] = (sigma_by_task["n_successful_runs"] - 1).clip(lower=0)

print(sigma_by_task.groupby(["human_source", "task_source"]).agg(
    n_tasks=("n_successful_runs", "size"),
    n_tasks_single_run=("degrees_of_freedom", lambda df: int((df == 0).sum())),
    median_log_time_std=("log_time_std", "median"),
))

# %% [markdown]
# The *raw* sample SDs differ by source and sit below METR's global σ = 0.78 at the
# median — but beware: with n = 2–4 runs, $s_j$ is a heavily downward-biased and noisy
# estimate of the true σ. Correcting that bias is exactly what the moment fit below
# does, and it changes the picture.

# %%
tasks_with_observed_spread = sigma_by_task.query("degrees_of_freedom >= 1").copy()
fig = px.ecdf(
    tasks_with_observed_spread,
    x="log_time_std",
    color="task_source",
    title="Between-baseliner spread of log completion time, by task source "
          f"(n≥2 tasks; dashed line: METR's global σ = {METR_SIGMA_BASELINED})",
    labels={"log_time_std": "sample SD of ln(completion minutes)"},
)
fig.add_vline(x=METR_SIGMA_BASELINED, line_dash="dash", line_color="gray")
fig.write_image(FIGURES / f"b_sigma_ecdf_by_source{SUFFIX}.png", width=900, height=500, scale=2)
show(fig)

# %% [markdown]
# Spread vs task length — is there a trend that would justify a larger σ for the long
# (frontier-driving) tasks?

# %%
fig = px.scatter(
    tasks_with_observed_spread.reset_index(),
    x="human_minutes", y="log_time_std",
    color="task_source", size="n_successful_runs",
    hover_name="task_id", log_x=True,
    title="Between-baseliner spread vs task length",
    labels={"human_minutes": "human_minutes (log scale)",
            "log_time_std": "sample SD of ln(completion minutes)"},
)
fig.add_hline(y=METR_SIGMA_BASELINED, line_dash="dash", line_color="gray")
fig.write_image(FIGURES / f"b_sigma_vs_length{SUFFIX}.png", width=900, height=500, scale=2)
show(fig)

# %% [markdown]
# ## Method-of-moments fit of the prior, per task source
#
# With $e_j = \log s_j^2$ and $z_j = e_j - \psi(d_j/2) + \log(d_j/2)$ (removing the
# known mean of $\log\chi^2$ sampling noise):
#
# - $\operatorname{Var}[z_j] = \overline{\psi'(d_j/2)} + \psi'(d_0/2)$ → solve for
#   $d_0$ by inverting the trigamma function (if the left side is not larger, the data
#   are consistent with a single common σ: $d_0 = \infty$, complete pooling);
# - $\operatorname{E}[z_j] = \log s_0^2 - \psi(d_0/2) + \log(d_0/2)$ → gives $s_0^2$.
#
# One task (`ai_rd_triton_cumsum`, 4 tied RE-Bench times) has $s_j = 0$ exactly —
# excluded from the moment fit (it would send $\log s_j^2 \to -\infty$) but still
# shrunk like every other task.
#
# RE-Bench has only 7 tasks with observed spread — too few for a stable moment fit —
# and its tasks (long, open-ended research) resemble HCAST far more than SWAA
# (seconds-scale software atoms), so RE-Bench borrows the HCAST prior.

# %%
def inverse_trigamma(value: float) -> float:
    """Solve ψ'(x) = value for x (ψ' is strictly decreasing on (0, ∞))."""
    return np.exp(brentq(
        lambda log_x: polygamma(1, np.exp(log_x)) - value,
        np.log(1e-8), np.log(1e8), xtol=1e-12,
    ))


def fit_log_variance_prior(sample_variances: np.ndarray,
                           degrees_of_freedom: np.ndarray) -> dict:
    """Smyth (2004) moment fit of the scaled-inverse-χ²(d0, s0²) prior on task
    variances. d0 = inf means the observed dispersion of s² is fully explained by
    χ² sampling noise around one common σ (complete pooling)."""
    half_df = degrees_of_freedom / 2
    z = np.log(sample_variances) - digamma(half_df) + np.log(half_df)
    observed_dispersion = z.var(ddof=1)
    expected_sampling_dispersion = polygamma(1, half_df).mean()
    between_task_variance = observed_dispersion - expected_sampling_dispersion
    if between_task_variance <= 0:
        prior_degrees_of_freedom, log_prior_variance = np.inf, z.mean()
    else:
        prior_degrees_of_freedom = 2 * inverse_trigamma(between_task_variance)
        log_prior_variance = (
            z.mean() + digamma(prior_degrees_of_freedom / 2)
            - np.log(prior_degrees_of_freedom / 2)
        )
    return {
        "n_tasks_in_fit": len(z),
        "prior_degrees_of_freedom": prior_degrees_of_freedom,
        "prior_sigma": float(np.exp(log_prior_variance / 2)),
        "observed_dispersion_of_log_s2": observed_dispersion,
        "expected_sampling_dispersion": expected_sampling_dispersion,
    }


prior_rows = []
for prior_source in ["HCAST", "SWAA"]:
    fit_tasks = tasks_with_observed_spread.query(
        "task_source == @prior_source and log_time_std > 0"
    )
    prior_rows.append({"task_source": prior_source} | fit_log_variance_prior(
        fit_tasks["log_time_std"].to_numpy() ** 2,
        fit_tasks["degrees_of_freedom"].to_numpy().astype(float),
    ))
prior_by_source = pd.DataFrame(prior_rows).set_index("task_source")
prior_by_source.loc["RE-Bench"] = prior_by_source.loc["HCAST"]  # borrowed (see above)
print(prior_by_source)

# %% [markdown]
# **Reading the fit:**
#
# - **HCAST: $d_0 = \infty$ — complete pooling.** The dispersion of the observed
#   $\log s_j^2$ is *fully explained* by χ² sampling noise around one common σ: with
#   2–4 runs per task, the data cannot distinguish task-to-task heterogeneity in σ from
#   noise. Every HCAST task gets the common $\tilde\sigma = s_0 \approx 0.89$ — which is
#   **above METR's global 0.78**, not below it. The raw medians (≈0.6) looked smaller
#   only because $s_j$ underestimates σ at these tiny n; the moment fit removes that
#   bias. (This is a point estimate: $d_0 = \infty$ sits at the boundary, and moderate
#   heterogeneity is also compatible with the data.)
# - **SWAA: $d_0 \approx 2.5$ — genuine heterogeneity**, prior worth ~2.5
#   pseudo-runs, around a much smaller $s_0 \approx 0.30$. Here METR's 0.78 overstates
#   the noise ~2.5×; but SWAA tasks are seconds-scale and barely influence frontier
#   horizons.
#
# ## Shrinkage
#
# $\tilde\sigma_j^2 = (d_0 s_0^2 + d_j s_j^2) / (d_0 + d_j)$ — tasks with many runs keep
# their own estimate, $n=2$ tasks are pulled toward $s_0$, $n \le 1$ tasks get exactly
# $s_0$.

# %%
prior_degrees_of_freedom_by_task = (
    sigma_by_task["task_source"].map(prior_by_source["prior_degrees_of_freedom"])
)
prior_variance_by_task = (
    sigma_by_task["task_source"].map(prior_by_source["prior_sigma"]) ** 2
)
observed_variance_weight = (
    sigma_by_task["degrees_of_freedom"] * sigma_by_task["log_time_std"].fillna(0) ** 2
)
sigma_by_task["log_time_std_shrunken"] = np.sqrt(np.where(
    np.isinf(prior_degrees_of_freedom_by_task),  # complete pooling: σ̃ = s0 exactly
    prior_variance_by_task,
    (prior_degrees_of_freedom_by_task * prior_variance_by_task + observed_variance_weight)
    / (prior_degrees_of_freedom_by_task + sigma_by_task["degrees_of_freedom"]),
))

fig = px.scatter(
    sigma_by_task.reset_index(),
    x="log_time_std", y="log_time_std_shrunken",
    color="task_source", size="n_successful_runs",
    hover_name="task_id",
    title="Empirical-Bayes shrinkage of per-task σ "
          "(diagonal = no shrinkage; horizontal pull-lines = per-source prior s₀)",
    labels={"log_time_std": "raw sample SD of ln(minutes)",
            "log_time_std_shrunken": "shrunken σ̃"},
)
axis_max = float(sigma_by_task["log_time_std"].max()) * 1.05
fig.add_shape(type="line", x0=0, y0=0, x1=axis_max, y1=axis_max,
              line=dict(dash="dot", color="gray"))
for prior_sigma, sources_with_prior in prior_by_source.groupby("prior_sigma"):
    fig.add_hline(y=prior_sigma, line_dash="dash", opacity=0.4,
                  annotation_text="s₀ " + "/".join(sources_with_prior.index))
fig.write_image(FIGURES / f"b_shrinkage{SUFFIX}.png", width=900, height=550, scale=2)
show(fig)

# %% [markdown]
# ## The x-axis uncertainty each task carries: SEM of its `human_minutes`
#
# What perturbs the logistic fit's x-axis is the uncertainty of the *geometric mean*,
# not of a single baseliner: $\text{SEM}_j = \tilde\sigma_j / \sqrt{n_j}$ for baselined
# tasks. Researcher-estimate tasks have no runs — their x-error is estimate error, not
# sampling error; we carry METR's σ = 1.05 for them (unverifiable from the public
# export, see check below). `gsem_factor` is the multiplicative version
# $e^{\text{SEM}}$ (×1.27 = "±27%").

# %%
is_baselined = sigma_by_task["human_source"] == "baseline"
sigma_by_task["sem_log"] = np.where(
    is_baselined,
    sigma_by_task["log_time_std_shrunken"]
    / np.sqrt(sigma_by_task["n_successful_runs"].clip(lower=1)),
    METR_SIGMA_ESTIMATE,
)
sigma_by_task["metr_sem_log"] = np.where(
    is_baselined,
    METR_SIGMA_BASELINED / np.sqrt(sigma_by_task["n_successful_runs"].clip(lower=1)),
    METR_SIGMA_ESTIMATE,
)
sigma_by_task["gsem_factor"] = np.exp(sigma_by_task["sem_log"])
sigma_by_task["sem_source"] = np.select(
    [~is_baselined, sigma_by_task["degrees_of_freedom"] == 0],
    ["metr_estimate_assumption", "prior_only"],
    default="empirical_shrunken",
)
print(sigma_by_task["sem_source"].value_counts())
print(sigma_by_task.groupby("task_source")["gsem_factor"]
      .describe()[["count", "25%", "50%", "75%", "max"]].round(3))

# %%
sem_comparison = pd.concat([
    sigma_by_task.loc[is_baselined].assign(
        sem_version="this work (shrunken σ̃/√n)", sem=sigma_by_task["sem_log"]),
    sigma_by_task.loc[is_baselined].assign(
        sem_version="METR global (0.78/√n)", sem=sigma_by_task["metr_sem_log"]),
])
fig = px.ecdf(
    sem_comparison, x="sem", color="sem_version",
    title="Per-task SEM of ln(human_minutes): empirical vs METR's global assumption "
          "(baselined tasks)",
    labels={"sem": "SEM of ln(human_minutes)"},
)
fig.write_image(FIGURES / f"b_sem_vs_metr{SUFFIX}.png", width=900, height=500, scale=2)
show(fig)

sem_ratio_to_metr = (
    sigma_by_task.loc[is_baselined, "sem_log"]
    / sigma_by_task.loc[is_baselined, "metr_sem_log"]
)
print("median per-task SEM ratio (this work / METR), by source:")
print(sem_ratio_to_metr.groupby(sigma_by_task["task_source"]).median().round(2))

# %% [markdown]
# ## Free check on METR's estimate-task σ = 1.05
#
# Three RE-Bench tasks have `human_source = "estimate"` **and** successful human runs in
# the export — so we can compare the researcher estimate against the gmean of actual
# runs. (n=3, and RE-Bench run-time conventions differ, so this is indicative only.)

# %%
estimate_tasks_with_runs = sigma_by_task.query(
    "human_source == 'estimate' and n_successful_runs > 0"
).index
gmean_of_runs_by_task = (
    successful_runs_with_derived_times
    .query("task_id in @estimate_tasks_with_runs")
    .groupby("task_id")["minutes_derived"].agg(stats.gmean)
)
estimate_check = sigma_by_task.loc[estimate_tasks_with_runs, ["human_minutes", "n_successful_runs"]]
estimate_check["gmean_of_runs"] = gmean_of_runs_by_task
estimate_check["abs_log_error"] = np.abs(
    np.log(estimate_check["gmean_of_runs"] / estimate_check["human_minutes"])
)
print(estimate_check.round(2))
print(f"METR's assumed estimate σ: {METR_SIGMA_ESTIMATE}")

# %% [markdown]
# ## Robustness (a): does the $d_0 = \infty$ boundary matter downstream?
#
# The HCAST $d_0 = \infty$ is a point estimate on the boundary of the parameter space.
# A bootstrap over the 60 spread-observable HCAST tasks refits finite heterogeneity in
# only ~⅓ of resamples (median $d_0 \approx 11$ among those) — so the data are also
# compatible with moderate heterogeneity, and a full-Bayes posterior on $d_0$ would keep
# mass on finite values. Does that ambiguity change what feeds (c)/(d)? Recompute the
# HCAST shrunken σ̃ (holding $s_0 = 0.89$) under $d_0 \in \{\infty, 20, 10\}$:

# %%
def shrink_sigma(sample_std, degrees_of_freedom, prior_degrees_of_freedom, prior_sigma):
    if np.isinf(prior_degrees_of_freedom):
        return np.full_like(np.asarray(sample_std, dtype=float), prior_sigma)
    return np.sqrt(
        (prior_degrees_of_freedom * prior_sigma ** 2
         + degrees_of_freedom * np.nan_to_num(sample_std) ** 2)
        / (prior_degrees_of_freedom + degrees_of_freedom)
    )


hcast_tasks = sigma_by_task.query("task_source == 'HCAST'")
hcast_prior_sigma = prior_by_source.loc["HCAST", "prior_sigma"]
sigma_by_alternative_d0 = pd.DataFrame({
    f"sigma_shrunken_d0={d0}": shrink_sigma(
        hcast_tasks["log_time_std"], hcast_tasks["degrees_of_freedom"],
        d0, hcast_prior_sigma,
    )
    for d0 in [np.inf, 20, 10]
}, index=hcast_tasks.index)
print("HCAST shrunken σ̃ across the 97 tasks, under alternative prior strengths:")
print(sigma_by_alternative_d0.describe().loc[["mean", "50%", "max"]].round(3))
print("\nThe most-affected tasks (largest raw s_j) — the only ones that move:")
most_affected_hcast_tasks = hcast_tasks["log_time_std"].nlargest(4).index
print(sigma_by_alternative_d0.loc[most_affected_hcast_tasks].round(3))

# %% [markdown]
# The **median is fixed at $s_0$** and the mean moves ~1% (0.889 → 0.901) across
# $d_0 \in \{\infty, 20, 10\}$. Only a handful of high-$s_j$ tasks move materially, and
# only at $d_0 = 10$ (the low end of what the bootstrap supports): the single most
# extreme task goes 0.89 → 1.30. So the *aggregate scale* that (c)/(d) consume is
# insensitive to the boundary; individual tail tasks are not — but per-task independent
# x-noise largely averages out across ~130 tasks in the horizon fit (this is exactly
# what the nonparametric (a0) bootstrap already showed: ±2.4 days on doubling time).
# The $d_0 = \infty$ result is best read as a statement about the *story* — the data
# cannot resolve task-to-task heterogeneity in σ from n=2–3 runs — rather than a claim
# that every task's true σ is identical.

# %% [markdown]
# ## Robustness (b): is the χ² sampling model (within-task log-normality) violated?
#
# The moment fit attributes observed dispersion of $s_j^2$ to sampling noise via the
# χ² model, which assumes within-task **log-times are normal**. If they are heavier-
# tailed, $s_j^2$ carries *more* sampling dispersion than credited — which would push
# the fit *toward* $d_0 = \infty$, not away from it. Check on the four HCAST tasks with
# enough runs (n ≥ 9) to see a within-task distribution:

# %%
high_n_hcast_tasks = (
    successful_runs_with_derived_times
    .query("task_source == 'HCAST'")
    .groupby("task_id")
    .filter(lambda task_runs: len(task_runs) >= 9)
)
normality_by_task_rows = []
for task_id, task_runs in high_n_hcast_tasks.groupby("task_id"):
    log_times = np.log(task_runs["minutes_derived"].to_numpy())
    normality_by_task_rows.append({
        "task_id": task_id,
        "n_successful_runs": len(log_times),
        "shapiro_p": stats.shapiro(log_times).pvalue,
        "excess_kurtosis": stats.kurtosis(log_times),
        "skew": stats.skew(log_times),
    })
normality_by_high_n_task = pd.DataFrame(normality_by_task_rows).set_index("task_id")
print(normality_by_high_n_task.round(3))

# %%
standardized_log_times = pd.concat([
    task_runs.assign(standardized_log_time=stats.zscore(
        np.log(task_runs["minutes_derived"]), ddof=1))
    for _, task_runs in high_n_hcast_tasks.groupby("task_id")
])
fig = px.ecdf(
    standardized_log_times, x="standardized_log_time", color="task_id",
    title="Within-task log-time shape vs normal (high-n HCAST tasks; "
          "black dashed = standard normal CDF)",
    labels={"standardized_log_time": "z-scored ln(completion minutes)"},
)
normal_grid = np.linspace(-3, 3, 200)
fig.add_scatter(x=normal_grid, y=stats.norm.cdf(normal_grid), mode="lines",
                line=dict(color="black", dash="dash"), name="N(0,1)")
fig.write_image(FIGURES / f"b_within_task_normality{SUFFIX}.png", width=900, height=500, scale=2)
show(fig)

# %% [markdown]
# Two of four are close to normal; two (`orm_somebugs` n=57, excess kurtosis +9.5;
# `pico_ctf/166` n=19, +3.5) are markedly **heavier-tailed** than lognormal. Heavy tails
# inflate the true sampling dispersion of $s_j^2$ beyond the χ² model — so the −0.12
# dispersion *deficit* that drove $d_0 = \infty$ is, if anything, understated. **The
# complete-pooling result is robust to this check: correcting the sampling model would
# reinforce it, not overturn it.** The caveat heavy tails *do* introduce is separate:
# σ (a second moment) is an incomplete summary of these distributions, and the
# CLT-based SEM = σ̃/√n understates gmean uncertainty at small n — a point in favour of
# the nonparametric bootstrap in (a0) as the honest floor, and a reason to read the
# parametric (c)/(d) as a smooth complement rather than a replacement.

# %% [markdown]
# ## Output

# %%
sigma_by_task.to_csv(DATA_OUT / f"b_sigma_by_task{SUFFIX}.csv")
print(f"wrote {DATA_OUT / f'b_sigma_by_task{SUFFIX}.csv'} ({len(sigma_by_task)} tasks)")
sigma_by_task.head()
