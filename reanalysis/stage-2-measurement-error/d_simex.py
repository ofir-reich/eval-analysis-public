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
# # Stage 2(d) — SIMEX with per-task σ
#
# The x-noise in `human_minutes` is a classic **errors-in-variables** problem: noise on
# the regressor *attenuates* the logistic slope, biasing the fitted curve. Wider CIs
# ((a)/(a0)) are not the whole story — the **point estimate** is biased too. SIMEX
# (simulation–extrapolation) corrects it: add *known* extra noise at levels
# λ = 0.5, 1, 1.5, 2, watch the horizon degrade as a function of λ, and extrapolate the
# trend back to **λ = −1** (the hypothetical zero-noise fit).
#
# METR's [modelling-assumptions note
# (2026-03-20)](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)
# ran SIMEX with a **global** noise scale — SEM² = 0.78²/n (baselined), σ² = 1.05²
# (estimates) — and reported Opus 4.6 p50 falling **−36%** (7h38m) and p80 **+9%**.
# Here we rerun the identical procedure but with the **per-task** SEM from
# [(b)](b_sigma_task.py) — where the noise scale differs by source (HCAST/RE-Bench a
# touch *above* 0.78, SWAA well below) — to see whether the haircut grows or shrinks.
#
# **Reproduction target check:** we also run METR's own global-σ version here, so any
# difference is attributable to per-task vs global σ, not to pipeline differences.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from joblib import Parallel, delayed

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

N_SIMEX = int(os.environ.get("N_SIMEX", 200))   # noised datasets per λ (METR used 200)
LAMBDA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]
METR_SIGMA_BASELINED = 0.78
METR_SIGMA_ESTIMATE = 1.05

# %%
runs = metr.load_runs()
official_human_minutes_by_task = metr.official_human_minutes_by_task(runs)
log_human_minutes_by_task = np.log(official_human_minutes_by_task)
sigma_by_task = pd.read_csv(DATA_OUT / f"b_sigma_by_task{SUFFIX}.csv", index_col=0)

frontier_agents, release_dates, official_fits_by_agent = (
    metr.load_frontier_agents_and_dates()
)
years_since_release_by_agent = metr.years_since_first_release_by_agent(
    frontier_agents, release_dates
)

# two noise models, both indexed to the runs' task order
per_task_sem_log = sigma_by_task["sem_log"].reindex(official_human_minutes_by_task.index)
n_successful_runs = (
    sigma_by_task["n_successful_runs"].reindex(official_human_minutes_by_task.index)
)
is_estimate = (
    sigma_by_task["human_source"].reindex(official_human_minutes_by_task.index)
    == "estimate"
)
metr_global_sem_log = np.where(
    is_estimate, METR_SIGMA_ESTIMATE,
    METR_SIGMA_BASELINED / np.sqrt(n_successful_runs.clip(lower=1)),
)
metr_global_sem_log = pd.Series(metr_global_sem_log, index=per_task_sem_log.index)
sem_log_by_noise_model = {"per_task (b)": per_task_sem_log, "metr_global": metr_global_sem_log}

# %% [markdown]
# ## Run SIMEX
#
# At each λ we draw `N_SIMEX` datasets with extra multiplicative noise of variance
# λ·SEM²_task, refit every agent's p50/p80, and average log-horizon over the draws
# (λ = 0 is the unperturbed fit). Then per agent we fit a quadratic in λ over the grid
# and evaluate it at λ = −1 — METR's extrapolant.

# %%
def simex_horizons_at_lambda(noise_model: str, lam: float) -> dict:
    """Mean over N_SIMEX draws of each agent's log2-horizon at added-noise level λ."""
    sem_log = sem_log_by_noise_model[noise_model]
    log_horizon_sums = {}
    for draw_index in range(N_SIMEX):
        rng = np.random.default_rng(10_000 * int(lam * 10) + draw_index)
        if lam == 0.0:
            human_minutes_by_task = official_human_minutes_by_task
        else:
            human_minutes_by_task = np.exp(
                log_human_minutes_by_task + np.sqrt(lam) * rng.normal(0, sem_log)
            )
        horizon_per_quantile_by_agent = metr.fit_horizons_per_quantile_by_agent(
            runs, human_minutes_by_task
        )
        for quantile, horizon_by_agent in horizon_per_quantile_by_agent.items():
            for agent, horizon in horizon_by_agent.items():
                key = (agent, f"p{int(quantile * 100)}")
                log_horizon_sums[key] = log_horizon_sums.get(key, 0.0) + np.log2(horizon)
    return {key: total / N_SIMEX for key, total in log_horizon_sums.items()}


simex_cache = DATA_OUT / f"d_simex_curves{SUFFIX}.csv"
if simex_cache.exists():
    simex_curve_rows = pd.read_csv(simex_cache)
    print(f"loaded cached SIMEX curves ({len(simex_curve_rows)} rows)")
else:
    jobs = [(noise_model, lam) for noise_model in sem_log_by_noise_model
            for lam in LAMBDA_GRID]
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(simex_horizons_at_lambda)(noise_model, lam) for noise_model, lam in jobs
    )
    simex_curve_rows = pd.DataFrame([
        {"noise_model": noise_model, "lambda": lam, "agent": agent,
         "quantile": quantile, "mean_log2_horizon": mean_log2_horizon}
        for (noise_model, lam), horizons in zip(jobs, results)
        for (agent, quantile), mean_log2_horizon in horizons.items()
    ])
    simex_curve_rows.to_csv(simex_cache, index=False)
    print(f"computed SIMEX curves ({len(simex_curve_rows)} rows)")

# %% [markdown]
# ## Extrapolate to λ = −1

# %%
def extrapolate_to_minus_one(curve: pd.DataFrame) -> float:
    """Quadratic fit of mean_log2_horizon on λ over the grid, evaluated at λ=−1."""
    coefficients = np.polyfit(curve["lambda"], curve["mean_log2_horizon"], 2)
    return float(np.polyval(coefficients, -1.0))

corrected_rows = []
for (noise_model, agent, quantile), curve in simex_curve_rows.groupby(
    ["noise_model", "agent", "quantile"]
):
    naive_log2 = curve.loc[curve["lambda"] == 0.0, "mean_log2_horizon"].iloc[0]
    corrected_log2 = extrapolate_to_minus_one(curve)
    corrected_rows.append({
        "noise_model": noise_model, "agent": agent, "quantile": quantile,
        "naive_horizon_min": 2 ** naive_log2,
        "simex_horizon_min": 2 ** corrected_log2,
        "pct_change": 100 * (2 ** (corrected_log2 - naive_log2) - 1),
    })
simex_correction_by_agent = pd.DataFrame(corrected_rows)
simex_correction_by_agent.to_csv(DATA_OUT / f"d_simex_corrections{SUFFIX}.csv", index=False)

# %% [markdown]
# ## Frontier summary: per-task vs METR-global haircut

# %%
frontier_correction = simex_correction_by_agent[
    simex_correction_by_agent["agent"].isin(frontier_agents)
]
median_pct_change = (
    frontier_correction.groupby(["noise_model", "quantile"])["pct_change"]
    .median().unstack()
)
print("Median SIMEX correction across frontier agents (%):")
print(median_pct_change.round(1).to_string())

# the newest frontier agent (analogue of METR's Opus 4.6 headline)
newest_frontier_agent = max(
    years_since_release_by_agent, key=years_since_release_by_agent.get
)
print(f"\nNewest frontier agent ({newest_frontier_agent}):")
print(simex_correction_by_agent[
    simex_correction_by_agent["agent"] == newest_frontier_agent
].round(1).to_string(index=False))

# %%
fig = px.box(
    frontier_correction, x="quantile", y="pct_change", color="noise_model",
    title="SIMEX horizon correction across frontier agents — per-task σ vs METR global",
    labels={"pct_change": "horizon change at λ=−1 (%)", "quantile": ""},
)
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.write_image(FIGURES / f"d_simex_correction{SUFFIX}.png", width=850, height=500, scale=2)
show(fig)

# %% [markdown]
# ## Example extrapolation curves (newest frontier agent)

# %%
example_curves = simex_curve_rows[simex_curve_rows["agent"] == newest_frontier_agent].copy()
example_curves["horizon_min"] = 2 ** example_curves["mean_log2_horizon"]
example_curves["curve"] = example_curves["noise_model"] + " · " + example_curves["quantile"]
fig = px.line(
    example_curves, x="lambda", y="horizon_min", color="curve", markers=True,
    log_y=True,
    title=f"SIMEX extrapolation curves — {newest_frontier_agent} "
          "(dashed = quadratic extrapolated back to λ=−1, the zero-noise fit)",
    labels={"lambda": "added-noise level λ", "horizon_min": "mean horizon (min, log)"},
)
# draw each curve's quadratic extrapolation from λ=0 back to λ=−1, ending on the marker
extrapolation_lambdas = np.linspace(0.0, -1.0, 20)
for curve_name, curve in example_curves.groupby("curve"):
    color = fig.data[[trace.name for trace in fig.data].index(curve_name)].line.color
    coefficients = np.polyfit(curve["lambda"], curve["mean_log2_horizon"], 2)
    fig.add_scatter(
        x=extrapolation_lambdas,
        y=2 ** np.polyval(coefficients, extrapolation_lambdas),
        mode="lines", line=dict(color=color, dash="dash"),
        showlegend=False, hoverinfo="skip",
    )
    fig.add_scatter(
        x=[-1.0], y=[2 ** np.polyval(coefficients, -1.0)],
        mode="markers", marker=dict(color=color, symbol="star", size=11),
        showlegend=False, hoverinfo="skip",
    )
fig.add_vline(x=0, line_dash="dot", line_color="gray")
fig.add_vline(x=-1, line_dash="dash", line_color="red")
fig.write_image(FIGURES / f"d_simex_curves{SUFFIX}.png", width=850, height=500, scale=2)
show(fig)

# %% [markdown]
# ## Takeaway
#
# The printed medians give the headline: whether per-task σ deepens or softens METR's
# −26…−36% p50 haircut, and whether the p80 sign (METR: +9%) holds. Because SIMEX
# corrects the *slope attenuation*, p50 (a location) and p80 (which also depends on the
# slope) move differently — the errors-in-variables mechanism behind the p50/p80 gap
# flagged in the Stage 2 intro. Caveats: the quadratic extrapolant to λ=−1 is itself
# uncertain (METR emphasised wide ranges), and SIMEX presumes the noise is *independent*
# across tasks — the *systematic* component is bounded separately in
# [(c)](c_coherent_shift.py).
