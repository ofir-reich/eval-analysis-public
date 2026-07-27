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
# # Stage 2(c) — Coherent ±1σ shift (systematic-error worst case)
#
# The bootstrap in [(a0)](a0_nonparametric_xboot.py) treats x-axis noise as
# **independent** across tasks — so it averages out (doubling-time spread was just
# ±2.4 days). But a *systematic* error — every task's `human_minutes` biased the same
# direction — does **not** average out. Sources of such correlated error:
#
# - **baseliner selection** (Stage 4): if the humans METR could recruit are uniformly
#   faster (or slower) than the reference population, every task's `human_minutes` is
#   biased coherently;
# - a shared baselining convention (instructions, environment, "give up" policy);
# - the flooring/δ conventions from [Stage 1](../stage-1-export-archaeology/).
#
# Here we bound it: shift **every** task's `human_minutes` coherently by ±1 σ̃_task (the
# shrunken per-task spread from [(b)](b_sigma_task.py)) in log space, refit METR's
# horizons, and read off the effect on p50, p80, and the doubling time. We also show the
# gentler ±1 SEM version (σ̃/√n — the correlated *sampling* error, smaller by √n).
#
# **What to expect:** a coherent shift that were *perfectly uniform* across all tasks
# would move every horizon by the same factor and leave the doubling time **unchanged**
# (it only shifts the intercept). Our shift is uniform only *within* a task source
# (σ̃ ≈ 0.89 for HCAST/RE-Bench, ≈ 0.30 for SWAA after complete-pooling), so any
# doubling-time movement comes from the source mix differing across agents.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.express as px

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

# %% [markdown]
# ## Build the shifted x-axes

# %%
runs = metr.load_runs()
official_human_minutes_by_task = metr.official_human_minutes_by_task(runs)
sigma_by_task = pd.read_csv(DATA_OUT / f"b_sigma_by_task{SUFFIX}.csv", index_col=0)

frontier_agents, release_dates, official_fits_by_agent = (
    metr.load_frontier_agents_and_dates()
)
years_since_release_by_agent = metr.years_since_first_release_by_agent(
    frontier_agents, release_dates
)
print(f"frontier agents ({len(frontier_agents)})")

# per-task log-space shift magnitudes, aligned to the runs' task order
sigma_shift_by_task = sigma_by_task["log_time_std_shrunken"].reindex(
    official_human_minutes_by_task.index
)
# n=1 / estimate tasks have no observed σ̃ — carry the SEM the (b) output assigned them
# (prior s0 for n=1 baselined; METR's 1.05 for estimates)
sigma_shift_by_task = sigma_shift_by_task.fillna(sigma_by_task["sem_log"])
sem_shift_by_task = sigma_by_task["sem_log"].reindex(official_human_minutes_by_task.index)

shifted_human_minutes_by_scenario = {
    "baseline": official_human_minutes_by_task,
    "+1σ̃ (all tasks)": official_human_minutes_by_task * np.exp(sigma_shift_by_task),
    "−1σ̃ (all tasks)": official_human_minutes_by_task * np.exp(-sigma_shift_by_task),
    "+1 SEM (all tasks)": official_human_minutes_by_task * np.exp(sem_shift_by_task),
    "−1 SEM (all tasks)": official_human_minutes_by_task * np.exp(-sem_shift_by_task),
}
print("median shift factor by scenario:")
for scenario, shifted in shifted_human_minutes_by_scenario.items():
    print(f"  {scenario:22} ×{np.exp(np.median(np.log(shifted / official_human_minutes_by_task))):.3f}")

# %% [markdown]
# ## Refit horizons and the doubling time under each scenario

# %%
scenario_rows = []
horizon_per_quantile_by_agent_by_scenario = {}
for scenario, shifted_human_minutes in shifted_human_minutes_by_scenario.items():
    horizon_per_quantile_by_agent = metr.fit_horizons_per_quantile_by_agent(
        runs, shifted_human_minutes
    )
    horizon_per_quantile_by_agent_by_scenario[scenario] = horizon_per_quantile_by_agent
    for quantile, horizon_by_agent in horizon_per_quantile_by_agent.items():
        doubling_days = metr.fit_doubling_days(
            horizon_by_agent, years_since_release_by_agent
        )
        frontier_horizon = pd.Series(horizon_by_agent).reindex(frontier_agents).dropna()
        scenario_rows.append({
            "scenario": scenario,
            "quantile": f"p{int(quantile * 100)}",
            "gmean_frontier_horizon_min": float(np.exp(np.log(frontier_horizon).mean())),
            "doubling_days": doubling_days,
        })
scenario_summary = pd.DataFrame(scenario_rows)

# express horizon/doubling as ratios vs the baseline scenario
baseline_by_quantile = scenario_summary.query("scenario == 'baseline'").set_index("quantile")
scenario_summary["horizon_vs_baseline"] = scenario_summary.apply(
    lambda row: row["gmean_frontier_horizon_min"]
    / baseline_by_quantile.loc[row["quantile"], "gmean_frontier_horizon_min"], axis=1
)
scenario_summary["doubling_vs_baseline"] = scenario_summary.apply(
    lambda row: row["doubling_days"]
    / baseline_by_quantile.loc[row["quantile"], "doubling_days"], axis=1
)
print(scenario_summary.round(3).to_string(index=False))
scenario_summary.to_csv(DATA_OUT / f"c_coherent_shift{SUFFIX}.csv", index=False)

# %% [markdown]
# ## The headline: horizons swing hugely; the trend barely moves
#
# A coherent ±1σ̃ bias multiplies frontier horizons by a large factor (σ̃ ≈ 0.89 nats for
# the long HCAST/RE-Bench tasks → up to ~×2.4 each way) — the *absolute* horizon is very
# sensitive to any systematic error in the baseline times. But the **doubling time**
# moves little, because the shift is nearly a uniform rescale and rescaling leaves the
# slope of log-horizon-vs-date almost untouched.

# %%
SCENARIO_ORDER = ["−1σ̃ (all tasks)", "−1 SEM (all tasks)", "baseline",
                  "+1 SEM (all tasks)", "+1σ̃ (all tasks)"]

fig = px.scatter(
    scenario_summary, x="scenario", y="doubling_vs_baseline",
    color="quantile", symbol="quantile",
    category_orders={"scenario": SCENARIO_ORDER},
    title="Doubling-time sensitivity to a coherent ±1σ̃ / ±1 SEM shift<br>"
          "<sub>1.0 = unchanged vs baseline; note the narrow vertical range</sub>",
    labels={"doubling_vs_baseline": "doubling time ÷ baseline doubling time",
            "scenario": ""},
)
fig.update_traces(marker=dict(size=11))
fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
fig.write_image(FIGURES / f"c_doubling_sensitivity{SUFFIX}.png", width=900, height=500, scale=2)
show(fig)

# %%
fig = px.scatter(
    scenario_summary, x="scenario", y="gmean_frontier_horizon_min",
    color="quantile", symbol="quantile", log_y=True,
    category_orders={"scenario": SCENARIO_ORDER},
    title="Time horizon under each coherent-shift scenario<br>"
          "<sub>geometric mean over the state-of-the-art-at-release agents</sub>",
    labels={"gmean_frontier_horizon_min": "time horizon (minutes, log axis)",
            "scenario": ""},
)
fig.update_traces(marker=dict(size=11))
fig.write_image(FIGURES / f"c_horizon_levels{SUFFIX}.png", width=900, height=500, scale=2)
show(fig)

# %% [markdown]
# ## p80 vs p50 under the shift
#
# A pure multiplicative shift moves p50 and p80 by the *same* factor (it slides the
# fitted curve horizontally without changing its slope), so their **ratio is
# preserved** — confirmed below. This is the systematic-error counterpart to the
# errors-in-variables *attenuation* in (a)/(d), where random x-noise flattens the slope
# and widens the p50/p80 gap; a coherent shift does not.

# %%
p50_p80_ratio_by_scenario = (
    scenario_summary.pivot(index="scenario", columns="quantile",
                           values="gmean_frontier_horizon_min")
)
p50_p80_ratio_by_scenario["p80/p50"] = (
    p50_p80_ratio_by_scenario["p80"] / p50_p80_ratio_by_scenario["p50"]
)
print(p50_p80_ratio_by_scenario.round(3).to_string())
