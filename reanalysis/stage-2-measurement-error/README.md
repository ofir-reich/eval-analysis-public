# Stage 2 — Measurement error in `human_minutes`

How much does uncertainty in the x-axis — each task's human completion time — matter for
the headline results? METR's hierarchical bootstrap resamples task families, tasks, and
agent runs, but carries a **fixed** `human_minutes` per task through every replicate: the
x-axis contributes nothing to their published confidence intervals.

Sub-analyses (a0, b done; c–d planned):

| | approach | assumption level |
|---|---|---|
| **(a0)** [Nonparametric x-bootstrap](#a0--nonparametric-x-bootstrap) | resample *which baseline runs* enter each task's gmean | none (lower bound) |
| **(b)** [Per-task σ via empirical Bayes](#b--per-task-σ-via-empirical-bayes-shrinkage) | hierarchical estimate of per-task log-time spread, incl. n=1 tasks | parametric |
| (c) | coherent ±1σ_task shift of all tasks (systematic-error worst case) | parametric |
| (d) | SIMEX with per-task σ from (b) (refines METR's global-σ version) | parametric |

## (a0) — Nonparametric x-bootstrap

**Design:** three bootstrap conditions × 500 replicates, all using METR's own fit code and
headline parameters (weighted logistic, regularization 1e-5, `invsqrt_task_weight`), on
Time Horizon v1.0:

- `metr` — METR's family→task→run resampling (replicates their CI);
- `metr+x` — same, plus per-replicate resampling of each task's successful baseline runs
  (δ-corrected times from [Stage 1](../stage-1-export-archaeology/)), re-computing its
  `human_minutes` as the gmean of the resample;
- `x_only` — x-resampling alone, isolating the pure x-axis contribution.

126 / 170 tasks have ≥2 successful baseline runs and a gmean-consistent `human_minutes`
and hence vary; **n=1 tasks (21), researcher-estimate tasks, and RE-Bench are held fixed**
— this is by construction a *lower bound* on x-axis uncertainty, and the frozen tasks are
disproportionately the long ones.

**Result: the assumption-free floor is small.**

- Per-agent p50 CIs widen by a **median 3.8%** in log-width across the 17 frontier agents
  (max ≈ 10%: o1-preview +9.8%, Claude 3.7 Sonnet +8.1%); the pure x-axis contribution is
  ~0.2–2% of total CI variance.
- The **doubling-time** 95% CI: [172, 224] days (`metr`) → [170, 226] days (`metr+x`),
  ~5% wider; medians essentially unchanged (199.3 → 197.9 days). Pure-x doubling-time
  spread is just ±2.4 days — per-task independent noise largely averages out across the
  ~130 varying tasks.

![CI comparison](figures/a0_ci_comparison.png)

![doubling time](figures/a0_doubling_time.png)

**Interpretation.** The *resampleable* component of x-axis noise barely moves the headline
results — reassuring for METR's published CIs as far as it goes. What this floor cannot
see is exactly where the remaining risk lives: single-baseline tasks, researcher estimates
(the long tasks driving frontier horizons), small-n variance understatement, and
*systematic* (correlated) errors in baseline conventions. Those are the targets of
(b)–(d), and of the survival analysis in Stage 3.

Code: [`a0_nonparametric_xboot.py`](a0_nonparametric_xboot.py) (jupytext; paired executed
notebook [`a0_nonparametric_xboot.ipynb`](a0_nonparametric_xboot.ipynb)). Outputs:
[`data/a0_ci_by_agent.csv`](data/a0_ci_by_agent.csv),
[`data/a0_doubling_times.csv`](data/a0_doubling_times.csv).

## (b) — Per-task σ via empirical-Bayes shrinkage

**Goal:** σ_task — the SD of ln(completion time) across human baseliners — for every
task, including the 21 single-baseline tasks where no spread is observable.

**Method:** the *limma/eBayes* moderated-variance model (Smyth 2004): true task variances
σ²_j drawn from a scaled-inverse-χ²(d₀, s₀²) prior; hyperparameters fit by **method of
moments** on log s²_j (observed dispersion = known χ² sampling noise + between-task
spread, in trigamma units); posterior mean is the precision-weighted shrinkage
σ̃²_j = (d₀s₀² + d_j s²_j)/(d₀ + d_j). Because between-baseliner spread differs
systematically by source, the prior is fit **per task source** (RE-Bench, with only 7
spread-observable tasks, borrows the HCAST prior).

**Results — the raw spreads were misleading; METR's global σ = 0.78 is vindicated for
the tasks that matter:**

- **HCAST: d₀ = ∞ (complete pooling), common σ ≈ 0.89.** The dispersion of observed
  log s²_j (2.33) is fully explained by χ² sampling noise around one common σ (expected
  2.46): with n = 2–4 runs per task, task-to-task heterogeneity in σ is undetectable.
  The raw sample SDs (median 0.61) sit below 0.78 only because s_j is downward-biased at
  tiny n — after bias removal the common σ is slightly **above** METR's 0.78 (per-task
  SEM ratio ours/METR ≈ 1.14).
- **SWAA: genuine heterogeneity** (d₀ ≈ 2.5) around a much smaller s₀ ≈ 0.30 — METR's
  global 0.78 overstates SWAA noise ~2.5×, but these are seconds-scale tasks with little
  influence on frontier horizons.
- **Estimate tasks:** three RE-Bench estimate tasks also have successful runs in the
  export; their estimate-vs-gmean log errors are 0.12 / 0.41 / 0.12 — small next to
  METR's assumed σ = 1.05 (n=3, indicative only; all three "estimates" are exactly the
  480-min budget).
- Implied multiplicative gsem factor e^SEM per task: HCAST median ×1.88, SWAA ×1.18.

**Implication for (c)/(d):** the sign of the refinement is no longer predetermined.
Relative to METR's global assumption, per-task σ says *slightly more* x-noise on
HCAST/RE-Bench (the frontier-driving tasks) and much less on SWAA and possibly
estimates — whether METR's −26…−36% SIMEX haircut grows or shrinks under per-task σ is
exactly what (d) will measure.

![sigma ECDF by source](figures/b_sigma_ecdf_by_source.png)

![shrinkage](figures/b_shrinkage.png)

![SEM vs METR](figures/b_sem_vs_metr.png)

Code: [`b_sigma_task.py`](b_sigma_task.py) (jupytext; paired executed notebook
[`b_sigma_task.ipynb`](b_sigma_task.ipynb)). Output:
[`data/b_sigma_by_task.csv`](data/b_sigma_by_task.csv) — 170 tasks with raw and
shrunken σ, SEM of ln(human_minutes), and a `sem_source` flag
(130 `empirical_shrunken`, 21 `prior_only`, 19 `metr_estimate_assumption`).
