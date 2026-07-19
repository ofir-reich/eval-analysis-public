# Stage 2 — Measurement error in `human_minutes`

How much does uncertainty in the x-axis — each task's human completion time — matter for
the headline results? METR's hierarchical bootstrap resamples task families, tasks, and
agent runs, but carries a **fixed** `human_minutes` per task through every replicate: the
x-axis contributes nothing to their published confidence intervals.

Sub-analyses (a0 done; b–d planned):

| | approach | assumption level |
|---|---|---|
| **(a0)** [Nonparametric x-bootstrap](#a0--nonparametric-x-bootstrap) | resample *which baseline runs* enter each task's gmean | none (lower bound) |
| (b) | hierarchical estimate of per-task log-time spread σ_task, incl. n=1 tasks | parametric |
| (c) | coherent ±1σ_task shift of all tasks (systematic-error worst case) | parametric |
| (d) | SIMEX with per-task empirical σ (refines METR's global-σ version) | parametric |

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
