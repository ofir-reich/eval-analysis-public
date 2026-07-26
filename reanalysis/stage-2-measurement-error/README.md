# Stage 2 — Measurement error in `human_minutes`

How much does uncertainty in the x-axis — each task's human completion time — matter for
the headline results? METR's hierarchical bootstrap resamples task families, tasks, and
agent runs, but carries a **fixed** `human_minutes` per task through every replicate: the
x-axis contributes nothing to their published confidence intervals.

Sub-analyses (all done):

| | approach | assumption level |
|---|---|---|
| **(a0)** [Nonparametric x-bootstrap](#a0--nonparametric-x-bootstrap) | resample *which baseline runs* enter each task's gmean | none (lower bound) |
| **(b)** [Per-task σ via empirical Bayes](#b--per-task-σ-via-empirical-bayes-shrinkage) | hierarchical estimate of per-task log-time spread, incl. n=1 tasks | parametric |
| **(a)** [Parametric x-bootstrap](#a--parametric-x-bootstrap-with-per-task-σ) | draw every task's `human_minutes` from LogNormal(·, SEM²_task) | parametric |
| **(c)** [Coherent ±1σ shift](#c--coherent-1σ-shift-systematic-error-worst-case) | shift *all* tasks the same direction (systematic-error worst case) | parametric |
| **(d)** [SIMEX with per-task σ](#d--simex-with-per-task-σ) | errors-in-variables *point-estimate* correction vs METR's global σ | parametric |

**Bottom line for Stage 2.** The x-axis matters for the *absolute* horizon numbers but
barely for the *trend*. The doubling time is remarkably robust: independent x-noise
averages out (CI widens ~2 days even with every task perturbed, (a)), and a coherent
systematic bias — even a full ±1σ — moves it under ±10% while swinging absolute horizons
by ~×2.4 ((c)). The one place the x-axis bites the headline is the errors-in-variables
*bias* in the p50/p80 point estimates ((d), SIMEX): a real effect METR already flagged.
Our per-task σ makes that correction **slightly larger, not smaller**, than METR's global
σ=0.78 — overturning our initial guess — because the frontier-driving HCAST/RE-Bench
tasks are a touch noisier than 0.78 (see (b)). Shared fit machinery for (a)/(c)/(d):
[`metr_fit_helpers.py`](metr_fit_helpers.py), which reproduces METR's published p50/p80 exactly.

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

**Two robustness checks on the stark HCAST result:**

- **(a) The $d_0 = \infty$ boundary doesn't move the numbers (c)/(d) consume.** A
  bootstrap over the 60 HCAST tasks detects finite heterogeneity in only ~⅓ of
  resamples (median $d_0 \approx 11$), so the boundary is a coin-flip and a full-Bayes
  posterior would keep mass on finite $d_0$. But recomputing shrunken σ̃ under
  $d_0 \in \{\infty, 20, 10\}$: median fixed at $s_0$, mean shifts ~1%; only a few
  high-$s_j$ tasks move (the most extreme, 0.89 → 1.30, and only at the implausibly
  heterogeneous $d_0 = 10$). Since $d_j \le 2$ for 46/60 tasks, every plausible prior
  gives heavy shrinkage. The result is a statement about the *story* (n=2–3 can't
  resolve heterogeneity), not the aggregate scale.
- **(b) Non-normal within-task log-times would *reinforce*, not overturn, the result.**
  Of four HCAST tasks with n ≥ 9, two are near-normal and two markedly heavier-tailed
  (`orm_somebugs` n=57, excess kurtosis +9.5; `pico_ctf/166` +3.5). Heavy tails inflate
  the true sampling dispersion of $s_j^2$ beyond the χ² model — so the −0.12 dispersion
  *deficit* driving $d_0 = \infty$ is if anything understated. The honest caveat heavy
  tails leave is separate: σ is an incomplete summary and the CLT SEM = σ̃/√n is
  optimistic at small n — which is why (a0)'s assumption-free bootstrap remains the
  reported floor and (c)/(d) are read as smooth complements.

![within-task normality](figures/b_within_task_normality.png)

![sigma ECDF by source](figures/b_sigma_ecdf_by_source.png)

![shrinkage](figures/b_shrinkage.png)

![SEM vs METR](figures/b_sem_vs_metr.png)

Code: [`b_sigma_task.py`](b_sigma_task.py) (jupytext; paired executed notebook
[`b_sigma_task.ipynb`](b_sigma_task.ipynb)). Output:
[`data/b_sigma_by_task.csv`](data/b_sigma_by_task.csv) — 170 tasks with raw and
shrunken σ, SEM of ln(human_minutes), and a `sem_source` flag
(130 `empirical_shrunken`, 21 `prior_only`, 19 `metr_estimate_assumption`).

## (a) — Parametric x-bootstrap with per-task σ

(a0) could only perturb the 126 tasks with ≥2 successful runs and *froze* the 21
single-baseline tasks, the researcher estimates, and RE-Bench — disproportionately the
long, frontier-driving tasks — making it a lower bound. Here we draw **every** task's
`human_minutes` from LogNormal(ln `human_minutes`, SEM²_task) each replicate, with
SEM_task = σ̃_task/√n (baselined) or 1.05 (estimates) from (b), combined with METR's
family→task→run resampling. Same code and 500 replicates as (a0), so they're directly
comparable.

**Result: the independent-noise floor roughly quadruples, but the trend holds.**

- Per-agent p50 CIs widen by a **median 13.6%** in log-width across the 17 frontier
  agents (vs a0's 3.8% floor) — the extra width comes almost entirely from noising the
  long frozen tasks that a0 couldn't touch. The most-affected agents shift too
  (Grok 4, gpt-3.5-turbo-instruct, Claude 4.1 Opus lead now, not a0's o1-preview).
- The **doubling-time** 95% CI: [171.7, 224.4] days (`metr`) → [168.8, 223.5]
  (`metr+x_param`), only ~2 days wider; the pure-x doubling spread is ±5 days. Even with
  *every* task independently perturbed at full per-task σ, independent x-noise still
  averages out across the frontier regression.

![parametric CI comparison](figures/a_ci_comparison.png)

![parametric doubling time](figures/a_doubling_time.png)

Code: [`a_parametric_xboot.py`](a_parametric_xboot.py) / [notebook](a_parametric_xboot.ipynb).
Outputs: [`data/a_ci_by_agent.csv`](data/a_ci_by_agent.csv),
[`data/a_doubling_times.csv`](data/a_doubling_times.csv).

## (c) — Coherent ±1σ shift (systematic-error worst case)

The bootstraps ((a0)/(a)) treat x-noise as *independent* across tasks, so it averages
out. A **systematic** error — every task biased the same direction (baseliner
selection, a shared convention, the flooring scheme) — does not. We bound it by shifting
*all* tasks' `human_minutes` coherently by ±1 σ̃_task in log space and refitting.

**Result: absolute horizons swing hugely; the trend barely moves.**

- A coherent ±1σ̃ bias multiplies the frontier geo-mean p50 horizon by **×2.23 / ×0.45**
  (16.4 min → 36.7 / 7.4 min): the *absolute* horizon is extremely sensitive to any
  systematic error in the baseline times.
- But the **doubling time** moves only **−6.5% / +7.4%** (201 → 188 / 216 days), because
  a coherent shift is nearly a uniform rescale, which slides log-horizons vertically
  without changing the slope. The small residual asymmetry is real: the shift is larger
  for the long HCAST/RE-Bench tasks than for short SWAA, so pushing all tasks *up*
  steepens the recent frontier slightly (faster doubling).
- The **p80/p50 ratio is preserved** (0.235–0.255 across scenarios) — a pure shift moves
  both quantiles by the same factor. This is the systematic-error counterpart to the
  errors-in-variables *attenuation* in (d), which instead widens the p50/p80 gap.

This is the quantitative hook for **Stage 4**: baseliner selection is precisely a
coherent shift, and (c) shows it would badly bias the absolute horizon claims while
sparing the doubling-time trend.

![coherent-shift doubling sensitivity](figures/c_doubling_sensitivity.png)

![coherent-shift horizon levels](figures/c_horizon_levels.png)

Code: [`c_coherent_shift.py`](c_coherent_shift.py) / [notebook](c_coherent_shift.ipynb).
Output: [`data/c_coherent_shift.csv`](data/c_coherent_shift.csv).

## (d) — SIMEX with per-task σ

Wider CIs aren't the whole story: x-noise *attenuates* the logistic slope, biasing the
horizon **point estimates**. SIMEX corrects it — add known noise at λ = 0.5…2, watch the
horizon drift, extrapolate to λ = −1 (zero noise). We run METR's own global-σ version and
our per-task-σ version through the identical pipeline. (Validation: the λ=0 fits
reproduce METR's published p50/p80 exactly.)

**Result: per-task σ makes the correction slightly *larger*, overturning our initial
guess.**

- For the newest frontier agent (Claude Opus 4.5), SIMEX moves p50 **−16.6%**
  (global σ) → **−18.8%** (per-task σ), and p80 **+24.0% → +26.3%**. The direction and
  rough magnitude reproduce METR's reported Opus 4.6 result (p50 −36% on the larger
  private suite; p80 +9%) — the p50 magnitude differs because it's a different agent and
  the v1.0 suite.
- Across all 17 frontier agents the median correction is p50 +4.1% → +3.4%,
  p80 +29.6% → +33.0% (global → per-task): per-task σ deepens the p50 haircut and
  enlarges the p80 rise for the recent, long-horizon agents. The reason is exactly the
  (b) finding: HCAST/RE-Bench noise (σ̃≈0.89) exceeds METR's global 0.78, and those tasks
  drive the frontier. **So METR's published SIMEX, if anything, slightly *understates*
  the errors-in-variables correction — not overstates it, as we'd hypothesised.**
- The p50/p80 split is the errors-in-variables signature: attenuation flattens the slope,
  which lowers p50 and raises p80, widening the gap — the mechanism behind the p50/p80
  divergence noted in the intro.

Caveat: the quadratic extrapolant to λ=−1 is itself uncertain (METR emphasised wide
ranges), and SIMEX assumes *independent* noise; the systematic component is (c).

![SIMEX correction across agents](figures/d_simex_correction.png)

![SIMEX extrapolation curves](figures/d_simex_curves.png)

Code: [`d_simex.py`](d_simex.py) / [notebook](d_simex.ipynb). Outputs:
[`data/d_simex_curves.csv`](data/d_simex_curves.csv),
[`data/d_simex_corrections.csv`](data/d_simex_corrections.csv).
