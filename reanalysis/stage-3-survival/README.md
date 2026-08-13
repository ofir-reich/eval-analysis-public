# Stage 3 — Failed human baselines as right-censored observations

`human_minutes` is the geometric mean of the completion times of the baseliners who **succeeded**. The ones who failed are dropped — but the export still records how long they worked. A failure after 8 hours is not missing data. It is the observation **"this person's completion time exceeds 8 hours"**: a textbook **right-censored** observation.

Discarding them is complete-case analysis of censored data, which biases the estimate *downwards*, because the discarded attempts are precisely the slow ones. This correction has not been done before: METR's [limitations note](https://metr.org/notes/2026-01-22-time-horizon-limitations/) lists it as open, and the existing Weibull survival work in this space models *agent* runs, not human baselines.

The correction comes in three tiers of escalating assumptions:

| tier | method | assumption |
|---|---|---|
| 1 | **hard bounds** — the geometric mean of censoring times bounds the true one from below | none |
| 2 | **Kaplan–Meier** — nonparametric survival of completion time | non-informative censoring |
| 3 | **censored-lognormal MLE**, σ̃ from [Stage 2's σ analysis](../stage-2-measurement-error/) | lognormal, σ known |

## Bottom line

- **Over a quarter of all human baseline attempts ended in failure, and the published numbers ignore them.** A failure is information — the person needed *more* time than they spent — so dropping failures biases the published task times low. The failures concentrate on exactly the long tasks that set the top-end horizons.
- **Correcting for them raises the absolute time horizons substantially.** Across correction methods, horizons rise by 28% to 47%; the most capable agent's 50%-horizon rises ~35% under the cleanest ("pure censoring") variant. The doubling time barely moves — it shortens by 4% to 5%.
- **This correction and METR's own SIMEX correction pull in opposite directions and largely cancel.** The net change to the newest agent's 50%-horizon is **+10% to +18% on v1.0, and −21% to −25% on v1.1** — which way the balance tips depends on the task suite. Reporting the SIMEX correction alone would therefore overstate how confidently the horizon should be revised downward.
- **The researcher-estimated tasks are exactly the tasks where every baseliner failed.** The estimates are fallbacks after attempted baselining, not guesses about tasks nobody tried — but that also means they stand in for tasks humans could not finish, and for 7 of the 16 the durations of the recorded failed attempts already exceed the published value, implying it is too low.

## How much censoring is there?

**226 of 793 baseline runs — 28.5% — ended in failure.** Failure is common on HCAST (38% of runs) and RE-Bench (48%) and rare on SWAA (under 3%). These were not quick give-ups: the median failed attempt lasted almost two and a half hours, and the longest ran for days. In all, the export records about **1,277 hours of discarded human effort**, 388 of them on the 16 tasks discussed below, where *every* baseliner failed. Twenty-five runs stopped at exactly 479 minutes — the visible signature of an **8-hour administrative cap**.

The censoring is also concentrated exactly where it does the most damage. The human success rate falls from near-certain on sub-minute tasks to about 40% on tasks of 4 to 16 hours, and to 20% beyond that. The long tasks that determine the top-end horizons are the ones whose `human_minutes` rests on the most heavily filtered data.

![success rate by length](figures/success_rate_by_length.png)

## The "estimate" tasks are the all-failure tasks

16 tasks have **zero** successful baseline runs. That set is **exactly** the set of HCAST tasks whose `human_minutes` carries `human_source == "estimate"` — the correspondence is one-to-one in both directions. METR fell back to a researcher's estimate precisely when *every* baseliner failed.

Two things follow. First, the estimates are not exogenous guesses — baselining was attempted on every one of these tasks, and the estimate is the fallback. Second, they stand in for tasks humans **could not finish**, and the export contains the record of how long people tried before giving up. 8.5% of all agent runs are scored against these 16 estimates.

## Tier 1 — hard lower bounds from the failed attempts

If every baseliner failed at times $c_1 \dots c_n$, then each true completion time satisfies $T_i > c_i$, so $\operatorname{gmean}(T) > \operatorname{gmean}(c)$. This needs no distributional model, and it can be checked against the published value:

- **7 of the 16 all-failure tasks publish a `human_minutes` below this hard lower bound.** For those tasks the published value is understated by a median factor of 2.9.
- **13 of the 16** had at least one baseliner who worked longer than the published estimate without solving the task.
- The worst case is `blackbox/apron`, published at **10 minutes**: eight baseliners failed it, one of them only giving up after **368 minutes**, and the bound alone puts the true value at 30 minutes or more.

![lower bounds](figures/lower_bounds_all_censored.png)

## Tier 2 — Kaplan–Meier

Treating failures as right-censored and pooling tasks within length buckets, the [Kaplan–Meier median](figures/kaplan_meier_by_length.png) runs about **1.5× the naive geometric mean of the successes** in each of the well-populated middle buckets (4 minutes to about 4 hours).[^kmbuckets] This is a distribution-free corroboration of the parametric correction below.

[^kmbuckets]: The bucket ratios are 1.35×, 1.54×, and 1.49× for the 4–16, 16–64, and 64–256 minute buckets. The 256–960 minute bucket dips below 1.0 because pooling there spans a 4× range of true task lengths — read the buckets as coarse checks, not per-task estimates.

## Tier 3 — censored-lognormal MLE

For a per-task point estimate, model completion time as lognormal and take **σ from Stage 2's σ analysis** — which is what makes this tractable: with σ known, μ is a one-parameter fit, stable even on a task with one success and several censored runs. The censored terms can only push μ up, so the correction is one-directional by construction.

**The correction factor — the MLE over the naive geometric mean — has a median of 1.47× across the 59 correctable tasks**, a 75th percentile of about 2×, and a long tail reaching 9.4×. On tasks longer than an hour the corrected `human_minutes` comes out at roughly 1.4 to 1.5 times the published value.

The extreme case is instructive. `blackbox/acorn` publishes **12.0 minutes** — the time of its *single* successful baseliner — while **six of its nine failed baseliners worked longer than 12 minutes** without solving it. The recorded attempts alone show that 12 minutes understates the typical completion time; the MLE's 113 minutes is one defensible answer.

Note the identifiability limit: with **no** successes the likelihood rises without bound in μ, so the 16 all-failure tasks have no MLE. For them Tier 1's arithmetic floor is all the data supports.

The per-task view — each bubble is one task, sized by how many of its runs were failures:

![correction factors](figures/correction_factor_by_task.png)

## Propagating to horizons and the doubling time

Six x-axis scenarios through METR's own weighted-logistic fit (the `published` scenario reproduces their headline p50/p80 exactly; the SOTA-at-release agent set is held fixed at METR's published membership throughout):

| scenario | p50, geo-mean over SOTA-at-release agents | p80, same | Claude Opus 4.5 p50 | doubling time |
|---|---|---|---|---|
| `published` | 16.4 min | 4.0 min | 246.8 min | 201.1 d |
| `censored_ratio` | ×1.28 | ×1.32 | ×1.36 | 193.1 d (−4.0%) |
| `censored_ratio + bounds` | ×1.35 | ×1.47 | ×1.35 | 191.4 d (−4.8%) |
| `censored_mle` | ×1.30 | ×1.31 | ×1.47 | 192.2 d (−4.5%) |
| `censored_mle + bounds` | ×1.37 | ×1.46 | ×1.45 | 190.6 d (−5.2%) |
| `max_impute` (crude) | ×1.39 | ×1.43 | ×1.59 | 189.3 d (−5.9%) |

Two scenario ingredients need defining. `+ bounds` additionally raises the 16 all-failure tasks — which have no MLE — to their Tier-1 hard lower bound; `max_impute` is a deliberately crude upper reference that gives every failed run its task's longest observed time and recomputes the geometric mean.

The two `censored_*` families differ in how the Tier-3 fit is applied, and the difference only matters for RE-Bench. `censored_mle` substitutes the MLE **level** — the corrected gmean of elapsed completion times. For HCAST and SWAA that is exactly the published convention (Stage 1), but RE-Bench's published `human_minutes` is *not* a gmean of elapsed times (8h-capped sessions, Stage 1 §1), so substituting the elapsed-time MLE there bundles the censoring correction with a **convention switch** — its ×1.47 on Opus 4.5 is not purely a censoring effect. `censored_ratio` instead multiplies each published value by the task's censoring **factor** (MLE ÷ naive gmean of successes), keeping every task in its published convention: the pure-censoring number is **×1.36**. Both are shown because both questions are legitimate — "what does censoring alone do?" (ratio) and "what would a fully elapsed-time-consistent x-axis look like?" (level).

**Across scenarios the horizons rise by roughly 28 to 47%, while the doubling time shortens by only 4 to 5%.** Claude Opus 4.5's 50%-horizon goes from the published 247 minutes to between 333 minutes (pure censoring) and 357 minutes (elapsed-time-consistent). The doubling time shortens slightly because long tasks stretch more than short ones, so recent agents gain more than old ones and the fitted slope steepens. This is consistent with Stage 2: the x-axis moves levels far more than it moves the doubling time. The same numbers as bar charts: [horizons by scenario](figures/horizon_by_scenario.png) · [doubling time by scenario](figures/doubling_by_scenario.png).

## How much rides on the borrowed σ?

σ is the one parameter Tier 3 imports rather than estimates. Rescaling the Stage-2 value by 0.75× and 1.5× moves the aggregate p50 correction from its central ×1.30 down to ×1.25 and up to ×1.41 ([full table](data/sigma_sensitivity.csv)). The correction *grows* with σ: a wider distribution makes each exact success weaker evidence about the location, so the censored observations dominate more.

That direction matters for reading the headline. σ̃ is estimated from **successful** runs only, so if failures are drawn from a heavier-tailed part of the distribution the true σ is *larger* than 0.89 — making our central estimate the **conservative** one.

## Net effect: censoring correction vs SIMEX

These are the two largest known x-axis corrections and they point in **opposite directions**. Applying both — the first time that has been done — to each suite's most capable agent, with the censoring correction in both variants:

| correction on most-capable-agent p50 | v1.0 (Claude Opus 4.5) | v1.1 (Claude Opus 4.6) |
|---|---|---|
| SIMEX errors-in-variables ([Stage 2](../stage-2-measurement-error/), per-task σ) | ×0.812 (−18.8%) | ×0.641 (−35.9%) |
| censoring, pure (`censored_ratio + bounds`) | ×1.349 (+34.9%) | ×1.171 (+17.1%) |
| **→ net** | **×1.096 (+9.6%)** | **×0.750 (−25.0%)** |
| censoring, elapsed-time-consistent (`censored_mle + bounds`) | ×1.447 (+44.7%) | ×1.227 (+22.7%) |
| **→ net** | **×1.175 (+17.5%)** | **×0.786 (−21.4%)** |

(The multiplicative composition assumes the two corrections commute; SIMEX rerun on a censoring-corrected x-axis, with σ re-estimated from it, would come out somewhat different. Read the net as indicative, not as a precise revised headline.)

**The two corrections are of comparable size and substantially cancel — but the sign of what is left over is not robust.** On v1.0 the censoring correction wins and the headline p50 should be revised *up* 10–18%; on v1.1, where SIMEX bites much harder (−36%, close to METR's own published figure), SIMEX wins and the net is *down* 21–25%.

The defensible claim is therefore not a specific net number: **the SIMEX correction is opposed by a censoring correction of similar magnitude** that had not previously been applied, and once both are in play the residual uncertainty in the headline horizon is roughly ±20% with a suite-dependent sign. Reporting the SIMEX correction alone would overstate how confidently the horizon should be revised downward.

## Extending to v1.1

`DATASET_VERSION=1-1` reruns everything on the 228-task v1.1 suite (agents from GPT-4 onward). Every qualitative finding carries over, including the structural one:

| | v1.0 | v1.1 |
|---|---|---|
| censoring rate | 226/793 (28.5%) | 216/773 (27.9%) |
| all-failure tasks | 16 | 23 |
| …identical to the HCAST `estimate` set? | **yes** | **yes** |
| understated per the Tier-1 bound | 7/16, median 2.9× | 10/23, median 2.44× |
| Tier-3 correction factor (median) | 1.47× | 1.54× |
| p50, geo-mean over SOTA-at-release agents (mle+bounds / ratio+bounds) | ×1.37 / ×1.35 | ×1.27 / ×1.26 |
| most-capable-agent p50 (mle+bounds / ratio+bounds) | ×1.45 / ×1.35 | ×1.23 / ×1.17 |
| doubling time (mle+bounds / ratio+bounds) | −5.2% / −4.8% | −1.8% / −1.4% |
| σ sensitivity (0.75/1/1.5×) | ×1.25 / ×1.30 / ×1.41 | ×1.18 / ×1.21 / ×1.30 |

The one-to-one correspondence between "all baseliners failed" and `human_source == "estimate"` holding on both suites confirms it is a deliberate pipeline rule, not a coincidence of task selection.

## Caveats

1. **Informative censoring.** Both Tier 2 and Tier 3 assume the decision to stop is independent of how much longer the person needed. Voluntary give-ups almost certainly violate this — someone who senses a task is hopeless quits earlier — which would make these corrections **understate** the truth. The 8-hour cap (25 runs) is genuinely non-informative; give-ups are not.
2. **σ borrowed from successes** — see the sensitivity above; the bias runs conservative.
3. **All-failure tasks are bounds, not estimates.** `censored_mle + bounds` raises them to the Tier-1 arithmetic floor; the true values lie somewhere above it by an unknown amount.
4. Tier 1's bound assumes only that a failed attempt implies a longer true completion time — i.e. that the person would have succeeded eventually. For a task a given person would *never* complete, the true value is unbounded, which again makes the bound conservative.
5. **The 10-second safety clip is inert.** A few runs are floored to 0 minutes (v1.0: 4, of which 1 censored); where Stage 1 has no δ they are clipped to 10 s purely to keep log(time) finite. The one clipped *censored* run sits on an all-failure task whose Tier-1 bound is ~0.07× the published value under any treatment of that run, so the set of tasks understated per the Tier-1 bound does not depend on a clipped value.

Code: [`analysis.py`](analysis.py) (jupytext; paired executed notebook [`analysis.ipynb`](analysis.ipynb)) — runs on v1.1 too via `DATASET_VERSION=1-1`. Outputs: [`data/survival_by_task.csv`](data/survival_by_task.csv), [`data/lower_bounds_all_censored.csv`](data/lower_bounds_all_censored.csv), [`data/scenario_summary.csv`](data/scenario_summary.csv), [`data/sigma_sensitivity.csv`](data/sigma_sensitivity.csv).
