# Stage 1 — Export archaeology: recovering per-run human baseline times

**TL;DR:** In the public export, the durations of human baseline runs were floored to the
whole minute before release, while the official per-task `human_minutes` (computed upstream
from the unfloored times) kept full precision. We demonstrate this, recover the lost
sub-minute structure exactly at the task level, and publish a corrected per-run dataset
that downstream analyses (and other researchers) can use directly.

## The finding

`human_minutes` is defined as the geometric mean of successful human baseliners' completion
times. Trying to re-derive it from the export's per-run timestamps fails by up to ~20% on
short tasks. Three facts explain everything:

1. **HCAST and RE-Bench durations are whole minutes** (100% of 558 runs); SWAA durations
   are stored to the millisecond (0% whole minutes). The flooring bites short tasks
   proportionally hardest.
2. **For all 21 single-baseliner tasks, the stored duration equals `floor(human_minutes)`
   exactly** — for n=1 the aggregate *is* that run's time, so this pins down the scheme:
   a floor (not a round), applied to the per-run times only. The official numbers have
   full precision; only the export is lossy.
3. **The flooring bracket `gmean(t) ≤ human_minutes < gmean(t+1)` holds for 81/81 HCAST
   baselined tasks** — including three successful runs stored as *0 minutes* (dropping
   those, instead of keeping them, is a trap: one zero collapses a geometric mean).

![bracket](figures/bracket.png)

## The fix: δ_task

The bracket guarantees a unique per-task offset δ_task ∈ [0,1) with
`gmean(t + δ_task) = human_minutes`. Solving it recovers the official aggregate to
**max residual 5×10⁻¹⁴** across all 81 tasks:

![before/after](figures/before_after_ecdf.png)

δ is a *shared per-task* constant — the true per-run sub-minute residuals are gone, so
treat corrected times as interval data `[t, t+1)` where sub-minute spread matters.

## The deliverable

[`data/human_runs_derived.csv`](data/human_runs_derived.csv) — all 793 human baseline runs
with `minutes_derived` and an explicit `derivation` flag:

| derivation | n | meaning |
|---|---|---|
| `delta_corrected` | 291 | HCAST success: floored + δ_task (aggregate-exact) |
| `swaa_exact` | 235 | SWAA: ms precision, no correction needed |
| `delta_imputed_failure` | 117 | HCAST failure: δ_task as best guess (δ identified from successes only) |
| `uncorrected_rebench` | 91 | RE-Bench: different time convention; no δ ∈ [0,1) exists |
| `uncorrected_no_delta` | 59 | HCAST runs on tasks with no successful baseline |

Full derivation with code: [`analysis.ipynb`](analysis.ipynb) (source: `analysis.py`,
jupytext-paired). Data: Time Horizon v1.0 (the paper's suite).
