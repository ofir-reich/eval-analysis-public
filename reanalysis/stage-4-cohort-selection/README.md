# Stage 4 — Cohort selection: is the x-axis tilted?

**Status:** proposal. Unlike Stages 1–3, this one cannot be done with the public export —
it needs one additional field from METR (pseudonymous baseliner IDs). The argument for
why it is worth running is below.

## The hypothesis

Baseliners were not randomly assigned to tasks — they picked up work, and the people who
took on (and finished) the 8-hour tasks are plausibly not drawn from the same skill
distribution as those who did the 5-minute ones. If the long-task cohort is systematically
faster, then every long task's `human_minutes` is measured against a stronger reference
human than every short task's, and the x-axis is not merely biased — it is **tilted**.

That distinction is what makes this the most consequential open question we have found.
Every error structure examined in [Stage 2](../stage-2-measurement-error/) was either
independent noise, which averages out (perturbing all 170 tasks widened the doubling-time
CI by ~2 days), or a coherent shift, which moves absolute horizons dramatically but leaves
the trend nearly untouched (a full ±1σ shift moves the frontier horizon ×2.4 but the
doubling time under ±10%, because a uniform rescale slides log-horizons vertically without
changing the slope). A **length-dependent tilt is the one error structure that changes the
slope** — so it bears directly on the doubling time and on the extrapolated forecasts,
which have so far survived every correction in Stages 1–3.

## Why the public data already points this way

[Stage 3](../stage-3-survival/) established that selection is operating and is strongly
length-dependent: human baseliner success rate falls from 97.5% on sub-minute tasks to
**20% beyond 960 minutes**. On long tasks `human_minutes` is therefore defined by a
shrinking successful minority, and on 7 of the 16 tasks where *every* baseliner failed the
published value is provably below what the recorded attempt durations already imply.

Stage 3 corrected the **within-task** form of that filtering (failures as right-censored
observations). The **between-person** form — which baseliners even attempted which tasks —
is invisible in the public export, because it ships no baseliner identifiers.

## The test, given IDs

With pseudonymous per-baseliner IDs the analysis is standard and cheap. Fit crossed random
effects over the human runs,

```
log t_ij = mu_task_j + pi_person_i + eps_ij
```

recover a per-person speed factor `pi` from baseliners who overlap across tasks, then
re-express every task's time relative to a single common reference human and refit the
horizons and the doubling time. If `pi` correlates with task length, the tilt is real and
now quantified. If it does not, that is a genuinely reassuring null result that
**strengthens** the published methodology by closing a limitation METR's own
[limitations note](https://metr.org/notes/2026-01-22-time-horizon-limitations/) lists as
open. Both outcomes are informative, which is what makes the request worth the effort.

## Data request

In descending order of value:

1. **Pseudonymous baseliner IDs** on human runs — an opaque, stable identifier per person
   (e.g. `baseliner_0417`), enough to tell *same person* from *different person* across
   tasks. No names, demographics, or other attributes needed. This is the one field that
   unlocks the analysis; nothing else here is essential.
2. **Unfloored per-run durations.** The public export floors HCAST/RE-Bench run times to
   whole minutes ([Stage 1](../stage-1-export-archaeology/) reconstructs a per-task offset
   to undo the resulting bias, exactly but only in aggregate). Full-precision times would
   remove that reconstruction step and sharpen the per-task spread estimates in
   [Stage 2(b)](../stage-2-measurement-error/).
3. **A failure-reason flag** distinguishing "gave up" from "hit the time cap." Stage 3's
   main caveat is informative censoring: survival methods assume stopping is independent of
   remaining time, which is defensible for the 8-hour administrative cap (25 runs at 479
   minutes) but not for voluntary give-ups. This flag would let that caveat be dropped
   rather than merely acknowledged.
4. **Baseliner recruitment/assignment notes** — how people were matched to tasks. Even
   informal prose would sharpen the priors on how much selection to expect.

Happy to run the analysis on data under any arrangement METR prefers, including a
restricted or on-premise setup with only aggregate results published, and to share results
before publication.
