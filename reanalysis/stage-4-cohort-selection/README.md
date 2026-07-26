# Stage 4 — Cohort selection: is the x-axis tilted?

**Status:** proposal. Unlike Stages 1–3 this cannot be done with the public export — it
needs one additional field, pseudonymous baseliner IDs.

## The hypothesis

Baseliners were not randomly assigned to tasks — they picked up work, and the people who
took on (and finished) the 8-hour tasks are plausibly not drawn from the same skill
distribution as those who did the 5-minute ones. If the long-task cohort is systematically
faster, then every long task's `human_minutes` is measured against a stronger reference
human than every short task's, and the x-axis is not merely biased — it is **tilted**.

That distinction is what makes this the most consequential open question we have found.
Every error structure examined in [Stage 2](../stage-2-measurement-error/) was either
independent noise, which averages out (perturbing all 170 tasks widened the doubling-time
CI by ~2 days), or a coherent shift, which moves absolute horizons a lot but the trend
barely at all (a full ±1σ shift moves the frontier horizon ×2.4 yet the doubling time under
±10%, because a uniform rescale slides log-horizons without changing the slope). A
**length-dependent tilt is the one error structure that changes the slope** — so it bears
directly on the doubling time and the extrapolated forecasts, which have survived every
correction in Stages 1–3.

[Stage 3](../stage-3-survival/) shows selection is already operating and is strongly
length-dependent: baseliner success rate falls from 97.5% on sub-minute tasks to **20%
beyond 960 minutes**, so on long tasks `human_minutes` is defined by a shrinking successful
minority. Stage 3 corrected the **within-task** form of that filtering (failures as
right-censored observations). The **between-person** form — which baseliners even attempted
which tasks — is invisible in the public export, which ships no baseliner identifiers.

## The test, given IDs

Fit crossed random effects over the human runs,

```
log t_ij = mu_task_j + pi_person_i + eps_ij
```

recover a per-person speed factor `pi` from baseliners who overlap across tasks, then
re-express every task's time relative to a common reference human and refit the horizons
and doubling time. If `pi` correlates with task length, the tilt is real and quantified. If
it does not, that is a reassuring null that **strengthens** the published methodology by
closing a limitation METR's own
[limitations note](https://metr.org/notes/2026-01-22-time-horizon-limitations/) lists as
open. Both outcomes are informative.

## Data request

**Pseudonymous baseliner IDs** on human runs — an opaque, stable identifier per person
(e.g. `baseliner_0417`), enough to tell *same person* from *different person* across tasks.
No names, demographics, or other attributes needed. That single field is the whole ask.

Happy to work under any arrangement METR prefers, including a restricted or on-premise
setup with only aggregate results published, and to share results before publication.

### Nice to have

- A flag distinguishing "gave up" from "hit the time cap", which would let Stage 3 drop its
  informative-censoring caveat rather than just acknowledge it.
- Any notes on how baseliners were matched to tasks, even informal.
