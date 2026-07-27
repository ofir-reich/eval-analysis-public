# Reanalysis of METR's Time Horizon data

Independent reanalysis of the human-baseline side of METR's
[time horizon methodology](https://arxiv.org/abs/2503.14499)
([upstream repo](https://github.com/METR/eval-analysis-public)).
Everything lives in this `reanalysis/` directory; the rest of the repo is unmodified upstream.

**The theme:** the x-axis of the whole methodology — `human_minutes`, how long a task takes
a human expert — is a measured quantity with structure and error that the headline analysis
treats as exact. We reconstruct how it was measured, quantify its uncertainty, and propagate
that uncertainty through to time horizons and doubling times.

## Stages

| Stage | Question | Status |
|---|---|---|
| [1 — Export archaeology](stage-1-export-archaeology/) | What exactly is in the public export, and can per-run human times be recovered? | ✅ done (v1.0 + v1.1) |
| [2 — Measurement error](stage-2-measurement-error/) | How does per-task uncertainty in `human_minutes` propagate to horizons and doubling times? | ✅ (a0,b,a,c,d; v1.0 + v1.1) |
| [3 — Survival analysis](stage-3-survival/) | What happens when failed human baselines are treated as censored observations instead of discarded? | ✅ done (v1.0 + v1.1) |
| [4 — Cohort selection](stage-4-cohort-selection/) | Are long-task baseliners systematically faster, *tilting* the x-axis and hence the trend? | proposal — needs pseudonymous baseliner IDs |

## Results so far, in one paragraph

The trend survives; the absolute numbers do not. **Nothing we tried moves the doubling
time by more than ~6%** — random per-task noise averages out across ~170 tasks (its 95%
interval widens by ~2 days even when *every* task is perturbed at full estimated noise),
and even a coherent ±1σ bias applied to all tasks at once shifts it under ±10%, because
rescaling every task slides the horizon-vs-date line without tilting it. The **absolute
horizons are a different story**: the same coherent ±1σ bias multiplies them by ×2.4 or
×0.45, and two specific corrections move them substantially. METR's own SIMEX
errors-in-variables correction *lowers* the most capable agent's 50%-horizon (−19% on
v1.0, −36% on v1.1, the latter matching their published figure) and raises every agent's
80%-horizon by 25–40%; our never-before-applied censoring correction pushes the other way,
*raising* the 50%-horizon by 23–45% by treating the 28% of human baseline attempts that
failed as right-censored rather than discarding them. These two are comparable in size and
opposite in sign, and which one wins depends on the suite (net +18% on v1.0, −21% on
v1.1) — so the honest summary is that the headline horizon carries roughly **±20%
unresolved uncertainty from the x-axis alone**, while the doubling time is robust. Along
the way we found that the 16 tasks whose `human_minutes` is a researcher estimate are
*exactly* the 16 tasks where every human baseliner failed, and that 7 of those published
values are **provably too low** by arithmetic alone. The one error structure that could
still move the trend — a length-dependent *tilt* from baseliner self-selection — needs one
field METR has not released, which is what [Stage 4](stage-4-cohort-selection/) asks for.

## Relation to prior work

METR's own
[modelling-assumptions note](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/)
applied a SIMEX noise correction with a *global* noise assumption, and their
[limitations note](https://metr.org/notes/2026-01-22-time-horizon-limitations/) explicitly
lists failed-baseline survival analysis and baseliner selection as open. Stage 2 refines the
SIMEX correction with per-task empirical noise estimates; Stage 3 does the failed-baseline
survival analysis for the first time, and finds it opposes SIMEX at comparable magnitude —
so the two together leave the headline horizon uncertain by roughly ±20% rather than
confidently revised downward.

## Reproducing

```bash
uv sync --all-extras            # from repo root (Python ≥3.11)
cd reanalysis/stage-1-export-archaeology
python analysis.py              # or open analysis.ipynb
```

Each stage directory contains: `README.md` (summary + key figures), `analysis.py`
(jupytext py:percent source of truth), `analysis.ipynb` (paired, executed), `figures/`,
and `data/` (derived datasets).

*By [Ofir Reich](https://github.com/ofir-reich), with Claude as research assistant.*
