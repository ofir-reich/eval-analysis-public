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

## Results so far

Short version: **the trend is robust, the absolute horizons are not.**

- **The doubling time barely moves — under ~6% in every scenario we tried.** Random
  per-task noise averages out over ~170 tasks, and even a coherent bias applied to all
  tasks at once mostly cancels, because rescaling every task slides the horizon-vs-date
  line without tilting it.
- **Absolute horizons are a different story.** A coherent ±1σ error in the baseline times
  moves them by more than a factor of two in either direction, so the headline "time
  horizon" numbers are far more fragile than the growth rate.
- **Two real corrections pull in opposite directions.** METR's own errors-in-variables
  (SIMEX) correction *lowers* the most capable agent's 50%-horizon and raises everyone's
  80%-horizon; treating failed human baselines as censored rather than discarding them
  *raises* the 50%-horizon by a comparable amount. Which one wins depends on the task
  suite, so the headline horizon carries roughly **±20% of unresolved uncertainty from the
  x-axis alone** — METR's published downward correction is not the last word.
- **Two things are simply wrong in the published data.** The tasks whose `human_minutes`
  is a researcher estimate turn out to be *exactly* the tasks where every human baseliner
  failed — and for several of them the recorded attempt durations already prove the
  published value is too low, by arithmetic alone.
- **One open risk could still move the trend:** if the people who took the long tasks were
  systematically faster, the x-axis is *tilted* rather than shifted, and a tilt does change
  the slope. Testing that needs one field METR has not released — see
  [Stage 4](stage-4-cohort-selection/).

Per-stage numbers and figures are in the stage READMEs linked above.

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
