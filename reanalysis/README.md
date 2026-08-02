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
| [4 — Cohort selection](stage-4-cohort-selection/) | Are long-task baseliners systematically faster, *tilting* the x-axis and hence the doubling time? | proposal — needs pseudonymous baseliner IDs |

## Results so far

Short version: **the doubling time is robust, the absolute horizons are not.**

- **The doubling time barely moves — under ~10% in every scenario we tried.** Random
  per-task noise averages out over ~170 tasks, and even a coherent bias applied to all
  tasks at once mostly cancels, because rescaling every task slides the horizon-vs-date
  line without tilting it.
- **Absolute horizons are a different story.** A coherent ±1σ error in the baseline times
  multiplies them by **×2.3 or ×0.44**, so the headline "time horizon" numbers are far
  more fragile than the doubling time.
- **Two real corrections pull in opposite directions.** METR's own errors-in-variables
  (SIMEX) correction *lowers* the most capable agent's 50%-horizon and raises everyone's
  80%-horizon; treating failed human baselines as censored rather than discarding them
  *raises* the 50%-horizon by a comparable amount. Which one wins depends on the task
  suite, so the headline horizon carries roughly **±20% of unresolved uncertainty from the
  x-axis alone** — METR's published downward correction is not the last word.
- **The researcher-estimated tasks deserve a second look.** They turn out to be *exactly*
  the tasks where every human baseliner failed. The export is upfront that these are
  estimates rather than baselines, so this is disclosed rather than hidden — but the
  correspondence itself is undocumented, and for several of those tasks the recorded
  durations of the failed attempts already exceed the published estimate, which suggests
  the estimates are low relative to METR's own data.
- **One open risk could still move the doubling time:** if the people who took the long
  tasks were systematically faster, the x-axis is *tilted* rather than shifted, and a tilt
  does change the slope. Testing that needs one field METR has not released — see
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

Everything is a plain Python script — no notebook interaction needed. The only ordering
constraints: Stage 1 writes the per-run CSVs that Stage 2 consumes, and Stage 2(b)
writes the per-task σ that (a), (c), (d), and Stage 3 consume.

```bash
uv sync --all-extras                     # from repo root (Python ≥3.11)
cd reanalysis/stage-1-export-archaeology && python analysis.py
cd ../stage-2-measurement-error
python b_sigma_task.py                   # per-task σ — must run before a/c/d
python a0_nonparametric_xboot.py         # v1.0 only (the floor that (a) supersedes)
python a_parametric_xboot.py
python c_coherent_shift.py
python d_simex.py
cd ../stage-3-survival && python analysis.py
```

- **v1.1 replication:** rerun any Stage-2/3 script with `DATASET_VERSION=1-1`
  (outputs get a `_v1_1` suffix; Stage 1 handles both suites in one run; a0 is
  v1.0-only).
- **Caches:** the committed `data/` CSVs double as caches for the expensive steps (the
  bootstraps in a0/a, the SIMEX curves in d) — delete a cache file to force a
  recompute; seeds are fixed, so a recompute reproduces it exactly. `N_BOOT` and
  `N_SIMEX` env vars trade runtime for precision.

Each stage directory contains: `README.md` (summary + key figures), `analysis.py`
(jupytext py:percent source of truth), `analysis.ipynb` (paired, executed), `figures/`,
and `data/` (derived datasets).

*By [Ofir Reich](https://github.com/ofir-reich), with Claude as research assistant.*
