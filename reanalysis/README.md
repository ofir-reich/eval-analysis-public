# Reanalysis of METR's Time Horizon data

Independent reanalysis of the human-baseline side of METR's [time horizon methodology](https://arxiv.org/abs/2503.14499) ([upstream repo](https://github.com/METR/eval-analysis-public)). Everything lives in this `reanalysis/` directory; the rest of the repo is unmodified upstream.

**The theme:** the x-axis of the whole methodology — `human_minutes`, how long a task takes a human expert — is a measured quantity with structure and error that the headline analysis treats as exact. We reconstruct how it was measured, quantify its uncertainty, and propagate that uncertainty through to time horizons and doubling times.

## Stages

| Stage                                                 | Question                                                                                            | Status                                      |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| [1 — Export archaeology](stage-1-export-archaeology/) | What exactly is in the public export, and can per-run human times be recovered?                     | ✅ done (v1.0 + v1.1)                        |
| [2 — Measurement error](stage-2-measurement-error/)   | How does per-task uncertainty in `human_minutes` propagate to horizons and doubling times?          | ✅ done (v1.0 + v1.1)                        |
| [3 — Survival analysis](stage-3-survival/)            | What happens when failed human baselines are treated as censored observations instead of discarded? | ✅ done (v1.0 + v1.1)                        |
| [4 — Cohort selection](stage-4-cohort-selection/)     | Are long-task baseliners systematically faster, *tilting* the x-axis and hence the doubling time?   | proposal — needs pseudonymous baseliner IDs |

## Results so far

Short version: **the doubling time is robust, the absolute horizons are not.**

- **The doubling time barely moves — under ~10% in every scenario we tried.** Random per-task noise averages out over ~170 tasks, and even a coherent bias applied to all tasks at once mostly cancels, because rescaling every task slides the horizon-vs-date line without tilting it.
- **Absolute horizons are a different story.** A coherent ±1σ error in the baseline times multiplies them by **×2.3 or ×0.44**, so the headline "time horizon" numbers are far more fragile than the doubling time.
- **SIMEX and survival analysis pull in opposite directions.** Correcting for noise in the baseline times (SIMEX) pulls the newest agent's 50% horizon down; treating failed human baselines as censored instead of discarding them pulls it up. On the v1.0 suite the net effect is up by about 10–18%; on v1.1 it is down by about 21–25%. Together they leave the headline horizon uncertain by roughly ±20% — a range of scenario results across suites and censoring conventions, not a fitted uncertainty interval.
- **The researcher-estimated tasks deserve a second look.** These are the tasks where every human baseliner failed. The export is upfront that these are estimates rather than baselines, and for several of those tasks the recorded durations of the failed attempts already exceed the published estimate, which suggests the estimates are low relative to METR's own data.
- **One open risk could still move the doubling time:** if the people who took the long tasks were systematically faster than those who took the short tasks, the x-axis is *tilted* rather than shifted, and a tilt does change the slope. Testing that needs one field METR has not released — see [Stage 4 - Cohort selection](stage-4-cohort-selection/).

Per-stage numbers and figures are in the stage READMEs linked above.

## Relation to prior work

METR's own [modelling-assumptions note](https://metr.org/notes/2026-03-20-impact-of-modelling-assumptions-on-time-horizon-results/) applied a SIMEX noise correction with a *global* noise assumption, and their [limitations note](https://metr.org/notes/2026-01-22-time-horizon-limitations/) explicitly lists failed-baseline survival analysis and baseliner selection as open. Stage 2 refines the SIMEX correction with per-task empirical noise estimates; Stage 3 does the failed-baseline survival analysis for the first time. The two corrections turn out to be of comparable size and opposite sign, so together they leave the headline horizon uncertain across scenarios by roughly ±20% rather than confidently revised downward.

## Reproducing

Everything is a plain Python script — no notebook interaction needed. The only ordering constraints: Stage 1 writes the per-run CSVs that Stage 2 consumes, and Stage 2's σ analysis writes the per-task σ that the rest of Stage 2 and Stage 3 consume.

```bash
uv sync --all-extras                     # from repo root (Python ≥3.10); the `reanalysis` extra pulls in plotly, kaleido, joblib, jupytext, nbconvert
cd reanalysis/stage-1-export-archaeology && python analysis.py
cd ../stage-2-measurement-error
python sigma_task.py                     # per-task σ — must run first
python xboot.py                          # nonparametric floor + parametric bootstrap
python coherent_shift.py
python simex.py
cd ../stage-3-survival && python analysis.py
```

- **Extending to v1.1:** rerun any Stage-2/3 script with `DATASET_VERSION=1-1` (outputs get a `_v1_1` suffix; Stage 1 handles both suites in one run).
- **Caches:** the committed `data/` CSVs double as caches for the expensive steps (the bootstrap replicates in `xboot`, the SIMEX curves) — delete a cache file to force a recompute (for the SIMEX curves, its `_meta.json` sidecar too); seeds are fixed, so a recompute reproduces it exactly. `N_BOOT` and `N_SIMEX` env vars trade runtime for precision; a cache that does not match the requested count is recomputed (`xboot`) or refused (`simex`) rather than silently reused.

Each stage directory contains: `README.md` (summary + key figures), `analysis.py` (jupytext py:percent source of truth), `analysis.ipynb` (paired, executed), `figures/`, and `data/` (derived datasets).

*By [Ofir Reich](https://github.com/ofir-reich), with Claude as research assistant.*
