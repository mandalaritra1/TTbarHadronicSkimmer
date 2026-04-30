# Repo Restructure — Progress Tracker

This file is the live source of truth for the reorganization. Each step has a checkbox; Claude updates this file as work proceeds so a fresh session (or a different model) can pick up where the last one stopped.

**Branch:** `reorg/cleanup` (off `coffea-2025`)
**Started:** 2026-04-30

---

## Context

The repo has accumulated loose notebooks and scripts at root, mixed code/notebooks inside `plots/`, near-empty `scripts/`, and a 1348-line `ttbaranalysis.md` that has drifted ahead of the CLI script `ttbaranalysis.py`. The notebook contains ~500 lines of non-analysis code (widgets, ntuple post-processing, Dask cluster setup, QCD combiner) that should live in `python/` modules so the notebook can be a thin launcher.

Goal: clean directory layout, minimal `ttbaranalysis.ipynb`, and a `ttbaranalysis.py` that no longer lags the notebook — without breaking the `coffea-2025` workflow.

User decisions:
- Keep `python/` named `python/` (no rename).
- Keep `kill_worker.ipynb`, `view_files.ipynb`, `dataset discovery.ipynb` — fix relative paths when moved.
- Delete `overlap/` (stale 2018 CSVs).
- Leave `coffea_env/`, `apps/` alone.
- Approach for `ttbaranalysis.md`: first port what's missing into `ttbaranalysis.py`, then extract shared logic into `python/` so both consume the same modules.

---

## Phase 0 — Branch + safety

- [x] `git checkout coffea-2025 && git pull` (only if remote is reachable)
- [x] `git checkout -b reorg/cleanup`
- [x] One commit per phase so any phase can be reverted independently

> Execution order chosen: 0 → 3 → 4 → 5 → 1 → 2 (mechanical moves first, then notebook surgery).

---

## Phase 1 — CLI parity + module extraction

Goal: `ttbaranalysis.py --iov ... --dataset ...` produces the same outputs as the notebook's run cell. Then both `.py` and `.ipynb` import the same shared functions from `python/`.

New modules to create:
- [ ] `python/run_analysis.py` — `run_analysis(args)` extracted from notebook (main loop, error recovery, ntuple post-proc, Dask teardown). Used by both `ttbaranalysis.py` and the notebook.
- [ ] `python/interactive_config.py` — `DEFAULTS`, `WIDGETS`, `build_ui()`, `build_args()`, `save_config()`, `reset_to_defaults()`. Notebook-only consumer.
- [ ] `python/ntuple_utils.py` — `_ntuple_paths_for_coffea`, `_merge_ntuple_chunks`, xrdfs helpers (from .md lines ~502-660).
- [ ] `python/dask_resources.py` — `_start_dask_resources(args)` + teardown (from .md lines ~662-806).

Updates:
- [ ] Slim `ttbaranalysis.py` to argparse + `run_analysis(args)` (~50 lines).
- [ ] Verify `python ttbaranalysis.py --iov 2024 --dataset TTbar --test` runs.
- [ ] Verify `python ttbaranalysis.py --iov 2024 --dataset TTbar --test --ntuple` runs.
- [ ] Commit: "phase 1: extract shared modules, bring CLI to parity with notebook"

---

## Phase 2 — Slim ttbaranalysis.ipynb

Target structure (~150-200 lines):
1. Imports
2. `ui = build_ui()` (widget render)
3. `args = build_args(ui)`
4. `run_analysis(args)`
5. QCD combiner — replace inline block (.md lines 1228-1248) with `combine_coffea_outputs(...)` call (already exists in `python/combine_coffea.py`).
6. Quick-look plot cells (kept).

- [ ] Replace widget block in notebook with `from python.interactive_config import build_ui, build_args; ui = build_ui()`.
- [ ] Replace ntuple/dask/run-loop blocks with `run_analysis(args)`.
- [ ] Replace QCD combiner cell with `combine_coffea_outputs(...)` call.
- [ ] `jupytext --sync ttbaranalysis.ipynb` (re-pair `.md`).
- [ ] Notebook smoke test: render widgets → `build_args()` → `run_analysis(args)` in `--test` mode produces same `.coffea` as CLI.
- [ ] Commit: "phase 2: slim ttbaranalysis.ipynb"

---

## Phase 3 — Move notebooks

```
notebooks/
├── analysis/
│   ├── dataset_discovery.ipynb       # was "dataset discovery.ipynb" (drop space)
│   └── kill_worker.ipynb
├── ntuple/
│   ├── view_files.ipynb (+.md)
│   └── ntuple_plots.ipynb (+.md)     # from plots/
├── plots/
│   ├── mc_viewer.ipynb (+.md)
│   ├── syst_viewer.ipynb
│   ├── data_viewer.ipynb
│   ├── viewPlots.ipynb
│   ├── makeTables.ipynb
│   └── scaleCoffeaFiles.ipynb
└── twodalphabet/
    ├── make2dalphabet.ipynb
    └── twodalphabet_py_demo.ipynb (+.md)
```

- [x] `mkdir -p notebooks/{analysis,ntuple,plots,twodalphabet}`
- [x] `git mv "dataset discovery.ipynb" notebooks/analysis/dataset_discovery.ipynb`
- [x] `git mv kill_worker.ipynb notebooks/analysis/`
- [x] `git mv view_files.ipynb view_files.md notebooks/ntuple/`
- [x] `git mv plots/ntuple_plots.ipynb plots/ntuple_plots.md notebooks/ntuple/`
- [x] `git mv plots/mc_viewer.ipynb plots/mc_viewer.md plots/syst_viewer.ipynb plots/data_viewer.ipynb plots/viewPlots.ipynb plots/makeTables.ipynb plots/scaleCoffeaFiles.ipynb notebooks/plots/`
- [x] `git mv notebooks/make2dalphabet.ipynb notebooks/twodalphabet_py_demo.ipynb notebooks/twodalphabet_py_demo.md notebooks/twodalphabet/`
- [x] Fix paths in `notebooks/analysis/dataset_discovery.ipynb`: `data/nanoAOD/QCD_ptbinned.json` → `../../data/nanoAOD/QCD_ptbinned.json`
- [x] Fix paths in `notebooks/ntuple/view_files.ipynb`: `data/nanoAOD/TTbar.json` → `../../data/nanoAOD/TTbar.json`
- [x] Fix paths in moved `plots/`-origin notebooks: `sys.path.append('../python/')` → `'../../python/'` in makeTables/scaleCoffeaFiles/viewPlots; `outputs/plots` → `../../outputs/plots` in ntuple_plots; auto-discovery of repo root in twodalphabet_py_demo.
- [ ] Smoke test: deferred until Phase 4 lands (notebooks importing `from plots.X` will keep working until then because plots/*.py files still exist).
- [x] Commit: "phase 3: move notebooks into notebooks/ subdirs"

> Note: `from plots import hep_plot` in `mc_viewer/syst_viewer/data_viewer` notebooks still works because `plots/hep_plot.py` is still in place. Phase 4 will move it to `python/` and rewrite imports.

---

## Phase 4 — Move scripts and lib modules

- [x] `git mv cutflow_p3.py run_evt_lumi.py write_ntuple.py scripts/`
- [x] `git mv shell scripts/shell`
- [x] `git mv python/get_missing_files_script.py scripts/`
- [x] `git mv plots/hep_plot.py plots/plotting.py plots/data_viewer.py plots/scaleCoffeaFiles.py plots/make2Drootfiles.py python/`
- [x] Rewrote `from plots.X` and `from plots import X` → `from python.X` / `from python import X` in 3 notebooks + python/data_viewer.py
- [x] Adjusted sys.path entries in ntuple_plots.ipynb (`../../plots` → `../../python`) and twodalphabet_py_demo.ipynb (removed redundant `plots` path)
- [x] Updated `README.md` paths: syst_viewer, ntuple_plots, write_ntuple.py, singularity launcher (`./scripts/shell`).
- [x] Import smoke: `PYTHONPATH=python python3 -c "import hep_plot; import plotting; import functions; print('ok')"` → ok (plotting.py uses bare `import functions`, requires python/ on path — pre-existing convention).
- [x] Removed empty `plots/` dir (after `rm -rf plots/__pycache__`).
- [x] Commit: "phase 4: move scripts to scripts/, plots/*.py to python/, update imports" (also folded in the Phase-3 path fixes that didn't land in that commit).

---

## Phase 5 — Delete cruft

- [x] `git rm -r overlap/`
- [ ] ~~`git rm -r rootfiles/`~~ — SKIPPED: working tree had 3 untracked `.root` files in `rootfiles/store/{data,mc}/`. Not safe to remove without confirmation. User can `rm -rf rootfiles/` manually if those files are also stale.
- [x] `rm -rf images/` (subdirs were empty, nothing tracked)
- [x] `rm -f out.log` (gitignored already)
- [x] Left `coffea_env/`, `.codex`, `CODEX.md` alone.
- [x] Commit: "phase 5: remove stale overlap/ CSVs and empty images/, out.log"

---

## Verification

- [ ] CLI parity (post-phase-1): `python ttbaranalysis.py --iov 2024 --dataset TTbar --test` produces a `.coffea` in `outputs/dy/`.
- [ ] CLI ntuple parity: same with `--ntuple` produces ntuple-bearing `.coffea`.
- [ ] Notebook parity (post-phase-2): identical `.coffea` from notebook in `--test` mode.
- [ ] Moved-notebook smoke (post-phase-3): no `FileNotFoundError` in first cells of moved notebooks.
- [ ] Import smoke (post-phase-4): the one-liner from Phase 4 prints `ok`.
- [ ] Mergeability check pre-PR:
  ```bash
  git fetch origin coffea-2025
  git merge --no-commit --no-ff origin/coffea-2025 && git merge --abort
  ```
- [ ] Final tree check: `ls` at root shows only the four anchors (`ttbaranalysis.{ipynb,md,py}`, `ttbarprocessor.py`) plus README/config/dirs — no loose `.ipynb` or analysis `.py`.

---

## Notes / handoff log

Append a short note here whenever a phase is finished or a new session takes over, so context isn't lost.

- 2026-04-30: Plan written. Ready to start Phase 0.
- 2026-05-01: Phases 0, 3, 4, 5 done in that order (mechanical-first strategy). Branch `reorg/cleanup` is at HEAD `phase 5: remove stale overlap/ CSVs ...`. Root tree is now clean: only `ttbaranalysis.{ipynb,md,py}`, `ttbarprocessor.py`, README/AGENTS/CLAUDE/CODEX, and the standard dirs. Next session: Phase 1 (extract widgets/ntuple/dask/runner from `ttbaranalysis.md` into new `python/` modules; bring `ttbaranalysis.py` to parity), then Phase 2 (slim notebook).
