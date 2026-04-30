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

**Status: implementation complete; final commit pending in this session.**

New modules created:
- [x] `python/run_analysis.py` — `run_analysis(args)` extracted from notebook (main loop, error recovery, ntuple post-proc, Dask teardown). Imports from `python.ntuple_utils`, `python.dask_resources`, `python.functions`, `ttbarprocessor`. Used by both `ttbaranalysis.py` and the notebook.
- [x] `python/interactive_config.py` — `DEFAULTS`, `DEFAULT_SIGNALS`, `build_ui()`, `build_args(W)`, `load_config()`. Notebook-only consumer. Public API: `from python.interactive_config import build_ui, build_args`.
- [x] `python/ntuple_utils.py` — combines library functions previously in `scripts/write_ntuple.py` (`write_ntuple`, `merge_root_ntuples`, xrootd helpers) with the .md helpers (`_ntuple_paths_for_coffea`, `_merge_ntuple_chunks`, `_default_ntuple_base_dir`, `_normalize_ntuple_base_dir`, `_write_lpc_ntuple_merge_instructions`, `_write_accumulated_ntuple`).
- [x] `python/dask_resources.py` — `_start_dask_resources`, `_close_dask_resources`, `_dask_write_visibility_probe`, `_check_dask_ntuple_chunk_visibility`. Imports xrootd helpers from `python.ntuple_utils`.

Refactored:
- [x] `scripts/write_ntuple.py` rewritten as a thin CLI wrapper that imports `write_ntuple` and `merge_root_ntuples` from `python.ntuple_utils`. CLI behavior preserved verbatim.
- [x] `python/run_analysis.py` uses `chunksize=100` in `--test` mode for both futures and Dask executors, while preserving production chunksizes (`200000` futures, `100000` Dask). Local smoke tests should use `ZPrimeLocal` with `--redirector rootfiles/` because XRootD is not available locally.

Still to do:
- [x] **Slim `ttbaranalysis.py`** to argparse + `run_analysis(args)`. Added argparse coverage for:
  - `--ntupleContent` (`slim` | `full`, default `slim`)
  - `--ntupleStorage` (`chunks` | `accumulator`, default `chunks`)
  - `--ntupleBaseDir` (str, default `""`)
  - `--overwrite` (action='store_true')
  - `--daskMemory` (int, default `5`)
  - Note: `args.subsample` already exists. Confirm `args.era`, `args.pt`, `args.mass` defaults are `[]` (they already are with `action='append', default=[]`).
  - Final structure: ~60 lines: imports, `parser = argparse.ArgumentParser(...)`, `parser.add_argument(...)` calls, `args = parser.parse_args()`, `from python.run_analysis import run_analysis`, `run_analysis(args)`.
- [ ] Verify `.venv/bin/python ttbaranalysis.py --iov 2024 --dataset ZPrimeLocal --test --redirector rootfiles/ --overwrite` runs (futures executor, no Dask). Attempted after `chunksize=100`; it reaches Coffea preprocessing/merge on the local ROOT file but did not finish within 120 s in this sandbox.
- [ ] Verify `.venv/bin/python ttbaranalysis.py --iov 2024 --dataset ZPrimeLocal --test --redirector rootfiles/ --ntuple --overwrite` runs. Deferred until the non-ntuple local smoke finishes reliably.
- [x] Run import smoke: `.venv/bin/python -c "from python.run_analysis import run_analysis; from python.interactive_config import build_ui, build_args; from python.ntuple_utils import write_ntuple; from python.dask_resources import _start_dask_resources; print('ok')"` → ok, with a Matplotlib cache warning because `/home/aritra/.config/matplotlib` is not writable in this sandbox.
- [x] Commit: `1e47d9e phase 1: slim CLI runner` (follow-up to `0fbd583 phase 1 (partial): extract shared modules from notebook`)

**Pitfalls / things to watch:**
- `python/run_analysis.py` writes `out.log` and copies `ttbarprocessor.py` into `outputs/dy/logs/` — must be run with cwd at repo root, otherwise `data/nanoAOD/*.json` lookups fail.
- `python.functions.printTime` and `makeSaveDirectories` are imported — confirm they still exist in `python/functions.py`.
- The notebook's old code referenced `from write_ntuple import ...` — that import is now broken in the notebook (since write_ntuple lives at scripts/write_ntuple.py and is no longer importable as a top-level module). Phase 2 (notebook slim) will replace those imports with `from python.ntuple_utils import ...` or simply rely on `run_analysis` calling them internally.
- `scripts/write_ntuple.py` adds repo root to `sys.path` so `from python.ntuple_utils import ...` works regardless of cwd. Verify this doesn't shadow anything.
- Local smoke tests should use `ZPrimeLocal` plus `--redirector rootfiles/`; the local machine currently has the needed ROOT files but not a working XRootD/pyxrootd environment. Remote TTbar/QCD smoke tests are not useful locally until XRootD is fixed.

---

## Phase 2 — Slim ttbaranalysis.ipynb

Target structure (~150-200 lines):
1. Imports
2. `ui = build_ui()` (widget render)
3. `args = build_args(ui)`
4. `run_analysis(args)`
5. QCD combiner — replace inline block (.md lines 1228-1248) with `combine_coffea_outputs(...)` call (already exists in `python/combine_coffea.py`).
6. Quick-look plot cells (kept).

- [x] Replace widget block in notebook with `from python.interactive_config import build_ui, build_args; ui = build_ui()`.
- [x] Replace ntuple/dask/run-loop blocks with `run_analysis(args)`.
- [x] Replace QCD combiner cell with `combine_coffea_outputs(...)` call.
- [x] `jupytext --sync ttbaranalysis.ipynb` (re-pair `.md`).
- [ ] Notebook smoke test: render widgets → `build_args()` → `run_analysis(args)` in `--test` mode produces same `.coffea` as CLI. Widget/build-args smoke passes; event-processing smoke is still blocked by the same local runner timeout noted in Phase 1.
- [x] Commit: `4dc0c00 phase 2: slim ttbaranalysis notebook`

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

- [ ] CLI parity (post-phase-1): `.venv/bin/python ttbaranalysis.py --iov 2024 --dataset ZPrimeLocal --test --redirector rootfiles/ --overwrite` produces a `.coffea` in `outputs/dy/`. Attempted after the `chunksize=100` change; still timed out after 120 s in the sandbox after Coffea preprocessing/merge.
- [ ] CLI ntuple parity: same with `--ntuple` produces ntuple-bearing `.coffea` and merged ROOT ntuple. Deferred until the non-ntuple smoke finishes reliably.
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
- 2026-05-01 (later): Started Phase 1. Created `python/{run_analysis,interactive_config,ntuple_utils,dask_resources}.py` and rewrote `scripts/write_ntuple.py` as a thin CLI wrapper. **Nothing committed yet** — all 4 new files plus the `scripts/write_ntuple.py` rewrite are untracked/unstaged. The current `ttbaranalysis.py` is still the old 339-line version. **Resume: pick up at "Slim `ttbaranalysis.py`" in the Phase 1 list above.** Then test, then commit Phase 1, then start Phase 2 (slim the notebook).
  - Files on disk to verify before next steps: `git status` should show `?? python/run_analysis.py`, `?? python/interactive_config.py`, `?? python/ntuple_utils.py`, `?? python/dask_resources.py`, `M scripts/write_ntuple.py`.
  - The notebook (`ttbaranalysis.ipynb` / `.md`) is still the bloated 1348-line version. Do not edit it during Phase 1 testing — Phase 2 handles that.
- 2026-05-01 (Codex): Slimmed `ttbaranalysis.py` to a thin argparse wrapper around `python.run_analysis.run_analysis(args)`. Static compile and import smoke pass under `.venv/bin/python`. CLI and CLI+ntuple smoke commands enter the shared runner but cannot complete in this local venv because importing `XRootD.client` segfaults inside `pyxrootd` (`.venv/bin/python -c "from XRootD import client"` exits 139). This is an environment blocker independent of the wrapper refactor.
- 2026-05-01 (Codex): User clarified local tests should always use `ZPrimeLocal` with files under `rootfiles/` because XRootD is unavailable locally. Updated test-mode chunksizes to 100 entries to make `maxchunks=1` smoke tests fast. Started a pre-change `ZPrimeLocal` smoke with the old large chunksize; it was still running when this note was added.
- 2026-05-01 (Codex): Stopped the long pre-change smoke, reran `ZPrimeLocal` with `chunksize=100`, and also tried `--noSyst`; both reached Coffea preprocessing/merge quickly but did not complete within a 120 s sandbox timeout. The CLI wrapper/import path is validated; event-processing smoke remains open.
- 2026-05-01 (Codex): Committed Phase 1 follow-up as `1e47d9e`. Slimmed `ttbaranalysis.md` from 1348 to 151 lines, synced `ttbaranalysis.ipynb`, kept the standalone QCD combiner and quick-look plot cells, and dropped the unrelated dataset-discovery tail. Smoke-tested `build_ui()` + `build_args(ui)` under `.venv/bin/python`; it restores the local `ZPrimeLocal` widget config and produces the expected args namespace. Committed Phase 2 as `4dc0c00`.
