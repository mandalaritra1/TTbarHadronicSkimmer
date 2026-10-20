
# TTbarHadronicSkimmer

Coffea processor for the CMS Run 3 all-hadronic tt̄ resonance search. It reads
NanoAODv15, selects events with two boosted AK8 top candidates, applies the
corrections and systematics, and writes `.coffea` histogram files. The
`mtt_vs_mt` histograms become the 2DAlphabet inputs for the background fit in
[bgestimation](https://github.com/mandalaritra1/bgestimation).

Supported IOVs: `2022preEE`, `2022postEE`, `2023preBPix`, `2023postBPix`,
`2024`, `2025`, `2026`. The working branch is `coffea-2025`.

## Setup

### LPC

```bash
ssh -Y -L 8XXX:127.0.0.1:8XXX LPCUSERNAME@cmslpc-el9.fnal.gov
```

Work in `nobackup`, make a grid proxy and clone the repository:

```bash
cd nobackup
voms-proxy-init --rfc --voms cms -valid 192:00
git clone -b coffea-2025 https://github.com/mandalaritra1/TTbarHadronicSkimmer.git
cd TTbarHadronicSkimmer
```

Set up [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue), then start the container:

```bash
./shell coffeateam/coffea-dask-almalinux9:2026.4.0-py3.12
```

For Jupyter, run this inside the container and open the printed link:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8XXX
```

### coffea.casa

Log in at [coffea.casa](https://coffea.casa), select the latest coffea image and press `Start`.

![coffea.casa image selection](docs/image.png)

Clone the repository with the same `git clone` command as above. Then open the
`Dask` tab on the left and press *SHUTDOWN*, because the runner starts its own
cluster.

![coffea.casa Dask tab](docs/image-1.png)

## Running

### Command line

Production runs use the command line:

```bash
python ttbaranalysis.py --iov 2024 --dataset TTbar --ttagWP tight --dask --env lpc --outdir myrun
```

Add `--test` to run one file with one worker before a full run.

| Flag | Description |
|---|---|
| `--iov` | `2022preEE`, `2022postEE`, `2023preBPix`, `2023postBPix`, `2024` (default), `2025`, `2026` |
| `--dataset` | `data`, `TTbar`, `QCD`, `ZPrime1`, `ZPrime10`, `ZPrime30` (the number is the Z′ width in %). Repeat for several. |
| `--era` | Data era(s), e.g. `--era C --era D`. All eras if omitted. |
| `--subsample` | One manifest section, e.g. `--subsample QCD_PT-1000to1500` |
| `--ttagWP` | Top-tag working point: `loose`, `medium` (default), `tight`. Top-tag SFs exist for `tight` only, and the analysis uses `tight`. |
| `--noSyst` | Nominal only |
| `--weights-only` | Keep the weight systematics, skip the JES/JER/JMS/JMR passes |
| `--ntuple` | Store a flat per-event ntuple in the output (`--ntuple-columns slim` for fewer columns) |
| `--event-list` | Store run/lumi/event after the tt̄-candidate cuts (debugging) |
| `--blind` | Process 1/10 of the data |
| `--ht` | HT cut: `1400` (default) or `950` |
| `--signal-batch` | Z′ masses per output file (default 4) |
| `--dask` | Dask executor instead of futures |
| `--env` | `lpc` (default), `casa`, `winterfell`, `local` |
| `-r` | XRootD redirector (default `root://cmsxrootd.fnal.gov/`). For PromptReco data, `root://cms-xrd-global.cern.ch/` avoids a few corrupt FNAL replicas. |
| `--outdir` | Write to `outputs/<OUTDIR>/` instead of `outputs/dy/` |
| `--test` | One file, one worker |

### Notebook

[`ttbaranalysis.ipynb`](ttbaranalysis.ipynb) has widgets for the same options
and saves the last settings to `.last_config.json`. It runs through its own copy
of the runner (`ttbar_notebook.py`), so check results against the command line
before using them.

## Driving coffea.casa from your laptop or an AI agent

coffea.casa has no SSH. The `cc_*` tools in
[cms-tools](https://github.com/mandalaritra1/cms-tools/tree/main/coffea_casa)
talk to your JupyterLab server over its REST API with a JupyterHub token. That
lets a terminal on your laptop, or a coding agent running there (Claude Code,
Codex, ...), launch jobs on casa, watch their logs and fetch the outputs.

### 1. Install the tools

```bash
git clone https://github.com/mandalaritra1/cms-tools.git
cd cms-tools/coffea_casa
pip install -r requirements.txt
```

They need only `requests` and `websocket-client`. Run them from this directory,
because each script imports `cc_common.py` from next to it.

### 2. Get a token

1. Log in at [coffea.casa](https://coffea.casa) and start your server.
2. Click **Token** at the top right, or open `https://coffea.casa/hub/token`.
3. Press *Request new API token*, give it a short expiry and copy it.

Export the token and the address of your own server in the shell you will work from:

```bash
export CC_TOKEN=<the-token>
export CC_BASE=https://coffea.casa/user/<your-username>
```

`CC_BASE` is your JupyterLab address up to and including the username. Copy it
from the browser's address bar. If it is not set, the tools go to the
maintainer's server and your token is rejected.

The token can do anything you can do on casa. Keep it out of files, git and
agent prompts, and mint a new one when it expires (the tools then report
`token rejected`).

Check the connection:

```bash
python cc_shell.py "hostname && ls"
```

### 3. Run code with `cc_exec` and `cc_run`

`cc_exec.py` runs Python on a new kernel, prints the output and deletes the
kernel (`--keep` keeps it). Use it for short checks:

```bash
python cc_exec.py --code "import coffea; print(coffea.__version__)"
python cc_exec.py --file my_check.py --timeout 300
```

`cc_run.py` is for anything longer than a couple of minutes. It writes the
command to `~/cc_run/<name>.sh` on casa and starts it detached, so the job
keeps running when your laptop sleeps or the connection drops. The output goes
to `~/cc_run/<name>.log`, which ends with `__CC_RUN_EXIT=<rc>__` when the job
finishes.

```bash
python cc_run.py --name ttbar24_test --dir '~/TTbarHadronicSkimmer' \
  'python ttbaranalysis.py --iov 2024 --dataset TTbar --ttagWP tight --dask --env casa --test'

python cc_run.py --tail ttbar24_test
```

With `--dask --env casa` the runner starts its own casa Dask cluster and sizes
it per fileset from the chunk count, up to `--casa-workers` (default 64).

The other tools:

| Tool | Use |
|---|---|
| `cc_shell.py "CMD"` | Short shell command, returns output and exit code. Each call is a new shell, so chain with `&&`. |
| `cc_read.py --ls PATH`, `--cat PATH --tail N` | List a directory or read a text file. Paths are relative to the casa home. |
| `cc_get.py --remote PATH --out LOCAL` | Download a binary output (`.coffea`, `.root`). It fetches the file in pieces and checks size and md5. |

`--start-server` (all tools except `cc_read`) starts a stopped server. If it
fails with 403, the token cannot start servers, so start yours in the browser.

### 4. Hand it to an agent

Export `CC_TOKEN` and `CC_BASE`, then start the agent from that same shell so
it inherits them. Tell it where the tools are and what to do, for example:

> Use the cc_* tools in ~/cms-tools/coffea_casa. Launch a `--test` run of
> ttbaranalysis.py for 2024 TTbar on coffea.casa with cc_run, poll the log
> with `cc_run.py --tail` until `__CC_RUN_EXIT` appears, then report the exit
> code and the output file name.

Rules worth giving the agent, for example in your `AGENTS.md`, `CLAUDE.md` or a skill:

- Launch anything long with `cc_run`. When `cc_shell` times out it deletes its
  terminal, and that kills the command.
- The casa image has no `ps`, so judge whether a job is alive from its log.
- Do not kill kernels or terminals the agent did not create. They may be your
  open JupyterLab session.

## Output

Each job writes one `.coffea` file to `outputs/dy/` (or `outputs/<OUTDIR>/`).
The name holds the sample, IOV, section and top tagger, plus a suffix for
options such as `--noSyst`, `--ntuple` or `--test`:

```text
data_2024_C_topvsqcd.coffea
TTbar_2024_inclusive_topvsqcd.coffea
ZPrime10_2024_1400to2000_topvsqcd.coffea
```

A Z′ file holds up to `--signal-batch` masses of one width, one bin per mass on
the `dataset` axis, each normalised to 1 pb.

Jobs split into parts are merged with:

- `scripts/merge_split_mc.py` for MC split by file. It undoes each part's
  lumi × xsec / sumw before adding, so do not add the parts by hand.
- `scripts/merge_signal_masses.py` for per-mass Z′ jobs, into the grouped layout above.

## 2DAlphabet inputs

[`plots/make2Drootfiles.py`](plots/make2Drootfiles.py) sums the coffea files of
each sample and writes the `mtt_vs_mt` histograms as ROOT `TH2D`s:

```bash
python plots/make2Drootfiles.py \
  --year 2024 \
  --coffea-dir outputs/myrun \
  --out-dir outputs/twodalphabet_2024 \
  --data-pattern "data_2024_*.coffea" \
  --signal-points 1000 4000 4000:10 4000:30 \
  --skip-qcd
```

This writes `TTbarAllHad<YY>_Data.root`, `TTbarAllHad<YY>_TTbar.root` and one
`TTbarAllHad<YY>_signalZPrime<MASS>[_<WIDTH>].root` per signal point. `<YY>` is
the IOV without the leading `20` (`24`, `22preEE`). A bare `MASS` means 1% width.
Each file holds:

```text
MttvsMtCen<YY>Pass
MttvsMtCen<YY>Fail
MttvsMtFwd<YY>Pass
MttvsMtFwd<YY>Fail
```

The bgestimation configs read these names, so a change to the regions or
binning here needs the matching change there. `--include-systs` adds the
systematic-shift histograms. Keep per-mass and grouped Z′ files in different
directories: if both are present, the per-mass file is used.

[`plots/make_bgest_datamc_plots.py`](plots/make_bgest_datamc_plots.py) makes
Data/MC projections of these inputs in the low sideband, signal region and high
sideband of jet mass. `--lumi` only sets the label, and its default is the
2024 value (109.95 fb⁻¹), so set it for other years.

## Cutflow tables

```bash
python make_cutflow_table.py outputs/dy/TTbar_2024_inclusive_topvsqcd.coffea \
  --format both --weight-mode none -o cutflows/TTbar_2024
```

`--format` is `markdown`, `latex` or `both`. `--weight-mode none` gives counts
and efficiencies, `raw` adds the generator `sumw` and `scaled` adds the
luminosity-scaled yield.

## Flat ntuple

Convert the ntuple stored with `--ntuple` to a ROOT TTree. This needs `uproot`, not ROOT:

```bash
python write_ntuple.py outputs/dy/TTbar_2024_inclusive_topvsqcd_ntuple.coffea TTbar_2024.root ttbar
```

The column sets are defined in `ntuple_columns_for_preset` in `python/hists.py`.

## Viewing histograms

- [`plots/syst_viewer.ipynb`](plots/syst_viewer.ipynb): histograms and systematic variations
- [`plots/ntuple_plots.ipynb`](plots/ntuple_plots.ipynb): distributions from the flat ntuple

## Tests

```bash
python -m pytest tests
```
