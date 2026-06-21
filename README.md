
# TTbarHadronicSkimmer

This code can be run on either LPC or coffea.casa (hosted at UNL). We highly recommend using coffea.casa because it is upto 5x faster.

## LPC setup

### Login

```bash
ssh -Y -L 8XXX:127.0.0.1:8XXX LPCUSERNAME@cmslpc-el9.fnal.gov
```


### Setup

It is recommended to work on `nobackup` area in LPC.
```bash
cd nobackup
```
Initialise the `voms-proxy`
```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

Clone coffea-2025 branch of this repository.
```bash
git clone -b coffea-2025 https://github.com/mandalaritra1/TTbarHadronicSkimmer.git
```

```bash
cd TTbarHadronicSkimmer 
```
Setup lpcjobqueue by following instruction from [here](https://github.com/CoffeaTeam/lpcjobqueue). Afterwards, the singularity container can be run with:

```bash
./shell coffeateam/coffea-dask-almalinux9:2025.12.0-py3.12  
```

### (optional) Jupyter Lab

For interactive jupyter lab environment run the following command inside the singularity container: 
```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8XXX
```

Then copy the provided link to your browser.

## coffea.casa setup

Go to [coffea.casa](https://coffea.casa) and login using your preferred method.

Select the latest coffea image for 2025 and press `Start`.
![alt text](docs/image.png)

Clone this repository:
```bash
git clone -b coffea-2025 https://github.com/mandalaritra1/TTbarHadronicSkimmer.git
```
Go to the `Dask` tab from left and click the *SHUTDOWN* button.

![alt text](docs/image-1.png)

Open [`ttbaranalysis.ipynb`](ttbaranalysis.ipynb) and run with the CASA configuration.

## Running Jobs

### Interactive notebook

Open [`ttbaranalysis.ipynb`](ttbaranalysis.ipynb) and use the configuration widgets to select datasets, IOV, taggers, systematics, and execution mode. Settings are auto-saved to `.last_config.json`.

### Command line

```bash
python ttbaranalysis.py --iov 2024 --dataset TTbar
```

Common options:

| Flag | Description |
|---|---|
| `--iov` | Year: `2022`, `2023`, `2024` |
| `--dataset` | `data`, `TTbar`, `QCD`, `ZPrime1`, `ZPrime10`, `ZPrime30`, `ZPrimeDM`, `RSGluon`, `ZPrimeLocal` |
| `--era` | Filter to specific era(s), e.g. `--era C --era D` |
| `--subsample` | Run a specific manifest subsection, e.g. `--subsample QCD_PT-1000to1500` |
| `--noSyst` | Run nominal only (no systematics) |
| `--ntuple` | Collect flat per-event ntuple in the `.coffea` output |
| `--test` | Run on 1 chunk with 1 worker |
| `--blind` | Process 1/10th of data |
| `--ttagWP` | Top-tagger working point: `loose`, `medium` (default), `tight` |
| `--toptagger` | Top-tagger: `deepak8` (default, baseline GloParTv3 `TopvsQCD`), `recomb` (learned per-pT recombination — see below), `cmsv2` (legacy) |
| `--recomb-weights` | Deploy JSON for `--toptagger recomb` (default `data/recomb/recomb_deploy_2024.json`) |
| `--ht` | HT cut: `1400` (default) or `950` |
| `--dask` | Use Dask executor instead of futures |
| `--env` | `lpc` (default), `casa`, `winterfell`, `local` |
| `-r` | XRootD redirector URL (default: `root://cmsxrootd.fnal.gov/`) |

For now the analysis can run on 2022, 2023, and 2024 datasets.

Example for a single QCD pT bin:

```bash
python ttbaranalysis.py --iov 2024 --dataset QCD --subsample QCD_PT-1000to1500
```

## Deriving 2024 Top-Tag Working Points

The standalone top-tag WP workflow derives pT-binned GloParTv3 `TopvsQCD`
thresholds from QCD mis-tag targets, then reports the matched-TTbar signal
efficiency at those thresholds. It writes a histogram accumulator first, then a
JSON/plot summary.

For a local test, cap the input to one file per dataset:

```bash
python run_toptag_wp.py \
  --env local --test --maxfiles 1 \
  --rootdir /Users/aritra/Projects/rootfiles/ttbar \
  --out outputs/toptag_wp_2024_local_test.coffea
```

On LPC, the runner uses the full 2024 `data/nanoAOD/QCD.json` and
`data/nanoAOD/TTbar.json` manifests through `root://cmsxrootd.fnal.gov/` and
starts an `LPCCondorCluster`. QCD generated-pT bins below `QCD_PT-300to470` are
excluded because their huge cross sections can populate the selected AK8
`pT > 400` tail with very large normalized weights:

```bash
python run_toptag_wp.py \
  --env lpc \
  --out outputs/toptag_wp_2024_full.coffea
```

On coffea.casa, use:

```bash
python run_toptag_wp.py \
  --env casa \
  --out outputs/toptag_wp_2024_full.coffea
```

Run data separately when you want the discriminator Data/MC check:

```bash
python run_toptag_wp.py \
  --env lpc --sample Data \
  --out outputs/toptag_score_data_2024.coffea
```

After the Coffea output is written, derive the JSON thresholds and validation
plots. Add `--data-infile` to produce `data_mc_score_distributions.png`, where
the stacked QCD+TTbar MC is scaled to the data integral in each pT panel for a
shape comparison. This Data/MC plot uses the full preselected `mSD` range, so
the 2D alphabet sidebands are included; the WP derivation itself still uses the
`105 < mSD < 210` top-mass window.
The plot includes a hatched MC statistical uncertainty band and a `Data/MC`
ratio panel under each pT bin:

```bash
MPLCONFIGDIR=/tmp/mplconfig python plot_toptag_wp.py \
  outputs/toptag_wp_2024_full.coffea \
  --data-infile outputs/toptag_score_data_2024.coffea \
  --iov 2024 \
  --json data/toptag/toptag_wp_2024.json \
  --plotdir plots/images/toptag_wp/2024 \
  --score-rebin 10
```

## GloParTv3 recombination top-tagger (`--toptagger recomb`)

An alternative top-tagger that replaces the hand-built baseline ratio
`TopvsQCD = (TopbWqq+TopbWq)/(TopbWqq+TopbWq+QCD)` with a **learned per-pT
recombination of the raw GloParTv3 class heads**. Run the analysis with it via:

```bash
python ttbaranalysis.py --iov 2024 --dataset TTbar --toptagger recomb --ttagWP medium
```

(or pick `recomb` in the toptagger dropdown in `ttbaranalysis.ipynb`). Output
files are tagged `_recomb`. Baseline behaviour is unchanged when `--toptagger`
is left at its default.

**What it does.** Per pT bin, the score is a logistic combination of the 9-D
engineered log-scores (the 7 heads + the hadronic-top sum + the heavy-flavour-X
sum):

```
s(jet) = sigmoid( B0(pT) + Σ_k A_k(pT) · log(head_k + ε) )
```

i.e. a learned, pT-dependent weighted-geometric combination of the heads. Versus
the baseline it keeps the top-vs-QCD core but adds weight on the X heads and an
asymmetric `TopbWqq:TopbWq` split — that extra information is the gain.

The per-pT WP thresholds are calibrated to the **standard CMS AK8 top-tagger
targeted QCD mis-tag**, flat across pT (consistent with the official
nomenclature):

| `--ttagWP` | targeted QCD mis-tag |
|---|---|
| `verytight` | 0.1% |
| `tight` | **0.5%** |
| `medium` | 1.0% (default) |
| `loose` | 2.5% |
| `veryloose` | 5.0% |

So to operate at **0.5% mis-tag, use `--ttagWP tight`**. (`verytight`/`veryloose`
exist in the deploy JSON; the CLI exposes `loose`/`medium`/`tight`.) The mis-tag
is held at the target in every pT bin, so a `recomb` run is directly comparable to
a baseline run at the matching mis-tag.

**Result (2024, full statistics).** On the search-relevant Z′ resonance tops vs
the full xs-weighted QCD, the recombined tagger beats baseline by **+3–7%
(mean +4.2%) signal efficiency at the same QCD mis-tag** across 400–3000 GeV, and
passes the mass-decorrelation gate (mis-tag flat vs jet `m_SD`). See
`glopart_recomb_SUMMARY.md` for the full study.

**The deploy artifact** `data/recomb/recomb_deploy_2024.json` holds the per-pT
logistic params + per-pT WP thresholds and is loaded by
`TTbarResProcessor(topTagger='recomb', recomb_weights=...)`. It is tracked in git
and uploaded to Dask workers automatically (it lives under `data/`, and
`glopart_recomb` under `python/`, both already in the worker upload lists). To
regenerate it after re-fitting:

```bash
python build_recomb_deploy.py        # reads the production fit + QCD ntuples
```

The end-to-end study pipeline that produced the fit is documented in
`glopart_recomb_SUMMARY.md` (`run_toptag_wp.py --recomb-ntuple` →
`run_glopart_recomb.py` → `plot_glopart_recomb.py`).

## Output

The processor saves a `.coffea` file per dataset/era to `outputs/dy/`. Files are named automatically, e.g.:

```
outputs/dy/TTbar_2024_noSyst_test.coffea
outputs/dy/TTbar_2024_ntuple.coffea       # when --ntuple is set
```

## 2DAlphabet ROOT Inputs for Background Estimation

After the processor has produced the `mtt_vs_mt` histograms in `outputs/dy/`,
use [`plots/make2Drootfiles.py`](plots/make2Drootfiles.py) to convert the coffea
histograms into ROOT `TH2D` inputs for the 2DAlphabet background-estimation
workflow. The script sums matching coffea files for each sample and writes
central/forward pass/fail histograms to `outputs/twodalphabet/`.

For the current 2024 local files, where QCD inputs may not be present yet, run:

```bash
python plots/make2Drootfiles.py \
  --year 2024 \
  --coffea-dir outputs/dy \
  --out-dir outputs/twodalphabet \
  --data-pattern "data_2024_*.coffea" \
  --ttbar-pattern "TTbar_2024*.coffea" \
  --signal-points 900 1000 4000 4000:10 4000:30 \
  --skip-qcd
```

This writes files like:

```text
outputs/twodalphabet/TTbarAllHad24_Data.root
outputs/twodalphabet/TTbarAllHad24_TTbar.root
outputs/twodalphabet/TTbarAllHad24_signalZPrime900.root
outputs/twodalphabet/TTbarAllHad24_signalZPrime4000.root
outputs/twodalphabet/TTbarAllHad24_signalZPrime4000_10.root
outputs/twodalphabet/TTbarAllHad24_signalZPrime4000_30.root
```

For ZPrime signals, `--signal-points MASS:WIDTH` preserves the coffea width
field in the input glob: `ZPrime<MASS>_<WIDTH>_2024*.coffea`. A bare `MASS`
defaults to 1% width. The 1% width uses output label `signalZPrime<MASS>`,
while non-1% widths use `signalZPrime<MASS>_<WIDTH>`.

Each ROOT file contains the nominal 2DAlphabet region histograms:

```text
MttvsMtCen24Pass
MttvsMtCen24Fail
MttvsMtFwd24Pass
MttvsMtFwd24Fail
```

If QCD pT-bin coffea files are available, omit `--skip-qcd` and set
`--qcd-pattern` if the filenames differ from the default
`QCD_<year>*_PT-*to*_noSyst.coffea`. To write systematic-shift histograms in
addition to nominal, add `--include-systs`.

Check the output layout with:

```bash
rootls -t outputs/twodalphabet/TTbarAllHad24_signalZPrime4000.root
```

To make the six-panel background-estimate Data/MC projections from those ROOT
inputs, run:

```bash
MPLCONFIGDIR=/tmp/mplconfig-ttbar .venv/bin/python plots/make_bgest_datamc_plots.py \
  --year 2024 \
  --region Fail \
  --qcd-coffea-pattern "QCD*.coffea"
```

The script projects `MttvsMt*Cen/Fwd*Fail` into the low-mass sideband
`25 < m_j < 105 GeV`, signal region `105 < m_j < 210 GeV`, and high-mass
sideband `210 < m_j < 475 GeV`. Outputs are written under
`outputs/plots/bgest_datamc/<year>/<region>/`. QCD is read from coffea files
with the `mtt_vs_mt` histogram and is normalized separately in each projected
panel to `Data - TTbar`. The 2024 luminosity label defaults to `109.95 fb^-1`.
If no QCD coffea or ROOT input is available, the script stops; pass
`--allow-missing-qcd` only for TTbar-only debugging plots.

## Cutflow Tables

Use `make_cutflow_table.py` to turn a processor `.coffea` output into a
Markdown or LaTeX cutflow table. For a plain event-count table, run:

```bash
env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/mplconfig \
  python make_cutflow_table.py \
  outputs/dy/TTbar_2024_inclusive.coffea \
  --format markdown \
  --weight-mode none \
  --title TTbar_2024_inclusive \
  -o cutflows/TTbar_2024_inclusive_cutflow.md
```

`--weight-mode none` prints only event counts and efficiencies. Use
`--weight-mode raw` to add the raw generator/LHE `sumw`, or
`--weight-mode scaled` to add the luminosity-scaled yield when normalization
metadata is available.

To write both Markdown and LaTeX versions:

```bash
env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/mplconfig \
  python make_cutflow_table.py \
  outputs/dy/TTbar_2024_inclusive.coffea \
  --format both \
  --weight-mode none \
  -o cutflows/TTbar_2024_inclusive_cutflow
```

This writes `cutflows/TTbar_2024_inclusive_cutflow.md` and
`cutflows/TTbar_2024_inclusive_cutflow.tex`. Compact category names are expanded
for readability, e.g. `atcen` becomes `Antitag, central rapidity`.

## Flat ntuple (ROOT TTree)

When `--ntuple` is set (or the **Ntuple** checkbox is ticked in the notebook), a flat per-event ntuple is embedded in the `.coffea` output. Convert it to a ROOT TTree with:

```bash
python write_ntuple.py outputs/dy/TTbar_2024_ntuple.coffea TTbar_2024.root ttbar
```

No ROOT installation is required — `uproot` handles the file writing.

Branches stored: `jet0/1_pt`, `jet0/1_eta`, `jet0/1_phi`, `jet0/1_msd`, `jet0/1_tdisc`, `jet0/1_rapidity`, `ttbarmass`, `ht`, `dy` (Δy), `chi` (χ_dijet = exp|Δy|), `weight`, `anacat`, `run`, `lumi`, `event`.

## Viewing Histograms

To view basic histograms and systematic variations after running, use [`plots/syst_viewer.ipynb`](plots/syst_viewer.ipynb).

To plot distributions from the flat ntuple, use [`plots/ntuple_plots.ipynb`](plots/ntuple_plots.ipynb).
