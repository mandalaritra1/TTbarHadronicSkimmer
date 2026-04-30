
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
./scripts/shell coffeateam/coffea-dask-almalinux9:2025.12.0-py3.12  
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
| `--ht` | HT cut: `1400` (default) or `950` |
| `--dask` | Use Dask executor instead of futures |
| `--env` | `lpc` (default), `casa`, `winterfell`, `local` |
| `-r` | XRootD redirector URL (default: `root://cmsxrootd.fnal.gov/`) |

For now the analysis can run on 2022, 2023, and 2024 datasets.

Example for a single QCD pT bin:

```bash
python ttbaranalysis.py --iov 2024 --dataset QCD --subsample QCD_PT-1000to1500
```

## Output

The processor saves a `.coffea` file per dataset/era to `outputs/dy/`. Files are named automatically, e.g.:

```
outputs/dy/TTbar_2024_noSyst_test.coffea
outputs/dy/TTbar_2024_ntuple.coffea       # when --ntuple is set
```

## Flat ntuple (ROOT TTree)

When `--ntuple` is set (or the **Ntuple** checkbox is ticked in the notebook), a flat per-event ntuple is embedded in the `.coffea` output. Convert it to a ROOT TTree with:

```bash
python scripts/write_ntuple.py outputs/dy/TTbar_2024_ntuple.coffea TTbar_2024.root ttbar
```

No ROOT installation is required — `uproot` handles the file writing.

Branches stored: `jet0/1_pt`, `jet0/1_eta`, `jet0/1_phi`, `jet0/1_msd`, `jet0/1_tdisc`, `jet0/1_rapidity`, `ttbarmass`, `ht`, `dy` (Δy), `chi` (χ_dijet = exp|Δy|), `weight`, `anacat`, `run`, `lumi`, `event`.

## Viewing Histograms

To view basic histograms and systematic variations after running, use [`notebooks/plots/syst_viewer.ipynb`](notebooks/plots/syst_viewer.ipynb).

To plot distributions from the flat ntuple, use [`notebooks/ntuple/ntuple_plots.ipynb`](notebooks/ntuple/ntuple_plots.ipynb).
