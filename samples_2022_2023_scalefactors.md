# 2022 / 2023 NanoAODv15 samples + scale-factor status

Added by request: extend the analysis filesets to additional years, then wire the
new sub-eras through the processor.

**Status:** filesets added **and the code is wired** to run 2022/2023 nominally
with GloParTv3 (validated end-to-end on one TTbar + one data file — see
"Code wiring" below). JES/JER systematics still need JEC text files vendored.

## Code wiring (done 2026-06-21)

Run `ttbaranalysis.py --iov {2022preEE|2022postEE|2023preBPix|2023postBPix}`
(add `--noSyst` until JEC is vendored). Changes:
- `ttbarprocessor.py`: `_LUMI_PB`, `_TAGGER_WPS`, `triggernames` get the 4 sub-era
  keys; new `_V15_IOVS` set + `_base_year()`; `_tscore` uses the GloParTv3
  composite for all v15 IOVs (was 2024-only).
- `python/corrections.py`: `getLumiMask`/`getMETFilter` map sub-era→base-year
  golden JSON / flag list; `GetPUSF` wired to the vendored
  `puWeights/{2022_Summer22,2022_Summer22EE,2023_Summer23,2023_Summer23BPix}`;
  `GetJECUncertainties` raises a clear "vendor JEC / use --noSyst" message for
  these IOVs (only reached by JES/JER systematics).
- `python/weights.py`: top-tag SF for the sub-eras = 2024 placeholder.
- `python/functions.py`: lumi table extended.
- `ttbaranalysis.py`: `--iov` choices extended; **signal placeholder** — Z′/RSGluon
  fall back to the 2024 v15 signal files (`source_iov`) while keeping the requested
  IOV in metadata, so per-year lumi normalization is correct.

**JEC/JER now vendored + wired (full JES/JER systematics run).** Per the official
CMS JERC recommendations (cms-jerc.web.cern.ch/Recommendations): JES = `_V4_MC`,
JER = `_JRV1_MC`. (The latest JER is JRV2, but its SF text files use a new
pt-dependent single-value format coffea's txt machinery can't parse — needs
central/down/up — so JRV1 is used, same as the repo's 2024 setup; JRV2 would
require moving JER to correctionlib.) Files vendored from the cms-jet JEC/JR
databases under `data/corrections/{JEC,JER}/`. JER tag dirs use `-` instead of the
campaign-internal `_` (e.g. `Summer22-22Sep2023_JRV1_MC`) because coffea's
`JetResolutionScaleFactor` requires the corrector name to be exactly 5
underscore-parts.

Validation (IterativeExecutor, 1 chunk each):
- `--noSyst`: TTbar 2023preBPix → 21k evts, 2-tag categories filled, `ttbarmass`
  lumi×xsec/sumw normalized; data 2022preEE/C → lumimask+METfilter+trigger applied.
- **JES/JER systematics** (`systematics=['nominal','jes','jer']`): TTbar 2023preBPix
  runs end-to-end; output carries a `systematic` axis (JEC V4 loaded, JECStack built,
  CorrectedJetsFactory applied).

Remaining: refine per-year lumi with brilcalc; derive real per-year top-tag WPs/SFs;
(optional) move JER to correctionlib to adopt the newer JRV2 SFs; jet veto maps
(mandatory Run 3) are not yet applied for any year. JERC veto-map tags: Summer22
`Summer22_23Sep2023`, Summer22EE `Summer22EE_23Sep2023`, Summer23 `Summer23Prompt23`,
Summer23BPix `Summer23BPixPrompt23` (all V1).

## Branch `jer-correctionlib-jrv2`: JEC/JER via correctionlib (JSON-POG) — recommended V4/JRV2

On this branch **all Run-3 MC** (the 2022/2023 sub-eras **and 2024**) read JEC+JER
from **correctionlib JSON-POG** files (coffea 2026 `CorrectionLibJECStack` +
`CorrectedJetsFactory`, the modern path) instead of txt; only Run-2 stays on txt.
`GetJECUncertainties` dispatches these IOVs to `_GetJECUncertainties_jsonpog`,
which **auto-discovers** the MC JEC/JER tags from the file, so a json.gz swap needs
no code change. 2024 thereby upgrades from the legacy txt **JEC V2** to the
recommended **V3**, and from a 2023BPix-JER *proxy* to the real **Summer24 JRV1**.

**Versions (JERC-recommended, + jet veto maps), per era:** 2022/2023 = `JEC V4` +
`JER JRV2`; 2024 = `JEC V3` + `JER JRV1` (Summer24). Obtained from the CERN GitLab
`cms-analysis-corrections/JME/Run3-*` repos via **SSH clone** (HTTPS/raw is
auth-gated; cvmfs `jsonpog-integration` only has older versions). Repos:
`Run3-22CDSep23-Summer22`, `-22EFGSep23-Summer22EE`, `-23CSep23-Summer23`,
`-23DSep23-Summer23BPix` (`-NanoAODv12`), `-24CDEReprocessingFGHIPrompt-Summer24`
(`-NanoAODv15`). Vendored under
`data/corrections/jsonpog/JME/<era>/{jet_jerc,fatJet_jerc,jetvetomaps}.json.gz`.

JRV2 reorganised the JER SF into a **split** layout — nominal `ScaleFactor` +
symmetric `SFUncertainty` in separate corrections (no `systematic` input), which
coffea's stock `CorrectionLibJERSF` can't combine. A small `_SplitJERSF` adapter
(in `corrections.py`) builds `[nom, nom+unc, nom-unc]`; `_build_jersf` picks it for
JRV2 and the stock adapter for JRV1, so the code handles both layouts.

Validated end-to-end: TTbar 2023preBPix with `jes`/`jer` →
`nominal/jesUp/jesDown/jerUp/jerDown` all produced and shifting the yield. The
legacy txt JEC/JER vendoring for the v15 sub-eras was removed (superseded by JSON).

To refresh versions later: re-clone the `cms-analysis-corrections/JME/*` repos
(SSH) and copy the json.gz in — no code change.

---

(original survey notes below)

## TL;DR

- **There is no 2025 (or 2026) analysis MC in NanoAODv15.** DAS only has the
  special `Run3Winter25NanoAOD` / `Run3Winter26NanoAODv15` pile-up–study flavours
  of the **flat QCD** sample (EpsilonPU / FlatPU / NoPU), and no `TTto4Q`,
  dijet-QCD, or signal MC for 2025. Collision-data NanoAODv15 for 2025 is also not
  in the standard re-reco yet. → Per instruction, fell back to **2022 + 2023**.
- 2022 and 2023 each split into **two sub-eras** with different detector
  conditions / corrections, so they are stored under four lossless year keys:
  `2022preEE`, `2022postEE`, `2023preBPix`, `2023postBPix`
  (standard CMS Run-3 IOV convention; can be merged to flat `2022`/`2023` later,
  but a merge cannot be undone without re-querying DAS).
- **TTbar, QCD (dijet + flat), and JetMET data** were found and added in v15.
  **Z′/RSGluon signal MC does NOT exist in NanoAODv15 for 2022/2023** (only the
  old NanoAODv12 existed for 2023) — see below.

## What was added to `data/nanoAOD/`

File counts (LFNs, `/store/...`, no redirector — same format as the existing
2024 entries; the runner prepends the redirector):

| year key       | TTbar | QCD bins | QCD files | QCD flat | data files (eras) |
|----------------|------:|---------:|----------:|---------:|-------------------|
| `2022preEE`    |    70 |       14 |       462 |       45 | 308  (C, D)       |
| `2022postEE`   |   226 |       14 |       412 |       45 | 804  (E, F, G)    |
| `2023preBPix`  |   137 |       14 |       293 |       38 | 517  (B, C)       |
| `2023postBPix` |    73 |       14 |       157 |       42 | 193  (D)          |

- **`TTbar.json`** — `TTto4Q_TuneCP5_13p6TeV_powheg-pythia8`, subsample
  `inclusive`. Campaigns: `Run3Summer22NanoAODv15`, `Run3Summer22EENanoAODv15`,
  `Run3Summer23NanoAODv15`, `Run3Summer23BPixNanoAODv15` (plain
  `150X_mcRun3_*` reco; JME/BTV/FastSim variants excluded).
- **`QCD.json`** — dijet `QCD_PT-*to*_TuneCP5_13p6TeV_pythia8` (plain;
  MuEnriched/EMEnriched/bcToE excluded). **NB: 2022/23 use different pT-hat bin
  edges than 2024.** Bins: `15to30, 30to50, 50to80, 80to120, 120to170, 170to300,
  300to470, 470to600, 600to800, 800to1000, 1000to1400, 1400to1800, 1800to2400,
  2400to3200` (vs 2024's `15to20 … 1000to1500, 1500to2000, 2000to2500,
  2500to3000`). Most 2022 bins are the `_ext1` productions.
- **`QCD_flat.json`** — `QCD_PT-15to7000_TuneCP5_Flat2022_13p6TeV_pythia8`,
  subsample `QCD_flat`.
- **`data.json`** — `JetMET` PD, `…-NanoAODv15…` re-reco.
  - 2022: single `/JetMET/` PD, `Run2022{C,D,E,F,G}-NanoAODv15-v1`.
  - 2023: `/JetMET0/` **+** `/JetMET1/` merged per era; era `C` merges the four
    Cv1–Cv4 versions, era `D` merges Dv1/Dv2.
  - Era→sub-era mapping: preEE = C,D · postEE = E,F,G · preBPix = B,C ·
    postBPix = D.
- **`xsec_pb.json`** — mirrored the new years for reference only (this file is
  **not read by any code**; the processor takes `xsec_pb` from each sample's
  `metadata` block).

### How they were generated (reproducible)

`dasgoclient` on `cmslpc303` (proxy `~/x509up_u25128`), e.g.
`dasgoclient -query="file dataset=/TTto4Q…/Run3Summer22NanoAODv15-…/NANOAODSIM"`.
The driver script is at `~/gen_filelists.sh` on cmslpc303.

---

## Cross sections (normalization)

Stored in each sample's `metadata.xsec_pb`. All processes are 13.6 TeV, so the
generator cross section is the **same across 2022/2023/2024** for a given process.

- **QCD dijet** — for the 9 pT-hat bins shared with 2024 (`30to50 … 800to1000`),
  the validated repo-2024 values were reused verbatim. The 5 bins unique to
  2022/23 were taken from **XSDB** (13.6 TeV):
  `15to30 = 1.301e9`, `1000to1400 = 8.92`, `1400to1800 = 0.8103`,
  `1800to2400 = 0.1148`, `2400to3200 = 0.007542` pb.
  Consistency check: XSDB `15to30 (1.301e9) ≈ 2024 15to20+20to30
  (8.857e8+4.157e8)`, and XSDB vs repo-2024 agree to ≤2 % (within LO scale unc.)
  on the shared bins.
- **QCD flat** — `1.417e9` pb (XSDB value, identical to repo-2024).
- **TTbar** — set to **350.6 pb** to match the existing `TTbar.json` 2024 entry
  (cross-year consistency).
  > ⚠️ **Open normalization issue to reconcile (pre-existing, not introduced
  > here):** XSDB lists `TTto4Q_TuneCP5_13p6TeV_powheg-pythia8` at **762.1 pb
  > (NLO)**, and `view_files.ipynb` shows the TTbar metadata as **762.1** — yet
  > `TTbar.json` (all years) carries **350.6**. The physics expectation is
  > σ(tt, NNLO 13.6 TeV) ≈ 923.6 pb × BR(4q) ≈ 420 pb. These three numbers
  > disagree; **decide the correct TTbar xsec once and apply it to all years**
  > (2024 included). I left 350.6 so the new years behave exactly like the
  > current 2024 setup.

---

## Scale factors — needed / available / not

Legend: ✅ available · 🟡 input available, **code wiring missing** · ❌ not available.

`jsonpog-integration` on cvmfs
(`/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/…`) has **full
2022/2023 coverage** (`2022_Summer22`, `2022_Summer22EE`, `2023_Summer23`,
`2023_Summer23BPix`) for JME, BTV, LUM, MUO.

| Correction | 2022/2023 status | Notes |
|---|---|---|
| **Golden JSON / lumi mask** | ✅ | `data/corrections/goldenJsons/Cert_Collisions{2022,2023}_*_Golden.json` present; `getLumiMask` handles `"2022"`/`"2023"`. |
| **MET filters** | ✅ | `getMETFilter` has 2022/2023 flag lists. |
| **Pileup weights** | 🟡 | JSONs already vendored: `data/corrections/puWeights/{2022_Summer22,2022_Summer22EE,2023_Summer23,2023_Summer23BPix}` (also in jsonpog LUM). But `GetPUSF` only branches on UL and `"2024"` — **needs 2022/23 wiring** (and the era→file map). |
| **JEC (AK4+AK8)** | ✅ | **Vendored + wired** to the JERC-recommended `_V4_MC` tags (`Summer22_22Sep2023_V4_MC`, `Summer22EE_22Sep2023_V4_MC`, `Summer23Prompt23_V4_MC`, `Summer23BPixPrompt23_V4_MC`) under `data/corrections/JEC/`; `GetJECUncertainties` handles the 4 sub-eras; JES systematics validated. |
| **JER (AK4+AK8)** | ✅ | **Vendored + wired** as `_JRV1_MC` (JRV2 SF txt not coffea-parseable) under `data/corrections/JER/` (dirs use `-` for the campaign `_`); JER systematics validated. |
| **Jet veto maps** | ✅ (input) | jsonpog JME `jetvetomaps.json.gz` per sub-era. Not currently used by the processor at all. |
| **b / subjet-tag** | 🟡 | jsonpog BTV `btagging.json.gz` per sub-era. The processor currently builds subjet-btag *efficiencies* on the fly (`GetFlavorEfficiency`, `btagCSVV2`); no external b-tag SF is applied for any year yet. |
| **Top-tag (GloParT) SF** | ❌ | No central SF. 2024 uses an explicit **placeholder flat 0.90** in `weights.py` (not measured). 2022/2023 would need the same in-house derivation as the 2024 WP work — none exists. Also verify the v15 2022/23 samples carry the `globalParT3_*` (and/or `particleNet_*`) branches the tagger expects. |
| **PDF / Q² (renorm/fact) weights** | ✅* | Read from in-file LHE weights (`GetPDFWeights`/`GetQ2weights`). Present for powheg TTbar; *QCD pythia samples generally lack LHE PDF/scale variations — confirm per sample.* |
| **ttbar pT reweighting** | ✅ | Built-in `pTReweighting`, no external input, applied to any TTbar. |
| **Trigger (HLT) SF / efficiency** | ❌ | Not present for any Run-3 year; needs measurement (AK8 PFJet / PFHT reference). |
| **Integrated luminosity** | 🟡 | `python/functions.py` has only preliminary `2023`/`2024` totals. Per-sub-era 2022/23 lumi must come from **brilcalc** with the golden JSON. |

### Minimum wiring to actually *run* 2022/2023 (when desired)
Not done here (code left untouched), listed for the follow-up:
1. Teach `GetJECUncertainties`, `GetPUSF`, `getLumiMask`, `getMETFilter` the four
   new IOV keys (or map them to `"2022"`/`"2023"` + a preEE/postEE,
   preBPix/postBPix switch).
2. Vendor the JEC/JER text files (or switch those helpers to read jsonpog
   correctionlib directly).
3. Add `2022preEE/2022postEE/2023preBPix/2023postBPix` to the lumi table and the
   top-tag-SF dict in `weights.py`.
4. Confirm the GloParT/ParticleNet top-tag branch names in v15 2022/23 NanoAOD.

---

## Not available in NanoAODv15 for 2022/2023

- **Z′→tt and RSGluon→tt signal MC** — DAS has no
  `ZPrimeToTT_*`/`RSGluonToTT_*` in any `Run3Summer22*/Run3Summer23*NanoAODv15`
  campaign (only the older **NanoAODv12** existed for 2023, already partially in
  `ZPrime1.json`). `ZPrime1/10/30.json`, `ZPrimeDM`, `RSGluon` were therefore
  **left unchanged** for these years. Re-check DAS later, or request production.
- **2025 / 2026 analysis MC** — none (see TL;DR).
