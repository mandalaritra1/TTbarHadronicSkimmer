# Skimmer changelog: v1.1 → v1.2

Baseline **v1.1** = tag `v1.1` (`fab5038`, 2026-08-01): the production behind the
current 39-point limits. **v1.2** = the next MC + data re-skim.

Status: **Done** = committed · **Working tree** = written, not committed ·
**Planned** = decided, not started · **Open** = needs a decision.

Last updated: 2026-09-24 (round 2)

---

## Summary

| # | Change | Status | Commit | Affects | Physics effect |
|---|---|---|---|---|---|
| 1 | `ttag_pt2`/`ttag_pt3` variations actually filled | Done | `d664e04` | MC | Adds the top-tag SF uncertainty for jets with pT > 480 GeV (±15–17% on a 4 TeV Z') |
| 2 | Run-3 jet-veto map as a preselection cut | Done | `9e405c1` | Data + MC | Removes ~17–18% of high-HT signal events after MET filters; ~2% of 2025C JetMET events |
| 3 | 2025 → JME `Run3-25Prompt-Summer24` campaign | Done | `3c47f7c` | 2025 data (+ 2025 MC JER) | 2025 data JEC with 2025 residuals (AK8 pT −3.1% vs NanoAOD-embedded); 2025 jet ID works |
| 4 | 2024 JEC V3 → V5, JER JRV1 → JRV2 | Done | `a694078` | 2024 data + MC | New η-dependent residuals and JER SFs |
| 5 | GloParTv3 score histograms `jet0/1_tdisc` | Done | `28c02a9` | Output only | None (new diagnostics for CR/VR data/MC) |
| 6 | Missing HLT path raises instead of accepting all events | Done | `8a078df` | Safety | None in a correct setup |
| 7 | Antitag jet SF: measure the [medium, tight) band SF directly | Decided (fix) | — | MC | Loose SF ~0.93–1.05 now; band SF from existing SFs ~1.2–1.5 but ill-constrained |
| 8 | PDF uncertainty: std/mean → Hessian | Last (reconsider) | — | MC | PDF uncertainty ~10× larger |
| 9 | Soft-drop mass variations: JES/JER propagated, JMS 1%, JMR 2% | Done | `d3fbc3e` | MC | JES now moves ⟨mSD⟩ ±0.7%; new jms/jmr shape nuisances |
| 10 | Q2/PDF templates yield-normalized | Open | — | MC | Separates acceptance from rate |
| 11 | AK8 jet ID on the two leading jets | Open | — | Data + MC | Not applied in v1.1 or v1.2 so far |
| 12 | Real 2025 MC run instead of `scale_iov` | Open | — | 2025 MC | Uses the 2025 JER SF for forward jets |
| 13 | 2025 pileup weights (LUM 2025 file) | Open | — | 2025 MC | 2025 MC currently reweighted to the **2024** data pileup profile |
| 14a | 2025 golden JSON → `..._398903` + lumi 110.59 → **110.37** fb⁻¹ (PPD table) | Done | `d4b719c` | 2025 data + MC norm | 2025 MC −0.2%; +1056 / −859 LS in data |
| 14b | 2022/2023/2024 golden JSONs → DC re-issue after the 2026 tracker-ML review | Done | `f6ff4c1` | Data | LS: 2022 +0.07%, 2023 −0.43%, 2024 −0.04% |
| 14c | 2022/2023/2024 lumi values from the current PPD tables | Open | — | MC norm | Waiting on the PPD table numbers |
| 15 | 2025 era B data | Dropped | — | — | PPD table (reference) covers eras C–G only |
| 16 | JEC is never re-applied to data (or to `--noSyst` MC) in the analysis processor | Open | — | Data | Data keeps the NanoAOD-embedded JEC while MC gets V5 / Summer24Prompt25_V3 |

Downstream (bgestimation, after the v1.2 inputs exist): attach `ttag_pt2`/`ttag_pt3`
in the six Run-3 configs, add `jms`/`jmr` likewise, then re-fit the 39 points. Update the hard-coded lumi labels
(110.59 → 110.37, 220.54 → 220.32) in `ttbar.py`, `plot_limits*.py`, `combine_cards25/2425.sh`,
`preunblind/plot_masked_postfit_2d_projections.py`, `docs/plotting_reference.md`; they are
correct for v1.1 results, so change them only with v1.2.

---

## Done

### 1. `ttag_pt2`/`ttag_pt3` variations filled — `d664e04`

- **Problem in v1.1:** `weights.py` registers `ttag_pt1/2/3`. The systematics list
  that builds the histogram systematic axis only named `ttag_pt1`, so the pt2/pt3
  Up/Down fills went into the axis overflow bin and were lost with no warning.
  Every v1/v1.1 output and 2DAlphabet input carries only `TTAG_PT1`.
- **Nominal was always correct:** all three pT bins' SFs are in the central weight.
- **Fix:** added `ttag_pt2`, `ttag_pt3` to the CLI (`ttbaranalysis.py`), notebook
  backend (`ttbar_notebook.py`) and processor default. The fill loop now raises if
  any weight variation is missing from the axis.
- **Size (4 TeV Z', per category):** pt1 ±0.2%, pt2 ±0.4%, **pt3 +15–17% / −14–15%**.
  Expected for two tagged jets in the [600, ∞) bin: SF 0.880 ± 0.072 → (1.081)² − 1 = 16.9%.
- **Validation:** local Z' 4 TeV, with and without systematics; all six variations
  fill, overflow empty.
- **Parked questions:** (a) split each bin's total into an uncorrelated fit part plus
  correlated model/JMR/JES/JER/PU nuisances; (b) the top bin is measured mostly at
  600–800 GeV and applied to 1.5–2 TeV jets with no extrapolation uncertainty.

### 2. Run-3 jet-veto map — `9e405c1`

- **Recipe (JME):** reject the event if any AK4 PUPPI jet with pT > 15 GeV,
  TightLepVeto ID and chEmEF + neEmEF < 0.9 falls in a non-zero bin of `jetvetomap`.
- **Jet ID:** used only to choose which jets the veto checks. No analysis-level
  jet-ID cut was added (v1.1 had none either). NanoAODv15 has no `Jet_jetId`, so
  TightLepVeto is evaluated from the official `jetid.json.gz`.
- **Cutflow:** new `jetVetoMap` step after `metfilter`.
- **Size:** 2024 Z' 4 TeV W10: 17.4–17.9% of events after MET filters vetoed.
  The map covers ~2.3% of η-φ, but high-HT events have many eligible jets. An
  independent NanoAOD-level check agrees (research-notes `ttbarhadronic_run3_reviewer_audit.md`).
- **Tests:** `tests/test_jet_veto_map.py`; diagnostic plot `plots/plot_jet_veto_map_diagnostic.py`.

### 3. 2025 JME campaign switch — `3c47f7c`

- **Source:** `gitlab.cern.ch/cms-analysis-corrections/JME/Run3-25Prompt-Summer24-NanoAODv15`,
  tag 2026-07-16, vendored as `data/corrections/jsonpog/JME/2025_Summer24Prompt25/`
  (with `PROVENANCE.md`). Replaces the unused Winter25-campaign `2025_Prompt25/`.
- **Why:** JME's official campaign for 2025 PromptReco data with Summer24 MC.
- **Data JEC:** `Summer24Prompt25_V3_DATA` with 2025 run-dependent residuals.
  - In v1.1, data was produced (2026-07-23) *before* data JEC re-correction existed
    (`7b51c08`, 2026-07-31), so v1.1 data used the JEC applied in NanoAOD.
  - With the pre-switch mapping, 2025 data would have crashed (run 392293 is outside
    the 2024 residual binning).
  - **Correction (2026-09-24):** this data path is only reached by the top-tag SF
    processor. `Run3JetManager.build_corrections` returns `None` for data, so the
    analysis processor does not re-apply JEC to data at all (item 16).
  - On 2025C data (pT > 200): AK8 corrected pT / NanoAOD pT = 0.9695, AK4 = 0.9900.
- **Jet ID:** 2025 NanoAODv15 has no `Jet_jetId`; the campaign's `jetid.json.gz`
  (byte-identical to 2024's) is now used, so 2025 data runs.
- **MC:** all MC JEC and the Total JES uncertainty are identical to Summer24Prompt24_V5.
  Only the JER SF differs (`Summer24Prompt25_JRV2`), see item 12.
- **Veto map:** identical vetoed bins to the old file (169/5904); only the name changed.

---

## Done (continued)

### 4. 2024 JEC V5 / JER JRV2 — `a694078`

- `data/corrections/jsonpog/JME/2024_Summer24/{jet,fatJet}_jerc.json.gz` updated to
  the `Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15` 2026-07-16 release
  (`Summer24Prompt24_V5`, `Summer24Prompt24_JRV2`). Byte-identical to GitLab `latest`.
- v1.1 MC used V3/JRV1; v1.1 data used the NanoAOD-embedded JEC.
- V4 → V5: the V4 release shipped unchanged residuals by mistake; V5 has the new
  η-dependent L2L3Residual.

### 5. Top-score histograms — `28c02a9`

- `jet0_tdisc`, `jet1_tdisc` (GloParTv3 Top-vs-QCD, finer bins above 0.9) per
  category and systematic, plus two entries in `plots/make_leading_jet_datamc_plots.py`.
  Requested by the B2G audit for CR/VR data/MC.

### 6. Trigger safety — `8a078df`

- A missing `HLT` branch or configured path now raises; v1.1 warned and accepted
  every event.

---

## Planned / open

### 7. Antitag SF (decided: fix)
The antitag (Fail) jet is required to have medium ≤ score < tight
(`ttbarprocessor.py` antitag_disc, tight analysis → low threshold = medium 0.8571) but is
weighted with the loose-WP SF (`weights.py:101-103`).

Building the band SF from the existing per-WP SFs,
SF_band = (SF_M·ε_M − SF_T·ε_T)/(ε_M − ε_T), with ε = T&P MC efficiencies
(`toptag-sf-derivation/results/outputs/sf_2024_full.json`) and the final P1 SFs:

| pT bin | ε_M | ε_T | SF_band | coherent ± | SF_M/SF_T errors uncorrelated | loose SF (now) |
|---|---|---|---|---|---|---|
| 400–480 | 0.670 | 0.575 | 1.23 | 1.04–1.41 | ±0.86 | 1.05 ± 0.05 |
| 480–600 | 0.727 | 0.629 | 1.22 | 1.21–1.23 | ±0.80 | 0.93 ± 0.05 |
| 600+ | 0.786 | 0.696 | 1.49 | 1.41–1.57 | ±1.18 | 0.95 ± 0.09 |

The band holds only 11–14% of medium-passing true tops, so the difference of two
nearly equal products is amplified; the result depends entirely on the SF_M/SF_T error
correlation. **Plan:** measure the band SF directly in the T&P (probes in [M, T) as
their own category) with `toptag-sf-derivation`, then use it for the antitag jet.

### 8. PDF uncertainty (last; reconsider)
`GetPDFWeights` uses std/mean across replicas; the review argued the PDF set needs the
Hessian formula (~10× larger). The code came from senior CMS colleagues, so revisit the
recipe carefully at the end rather than change it now.

### 9. Soft-drop mass systematic (planned)
`msoftdrop` is taken raw from NanoAOD everywhere: `jetmsd = jet0.msoftdrop` is the
fit's jet-mass axis, and jet1 must satisfy 105 < mSD < 210 GeV. The JES/JER factory
corrects pT and the ungroomed `mass`, not `msoftdrop`, so there is no jet mass scale
(JMS) or resolution (JMR) correction and no uncertainty on either. The T&P measured
data/MC mass scale s ≈ 0.990–0.995 (profiled) and used a prescribed 2% JMR.
Effects: the mass-window acceptance for signal and tt̄ differs between data and MC
(normalization), and events migrate between the jet-mass regions (shape). 
**Done — `d3fbc3e`.** Nominal JMS = JMR = 1.000 (no nominal correction; mSD stays on the
NanoAOD value, which is built from AK4-PUPPI-corrected subjets). Variations as extra
jet passes in `Run3JetManager.build_corrections`:
- `jes`/`jer` Up/Down: mSD × (pT_var / pT_nominal), from coffea's CorrectedJetsFactory.
- `jms` Up/Down: mSD × (1 ± 0.01).
- `jmr` Up: Gaussian smearing σ = 0.02 × mSD (seeded per chunk); Down filled with
  nominal and mirrored as 2·nominal − up in `make2Drootfiles.py`.
Same conventions as the top-tag SF measurement (`p1_systematics.py`).
Smoke test (Z' 4 TeV): ⟨mSD⟩ jes ±0.7%, jms ±1%; jmr widens/narrows; overflow empty.
Downstream: add `jms`, `jmr` to the six bgestimation configs (templates
`JMSup/down`, `JMRup/down`).

### 16. No JEC on data in the analysis processor (open)
`build_corrections` returns `None` for data and for `--noSyst`, so those events keep
the JEC embedded in NanoAOD, while MC with systematics gets the re-applied payloads
(2024 V5, 2025 Summer24Prompt25_V3). CMS recommends the same JEC version on data and
MC. Options: re-apply JEC to data (nominal pass only), or keep the NanoAOD JEC on
both. Needs a decision before v1.2.

### 10. Q2/PDF normalization (open)
`q2`/`pdf` Up/Down are applied as raw event weights, so templates carry both a rate
and an acceptance change.

### 11. AK8 jet ID (open)
No jet ID on the two leading AK8 jets. Run-2 applied one; reviewers will ask.

### 12. 2025 MC production (open)
`scale_iov.py` builds 2025 MC as 2024 × (L₂₀₂₅/L₂₀₂₄). With item 3, 2025 MC has its
own JER SF: consistent within uncertainty for |η| < ~1.5, but e.g. 1.36 → 1.17 at
|η| 2.3, 1 TeV. A real 2025 MC run removes the approximation.

### 13. 2025 pileup weights (open)
`GetPUSF` uses the 2024 file (`Collisions24_CDEFGHI_goldenJSON` profile) for 2025.
LUM now ships preliminary 2025 weights for Summer24 MC:
`LUM/Run3-25Prompt-Summer24-NanoAODv15` 2026-06-05, `puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz`
(copy in the session scratchpad; not vendored). Like item 12, this also breaks the
`scale_iov` shortcut, so it favours a real 2025 MC run.

### 14. Golden JSON refresh (open)
- **2024:** the DC `Collisions24/latest` file was re-issued after a 2026 tracker-ML
  review; ours equals the `_before_TrkML2026_review` copy. The new one removes
  106 lumisections in 27 runs (no additions).
- **2025:** ours is `Cert_Collisions2025_391658_398860_Golden.json`; the newest is
  `..._398903_Golden.json` (= `golden_json_latest.json`, updated 2026-09-23):
  +1056 LS in 3 runs (395103/5/7), −859 LS in 37 runs.
- **brilcalc (normtag_BRIL, recorded), 2025 eras C–G (runs ≥ 392174):** 110.03 fb⁻¹
  with our golden, 110.37 fb⁻¹ with the newest. The code uses 110.59 fb⁻¹
  (`ttbarprocessor.py`, PPD summary table), 0.5% above the brilcalc value for the
  golden we actually apply. 2024 (109.95 fb⁻¹) not yet re-checked.

### 14b. 2022–2024 golden JSON refresh — `f6ff4c1`
DC re-issued the 2022, 2023 and 2024 golden JSONs after a 2026 tracker-ML review; ours
were the `_before_TrkML2026_review` versions. Same filenames, new content (= DC `latest`):
2022 +123/−38 LS (38 runs), 2023 +4/−394 LS (8 runs), 2024 −106 LS (27 runs).
Lumi values in `ttbarprocessor._LUMI_PB` are unchanged until the PPD tables are read
(item 14c); the 2024 LS change is −0.04%.

### 15. 2025 era B data (dropped)
DAS has `/JetMET{0,1}/Run2025B-PromptReco-v1/NANOAOD` (13.6M events each); `data.json`
starts at era C. The golden JSON certifies 24 era-B runs (391668–392046, 6691 LS),
0.254 fb⁻¹ recorded. **Dropped:** the PPD Run3-2025 table, which we follow as the
reference, lists eras C–G only.

### Luminosity reference
The PPD Run-3 tables on the PdmV Run-3 analysis TWiki are the reference; they are updated
as the golden JSON evolves. 2025 (version 20 Jan 2026): C 21.56, D 25.82, E 14.05, F 26.69,
G 22.25 = 110.37 fb⁻¹, reproduced per era to <0.01 fb⁻¹ with brilcalc --normtag
normtag_BRIL on `Cert_Collisions2025_391658_398903_Golden.json`. Re-check both before
the v1.2 production.

### MC availability (checked in DAS 2026-09-24)
No 2025 (Winter25/Summer25) TTto4Q or Z'→tt̄ exists; no Summer25 campaign at all.
Summer24 stays the 2025 MC. Winter25 QCD HT bins exist (QCD is data-driven).

---

## Validation log

| Date | Check | Result |
|---|---|---|
| 2026-09-24 | Z' 4 TeV 2024 smoke test, syst + noSyst | all ttag variations fill; overflow empty |
| 2026-09-24 | Veto-map unit tests (committed tree, clean checkout) | pass; processor imports |
| 2026-09-24 | 2025C data through 2025 JEC + veto map + jet ID | runs; veto pass 97.85% |
| 2026-09-24 | 2024 Z' as 2025 MC, 2025 vs 2024 payload | mean AK8 pT ratio 1.0002 |
| 2026-09-24 | Vendored JME files vs GitLab `latest` (Kerberos clone on lxplus) | md5 identical |
| 2026-09-24 | DAS: 2025 NanoAOD data + MC | only unused data = era B PromptReco; no new 2025 signal/ttbar MC |
| 2026-09-24 | brilcalc per era vs PPD table (newest golden) | all five eras match to <0.01 fb⁻¹ |
| 2026-09-24 | brilcalc 2025 C–G | 110.03 fb⁻¹ (current golden), 110.37 (newest) vs 110.59 in code |
| 2026-09-24 | Sweep of all 2025 correction packages on cvmfs (BTV/DC/EGM/JME/LUM/MUO/TAU) | relevant: JME (done), LUM pileup, DC golden JSON |
