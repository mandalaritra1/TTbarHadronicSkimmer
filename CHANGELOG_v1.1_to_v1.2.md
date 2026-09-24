# Skimmer changelog: v1.1 → v1.2

Baseline **v1.1** = tag `v1.1` (`fab5038`, 2026-08-01): the production behind the
current 39-point limits. **v1.2** = the next MC + data re-skim.

Status: **Done** = committed · **Working tree** = written, not committed ·
**Planned** = decided, not started · **Open** = needs a decision.

Last updated: 2026-09-24

---

## Summary

| # | Change | Status | Commit | Affects | Physics effect |
|---|---|---|---|---|---|
| 1 | `ttag_pt2`/`ttag_pt3` variations actually filled | Done | `d664e04` | MC | Adds the top-tag SF uncertainty for jets with pT > 480 GeV (±15–17% on a 4 TeV Z') |
| 2 | Run-3 jet-veto map as a preselection cut | Done | `9e405c1` | Data + MC | Removes ~17–18% of high-HT signal events after MET filters; ~2% of 2025C JetMET events |
| 3 | 2025 → JME `Run3-25Prompt-Summer24` campaign | Done | `3c47f7c` | 2025 data (+ 2025 MC JER) | 2025 data JEC with 2025 residuals (AK8 pT −3.1% vs NanoAOD-embedded); 2025 jet ID works |
| 4 | 2024 JEC V3 → V5, JER JRV1 → JRV2 | Working tree | — | 2024 data + MC | New η-dependent residuals and JER SFs |
| 5 | GloParTv3 score histograms `jet0/1_tdisc` | Working tree | — | Output only | None (new diagnostics for CR/VR data/MC) |
| 6 | Missing HLT path raises instead of accepting all events | Working tree | — | Safety | None in a correct setup |
| 7 | Antitag jet SF | Open | — | MC | Fail-window jet currently gets the loose-WP SF |
| 8 | PDF uncertainty: std/mean → Hessian | Planned | — | MC | PDF uncertainty ~10× larger |
| 9 | Soft-drop mass scale/resolution systematic | Planned | — | MC | New shape nuisance on the fit mass axis |
| 10 | Q2/PDF templates yield-normalized | Open | — | MC | Separates acceptance from rate |
| 11 | AK8 jet ID on the two leading jets | Open | — | Data + MC | Not applied in v1.1 or v1.2 so far |
| 12 | Real 2025 MC run instead of `scale_iov` | Open | — | 2025 MC | Uses the 2025 JER SF for forward jets |
| 13 | 2025 pileup weights (LUM 2025 file) | Open | — | 2025 MC | 2025 MC currently reweighted to the **2024** data pileup profile |
| 14 | Golden JSON refresh + brilcalc luminosity | Open | — | Data + MC norm | 2025 C–G lumi: code 110.59, brilcalc 110.03 (current golden) / 110.37 (newest) fb⁻¹ |
| 15 | Add 2025 era B data (JetMET0/1 PromptReco) | Open | — | Data | +0.254 fb⁻¹ certified (+0.23%) |

Downstream (bgestimation, after the v1.2 inputs exist): attach `ttag_pt2`/`ttag_pt3`
in the six Run-3 configs, then re-fit the 39 points.

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
  - On 2025C data (pT > 200): AK8 corrected pT / NanoAOD pT = 0.9695, AK4 = 0.9900.
- **Jet ID:** 2025 NanoAODv15 has no `Jet_jetId`; the campaign's `jetid.json.gz`
  (byte-identical to 2024's) is now used, so 2025 data runs.
- **MC:** all MC JEC and the Total JES uncertainty are identical to Summer24Prompt24_V5.
  Only the JER SF differs (`Summer24Prompt25_JRV2`), see item 12.
- **Veto map:** identical vetoed bins to the old file (169/5904); only the name changed.

---

## Working tree (not committed yet)

### 4. 2024 JEC V5 / JER JRV2

- `data/corrections/jsonpog/JME/2024_Summer24/{jet,fatJet}_jerc.json.gz` updated to
  the `Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15` 2026-07-16 release
  (`Summer24Prompt24_V5`, `Summer24Prompt24_JRV2`). Byte-identical to GitLab `latest`.
- v1.1 MC used V3/JRV1; v1.1 data used the NanoAOD-embedded JEC.
- V4 → V5: the V4 release shipped unchanged residuals by mistake; V5 has the new
  η-dependent L2L3Residual.

### 5. Top-score histograms

- `jet0_tdisc`, `jet1_tdisc` (GloParTv3 Top-vs-QCD, finer bins above 0.9) per
  category and systematic, plus two entries in `plots/make_leading_jet_datamc_plots.py`.
  Requested by the B2G audit for CR/VR data/MC.

### 6. Trigger safety

- A missing `HLT` branch or configured path now raises; v1.1 warned and accepted
  every event.

---

## Planned / open

### 7. Antitag SF (open)
The antitag (Fail) jet sits in the [medium, tight) score window but gets the
loose-WP SF (`weights.py:101-103`). Options: medium-WP SF, or an SF for that band.

### 8. PDF uncertainty (planned)
`GetPDFWeights` uses std/mean across replicas; the PDF set needs the Hessian
formula, so the uncertainty is ~10× too small.

### 9. Soft-drop mass systematic (planned)
`msoftdrop` has no scale/resolution variation, but it is the fit's jet-mass axis.
Needs JMS/JMR variations in the processor.

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

### 15. 2025 era B data (open)
DAS has `/JetMET{0,1}/Run2025B-PromptReco-v1/NANOAOD` (13.6M events each); `data.json`
starts at era C. The golden JSON certifies 24 era-B runs (391668–392046, 6691 LS),
0.254 fb⁻¹ recorded. Small, but it is certified data. The 2025 lumi would need
updating if added.

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
| 2026-09-24 | brilcalc 2025 C–G | 110.03 fb⁻¹ (current golden), 110.37 (newest) vs 110.59 in code |
| 2026-09-24 | Sweep of all 2025 correction packages on cvmfs (BTV/DC/EGM/JME/LUM/MUO/TAU) | relevant: JME (done), LUM pileup, DC golden JSON |
