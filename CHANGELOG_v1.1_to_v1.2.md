# Skimmer changelog: v1.1 → v1.2

Baseline **v1.1** = tag `v1.1` (`fab5038`, 2026-08-01): the production behind the
current 39-point limits. **v1.2** = the next MC + data re-skim.

Status: **Done** = committed · **Working tree** = written, not committed ·
**Planned** = decided, not started · **Open** = needs a decision.

Last updated: 2026-09-24 (round 3)

---

## Summary

| # | Change | Status | Commit | Affects | Physics effect |
|---|---|---|---|---|---|
| 1 | `ttag_pt2`/`ttag_pt3` variations actually filled | Done | `d664e04` | MC | Adds the top-tag SF uncertainty for jets with pT > 480 GeV (±15–17% on a 4 TeV Z') |
| 2 | Run-3 jet-veto map as a preselection cut | Done | `9e405c1` | Data + MC | Removes ~17–18% of high-HT signal events after MET filters; ~2% of 2025C JetMET events |
| 3 | 2025 → JME `Run3-25Prompt-Summer24` campaign | Done | `3c47f7c` | 2025 data JEC + MC JER, jet ID, veto map | 2025 data residuals now applied (item 16) |
| 4 | 2024 JEC V3 → V5, JER JRV1 → JRV2 | Done | `a694078` | 2024 data + MC | New η-dependent residuals and JER SFs |
| 5 | GloParTv3 score histograms `jet0/1_tdisc` | Done | `28c02a9` | Output only | None (new diagnostics for CR/VR data/MC) |
| 6 | Missing HLT path raises instead of accepting all events | Done | `8a078df` | Safety | None in a correct setup |
| 7 | Antitag jet SF: measure the [medium, tight) band SF directly | Approved | — | MC | Loose SF ~0.93–1.05 now; band SF from existing SFs ~1.2–1.5 but ill-constrained |
| 8 | PDF uncertainty: std/mean → Hessian | Last (reconsider) | — | MC | PDF uncertainty ~10× larger |
| 9 | msoftdrop rebuilt from re-corrected subjets; JES/JER from varied subjets; JMS 1%, JMR 2% | Done | `d3fbc3e`, `e922747` | Data + MC | Nominal mSD: 2024 data +0.3%, 2025 data +3.3%, MC −0.9%; JES ±0.7% on ⟨mSD⟩ |
| 10 | Q2/PDF templates normalized to generator-level nominal sumw | Done | `5bcdf08` | MC | Z′ 4 TeV q2 ±11% → ±2.4%, pdf ±5% → ±1.4% (acceptance only) |
| 11 | AK8 candidates: PUPPI Tight jet ID | Done | `27d8c36` | Data + MC | Z′ 99.85%, tt̄ 100%, certified 2025C data 98.9% of AK8 jets |
| 12 | Real 2025 MC run instead of `scale_iov` | Decided: no | — | 2025 MC | 2024 MC stands in for 2025 and 2026 (2024 × lumi) |
| 13 | 2025 pileup weights (LUM 2025 file) | Decided: no | — | 2025 MC | Stays with 2024 (follows item 12) |
| 14a | 2025 golden JSON → `..._398903` + lumi 110.59 → **110.37** fb⁻¹ (PPD table) | Done | `d4b719c` | 2025 data + MC norm | 2025 MC −0.2%; +1056 / −859 LS in data |
| 14b | 2022/2023/2024 golden JSONs → DC re-issue after the 2026 tracker-ML review | Done | `f6ff4c1` | Data | LS: 2022 +0.07%, 2023 −0.43%, 2024 −0.04% |
| 14c | 2022/2023/2024 lumi values | Decided: unchanged | — | — | PPD values not updated since; keep |
| 15 | 2025 era B data | Dropped | — | — | PPD table (reference) covers eras C–G only |
| 16 | Re-apply JEC to data (and `--noSyst` MC), Run-3 IOVs | Done | `77fa52e` | Data | **Data AK8 pT −4.6% (2024), −2.9% (2025)** vs NanoAOD JEC, pT > 400 GeV |
| 17 | ISR/FSR parton-shower variations (`isr`, `fsr`), normalized like Q2/PDF | Done | `13dc1e3` | MC | Z′ 4 TeV (W10, local): ISR ±0.3%, FSR ±2.1% on the selected yield; none in v1.1 or Run 2 |
| 18 | Top-tag SF uncertainty doubled for jets above 1.2 TeV (beyond the T&P data) | Done | `571526c` | MC | Z′ 4 TeV (W10, local): `ttag_pt3` +15.5/−14.4% → +23.5/−20.8%; nominal unchanged |
| 19 | Weight variations no longer skipped for a category without gen-matched tops | Done | `2ec6420` | MC | v1 TTbar: 305 of 103,611 events were missing from every weight-variation template; Z′ W1% 400–700 GeV up to 11% (review numbers) |
| 20 | Chunks with 1–9 events after the baseline selection kept | Done | `2ec6420` | Data + MC | Those events were dropped from the templates but kept in sumw; small (file-tail chunks) |
| 21 | AK8 JER gen pT from the matched GenJetAK8 (was the nearest AK4 GenJet) | Done | `7298517` | MC | Z′ 2 TeV: smeared AK8 pT −0.2% on average; gen match 91% → 100% |
| 22 | Notebook runner uses the CLI's `_signal_xsec` (1-TeV-spike fix) | Done | `d02e9d7` | Signal run from the notebook | None for CLI runs; notebook signal runs no longer save 400–900 GeV at raw counts |

Downstream (bgestimation, after the v1.2 inputs exist): attach `ttag_pt2`/`ttag_pt3`
in the six Run-3 configs, add `jms`/`jmr`/`isr`/`fsr` likewise, then re-fit the 39 points. Update the hard-coded lumi labels
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
  correlated model/JMR/JES/JER/PU nuisances — **decided no (2026-09-24): one total per
  bin, as in Run 2**; (b) the top bin is measured mostly at 600–800 GeV and applied to
  1.5–2 TeV jets with no extrapolation uncertainty — **decided (2026-09-24): above 1.2 TeV
  the uncertainty is doubled, same nuisance (item 18, `571526c`)**; an 800 GeV+ T&P bin is
  fitted as a check only (if it agrees with 600–800, the three-bin scheme stays).

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

### 19–22. Code-review bug fixes — `2ec6420`, `7298517`, `d02e9d7`
A static review of the skimmer (separate session, 2026-09-24) found four bugs that
predate v1.2; none was a decision. Fixed before the re-skim:

- **19.** In the per-category loop, `if ak.sum(truth_cat_mask) == 0: continue` sat in
  the gen-truth block, above the weight-variation fills, so a category with events
  but no gen-matched tops lost its pileup/PDF/Q2/ISR/FSR/ttag_pt fills and its
  `systematics` counter (up and down both low). QCD MC was hit in every category.
  Now only the truth fills are guarded.
- **20.** `if len(events) < 10: return output` after the baseline selection ran after
  sumw was recorded. Both early returns now fire only for empty chunks. The first
  one (before sumw) dropped whole tiny chunks consistently, so only data lost events there.
- **21.** `Run3JetManager.prepare_for_corrections` set FatJet `pt_gen` from the nearest
  AK4 GenJet within ΔR 0.2 (the AK4 recipe), overriding the GenJetAK8 fallback in
  `GetJECUncertainties`. Z′ 2 TeV, AK8 pT > 400 GeV: median (pT − pT_gen)/pT +8.3% →
  −0.6%. The top-tag T&P already used GenJetAK8.
- **22.** fab5038 fixed `_signal_xsec` in `ttbaranalysis.py` only; `ttbar_notebook.py`
  now imports it.

Tests: `tests/test_processor_chunks.py` (15-event chunks of the local Z′ 2 TeV file;
skipped without it) fails on the old code for 19 and 20 separately;
`tests/test_signal_xsec.py` for 22.

Other review findings (truth-histogram memory, correction-file caching, the two
runners, lumi constants in six places, Run-2 leftovers) are not in v1.2 yet. Local
note: with coffea's futures/iterative executors, merging three or more chunks crashes on
`event_list` (`list_accumulator.identity()` returns a plain list, same code in coffea
2025.12.0–2026.5.0). The Dask path merges in place and is unaffected (v1.1 production).

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

**Further findings (2026-09-24, from the local T&P ntuples `ntuples_2024_full_fswp`):**
- **Threshold mismatch.** The T&P flags use per-pT thresholds (tight 0.926 → 0.917,
  medium 0.855 → 0.837); the analysis cuts at the flat scalars 0.9284 / 0.8571. Fully
  merged tt̄ MC ε_T at the analysis cut vs the T&P flag: 0.663 vs 0.667 (400–480),
  0.669 vs 0.677 (480–600), 0.676 vs 0.694 (600+). The tight SF must also be re-measured
  at the analysis cut (no reprocessing needed for this alone: the ntuples store the raw
  score).
- **Probe pT reach.** 29065 data probes: 14313 / 9909 / 3993 / 674 / 128 / 48 in
  400–480 / 480–600 / 600–800 / 800–1000 / 1000–1200 / 1200+. The 600+ bin has median
  680 GeV (90% below 870); multi-TeV Z′ jets at 1.5–2 TeV are an extrapolation.
- **Band at the analysis cuts.** ε_band ≈ 0.09 of fully merged tops in every pT bin
  (ε_T ≈ 0.66–0.68).
- **T&P object definitions predate v1.2**: V3/JRV1 JEC, NanoAOD msoftdrop, no veto map,
  no AK8 jet ID. Re-producing the ntuples with the v1.2 definitions covers items 7 and
  the T&P msoftdrop follow-up in one pass.

**T&P port to the v1.2 objects — done locally (2026-09-24), `toptag-sf-derivation`
`81538d9` (branch `aritra`, not pushed):** `--apply-jec` re-corrects AK8, AK4 and subjets
(V5/JRV2; data JEC with residuals) and rebuilds msoftdrop from the subjets (JES/JER
variations included; NanoAOD value kept as `probe_msd_nano`); probe PUPPI Tight ID and
|y| < 2.4 (was |η| < 2.5); tag-side AK4 Tight ID; jet-veto map; re-issued 2024 golden
JSON; missing trigger/golden JSON/JEC failures now raise. 22 unit tests pass. On 2024
JetMET data (73716 AK8 jets, pT > 400): corrected/NanoAOD pT 0.954, rebuilt/NanoAOD mSD
1.004 (skimmer: 0.954, 1.003). **Next:** casa production (Data, TTbar, SingleTop, WJets;
`--apply-jec --apply-pu`, MC also `--apply-jec-syst`), then the three-category fit at the
analysis cuts (pass ≥ 0.9284, band [0.8571, 0.9284), fail) with an 800 GeV+ check bin.

**Band-SF fit ready, production running on the LPC (2026-09-24).** `toptag-sf-derivation`
`519c9d0` / `4dc71d6`: three-category fit (scipy `p1_bandfit.py`, Combine
`make_p1_band_datacards.py`, systematics `p1_band_systematics.py`); closure and injection
exact. Preview on the **pre-v1.2** ntuples (Combine, rest tag rates fixed, jms profiled):

| pT bin | SF (D ≥ 0.9284) | band SF [0.8571, 0.9284) | loose SF applied now |
|---|---|---|---|
| 400–480 | 0.929 ± 0.029 | 1.33 ± 0.10 | 1.05 |
| 480–600 | 0.821 ± 0.03 | 1.29 ± 0.09 | 0.93 |
| 600–800 | 0.834 ± 0.05 | 1.62 ± 0.13 | 0.95 |
| 800+ | 0.714 +0.087/−0.063 | 0.69 ± 0.22 | 0.95 |

Stat errors only; systematics (old objects) add ~0.04 (SF) and ~0.07 (band SF), mostly JMR.
The tight SF above 800 GeV agrees with 600–800 within ~1.3σ; the band SF there does not
(58 band events) — re-check on v1.2. coffea-casa's worker pool was down (2 slots, all casa
worker jobs idle since 2026-09-09), so the v1.2 T&P production runs on the LPC
(`~/nobackup/toptag-sf-derivation`, coffea 2026.4.0; runner fix `1ca2b22`).

**v1.2 production status (20:45 CEST).** Data (7 eras, 2.5 h) and TTbar (2.5 h) done.
Data probes 20,938 vs 29,065 with the old objects (−28%; expected direction from the
−4.6% data AK8 pT, Tight ID, |y| < 2.4 and veto map — to be validated). SingleTop
failed on a 152-event file whose 2 preselected events had no AK8 jet: coffea's JER
smearing raises on an empty collection. Fixed in T&P `450ee8d` (return before the
corrections when a chunk has no AK8/AK4/subjet; checked on that file at the LPC);
SingleTop + WJets relaunched. Preliminary scipy fit on data + TTbar (non-tt̄ component
floated, stat only): SF 0.81/0.85/0.84/0.96, SB 1.35/1.30/1.75 for 400–480/480–600/
600–800/800+ GeV; SB above 800 GeV at the lower bound (25 band events). Combine fit
after the full MC.

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
**Update — `e922747` (supersedes the pT-ratio propagation below for Run 3).**
Following the JMAR recipe used in `smp_jetmass_run2` (`dijet_processor.py`): the soft-drop
subjets are un-corrected (`rawFactor`) and re-corrected with the vendored AK4 PUPPI JEC
(data: DATA chain with residuals; MC: MC chain + JER, subjet pT_gen from the nearest
`SubGenJetAK8`, ΔR < 0.4), and `msoftdrop` = mass(subjet1 + subjet2) is rebuilt for the
nominal and the JES/JER-varied passes. Jets without two subjets keep the NanoAOD value.
Rebuilding from the untouched NanoAOD subjets reproduces NanoAOD `msoftdrop` exactly.

| Sample | median new / NanoAOD mSD |
|---|---|
| 2024 data | 1.0026 |
| 2025 data | 1.0277 |
| 2024 tt̄ MC (nominal, JEC + JER) | 0.9914 |

Data and MC now move toward each other (the T&P saw the data top peak ~3% below MC with
NanoAOD mSD). 2025 data moves most: its PromptReco subjets carried smaller residuals.

**Follow-up:** the top-tag SF measurement (`toptag_sf_processor.py`) still uses NanoAOD
`msoftdrop` for its mSD window. For full consistency it should use the same rebuilt
msoftdrop, most naturally when the antitag band SF (item 7) is measured.

**Original — `d3fbc3e`.** Nominal JMS = JMR = 1.000 (no nominal correction; mSD stays on the
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

### 16. JEC re-applied to data — `77fa52e`
`build_corrections` returns `None` for data and for `--noSyst`, so those events keep
the JEC embedded in NanoAOD, while MC with systematics gets the re-applied payloads
(2024 V5, 2025 Summer24Prompt25_V3). CMS recommends the same JEC version on data and
MC.

**Fix:** Run-3 IOVs always run the correctionlib JEC. Data: DATA L1L2L3Res chain with
run-dependent residuals, nominal pass only, no JER smearing. `--noSyst` MC: nominal
JEC + JER. Run 2 unchanged (legacy data-JEC path unusable). `msoftdrop` untouched.

**Size (data, AK8 pT > 400 GeV, new / NanoAOD JEC):** 2024 **0.954**, 2025 **0.966**;
AK4 (pT > 100) 0.993 / 0.985. Checked: AK8 and AK4 L2L3Residual are identical at every
η/run/pT; V5 re-derived residuals vs η (central ≈ 0.99–1.00). The difference is the JEC
version embedded in the NanoAOD data vs V5. The top-tag SF measurement already applied
this data JEC, so the SFs are consistent with it.

**Expect in v1.2:** data mtt, jet pT and HT shift down by ~4–5% (2024) relative to
v1.1; fewer data events pass HT > 1400 GeV. Validate with data/MC jet pT and HT
comparisons on the first v1.2 outputs.

Note (pre-existing): `--noSyst` MC skips all event weights after `weights.py:15`
(pileup, top-tag SF, ...), so its nominal yield is ~11% above the full-syst nominal
for Z' 4 TeV. Fit inputs always come from full-syst runs.

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

## Decision log

| Date | Decision |
|---|---|
| 2026-09-24 | 2022/2023/2024 luminosity values stay (PPD not updated). Q2/PDF normalization approved and done (`5bcdf08`). AK8 Tight jet ID (not TightLepVeto) approved and done (`27d8c36`). Top-tag SF nuisance model: to be improved on physics grounds. |
| 2026-09-24 | **Correction:** the earlier 2025 data numbers (item 3/16: AK8 0.966, AK4 0.985, mSD 1.028) used run 392293 without the lumi mask; LS 1–35 are uncertified tracker-off data. Certified LS only: AK8 **0.971**, AK4 **1.002**, mSD **1.033** (one run, low statistics). 2024 numbers unaffected (file 100% certified). |
| 2026-09-24 | Presented (Slides artifact 6aw4XinzZgEhf8WiU7KfS4; research-notes `topics/ttbarhadronic_skimmer_v12_changes_and_mc_plan.md`). **2025 and 2026 MC: use 2024 (Summer24) for everything** — `scale_iov` 2024 × lumi, no 2025 MC run, no 2025 pileup (items 12, 13). **2022/2023 Z′ signal:** contact the B2G MC contact for a NanoAODv15 re-processing. **Approved as presented:** antitag band SF measured directly (7), PDF recipe last (8), rerun list and order. |

| 2026-09-24 | **ISR/FSR:** add both (done, `13dc1e3`). **Top-tag SF nuisances:** keep one total (stat ⊕ syst) uncertainty per pT bin, bins uncorrelated — the Run-2 scheme (`ttag_pt1/2/3_{16,17,18}` in the Run-2 configs); no stat/syst split. **Extrapolation:** jets above 1.2 TeV (where the T&P data run out) keep the top-bin SF with doubled uncertainty, same nuisance (done, `571526c`); an 800 GeV+ T&P bin is fitted as a check only. |
| 2026-09-24 | Code review (separate session): its bugs 1–4 are pre-existing bugs, not v1.2 decisions; fixed for v1.2 (items 19–22). |

## Validation log

| Date | Check | Result |
|---|---|---|
| 2026-09-24 | Items 19–22: unit tests (74) + `tests/test_processor_chunks.py` on old code | 74/74 pass; old code fails the few-events check, old code + only the item-20 fix fails the weight-variation check (3 vs 4) |
| 2026-09-24 | Item 20: 50-event vs one 2000-event chunk (Z′ 2 TeV, nominal) | old 147 vs 149, new 148 vs 149; the remaining events differ in both directions (3 vs 5), i.e. per-chunk random smearing, not dropped events |
| 2026-09-24 | Item 21: AK8 JER old vs new gen match (Z′ 2 TeV, 26k jets) | median (pT − pT_gen)/pT +8.3% → −0.6%; smeared pT new/old 0.998 |
| 2026-09-24 | Z′ 4 TeV local CLI smoke test after items 19–22 | exit 0; categories 2581/2564/4479/5793; 1 pb reference applied |
| 2026-09-24 | Z' 4 TeV 2024 smoke test, syst + noSyst | all ttag variations fill; overflow empty |
| 2026-09-24 | Veto-map unit tests (committed tree, clean checkout) | pass; processor imports |
| 2026-09-24 | 2025C data through 2025 JEC + veto map + jet ID | runs; veto pass 97.85% |
| 2026-09-24 | 2024 Z' as 2025 MC, 2025 vs 2024 payload | mean AK8 pT ratio 1.0002 |
| 2026-09-24 | Vendored JME files vs GitLab `latest` (Kerberos clone on lxplus) | md5 identical |
| 2026-09-24 | PSWeight index parsing on local Z′, TTto4Q (4 entries) and QCD (44 entries) | correct isr/fsr ×2/×0.5 entries in all three |
| 2026-09-24 | Z′ 4 TeV (W10) local smoke test with isr/fsr | isrUp/Down −0.3/+0.3%, fsrUp/Down +2.1/−2.2%; norm factors 1.000–1.001 |
| 2026-09-24 | Top-tag extrapolation: unit tests (`tests/test_ttag_weights.py`) + Z′ 4 TeV smoke test | 3/3 pass (doubling only above 1.2 TeV, antitag jet included, Run 2 untouched); `ttag_pt3` +23.5/−20.8%, nominal identical |
| 2026-09-24 | DAS: 2025 NanoAOD data + MC | only unused data = era B PromptReco; no new 2025 signal/ttbar MC |
| 2026-09-24 | brilcalc per era vs PPD table (newest golden) | all five eras match to <0.01 fb⁻¹ |
| 2026-09-24 | brilcalc 2025 C–G | 110.03 fb⁻¹ (current golden), 110.37 (newest) vs 110.59 in code |
| 2026-09-24 | Sweep of all 2025 correction packages on cvmfs (BTV/DC/EGM/JME/LUM/MUO/TAU) | relevant: JME (done), LUM pileup, DC golden JSON |
