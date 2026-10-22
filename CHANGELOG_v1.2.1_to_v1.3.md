# Skimmer changelog: v1.2.1 → v1.3

Baseline **v1.2.1** = tag `v1.2.1` (`c34dbd1`, 2026-10-02): the production in
`outputs/tight_v1.2.1` behind the current Run-3 fits. Two later productions also belong
to it: the 2022/23 Z′ pileup rerun (`c7c1365`, item 37 of the v1.2 changelog) and the
2026 MC with `--iov 2026` (`5de1df9`). The full v1.1 → v1.2.1 history is in
`CHANGELOG_v1.1_to_v1.2.md`.

**v1.3** = the next TTbar + Z′ MC re-skim (data only if a change below touches it). It
gets the tag `v1.3` on the commit that starts the production.

Status: **Done** = committed · **Working tree** = written, not committed ·
**Planned** = decided, not started · **Open** = needs a decision.

Last updated: 2026-10-22

---

## Summary

| # | Change | Status | Commit | Affects | Physics effect |
|---|---|---|---|---|---|
| 1 | Real 2025 MC run (`--iov 2025`) instead of `scale_iov` (2024 × 1.00382); reverses v1.2 item 12 | Planned | — | 2025 TTbar + Z′ | 2025 MC gets its own JER SF (`Summer24Prompt25_JRV2`). The scaled files carry 2024's (`Summer24Prompt24_JRV2`): AK8 SF at \|η\| 2.3 is 1.36 / 1.42 (2024) vs 1.17 / 1.17 (2025) at 1 / 2 TeV, and within 1–3% for \|η\| < 1.8. MC JEC (L1L2L3) and the JES Total uncertainty are identical between the two payloads (checked at \|η\| 0–2.3, pT 0.5–2 TeV), so JES is unchanged. Fit side: JER for 2025 becomes its own nuisance (`jer25`) |
| 2 | 2025 pileup weights from the central LUM 2025 file; reverses v1.2 item 13 | Planned | — | 2025 MC | Needs the LUM 2025 file vendored under `data/corrections/puWeights/` (not there yet) |
| 3 | Top-tag SFs v1.3 from the high-pT T&P re-measurement (`toptag-sf-derivation` `ff241db`…`6cc18cf`, `results/outputs/ttag_sf_v1.3_candidate.json`) | Planned (after the check in item 4) | — | TTbar + Z′, 2024–2026 | Four pT bins 400–480 / 480–600 / 600–800 / 800–1200 GeV; tight ≈ 0.81 flat; band 1.18 / 1.25 / 1.52 ± 0.14 / 1.06 +0.65/−0.57 (interim 1.50 ± 0.30 and 1.00 ± 0.50 above 600 GeV). Consumer changes in `weights.py`: per-IOV `pt_edges`, loop over all bins (adds `ttag_pt4`), the JSON's extrapolation factors above 1.2 TeV instead of a fixed 2 (tight down ×3.4, up ×1.25, band ×1.74), drop `antitag_hipt` (replaced by the binned band SF), drop the `flep` weights (the v1.3 SFs include the f_lep < 0.59 efficiency). bgestimation: `ttag_pt4` nuisance, `ttag_band_hipt` removed |
| 4 | Check of the tight-tag deficit above 1.2 TeV (1 tight probe where 6.6–7.6 are expected; tight SF < 0.44 at 95% CL from 40 probes) | Open | — | Decides item 3's extrapolation | If real, the high-mass Z′ (both jets above 1.2 TeV) loses most of its Pass efficiency; T&P session proposed a follow-up, also with the 2026 data |
| 5 | Event-level top-pT residual weight (TWiki data/NNLO exp(0.0416 − 0.0003 pT) on TOP-23-008, held at 1 TeV) with its up/down (no residual / residual²) | Open (depends on adopting the fit model-study candidate) | — | TTbar | Replaces the per-bin approximation in bgestimation `make_ttbar_residual.py`; tt̄ ≈ 0.85 × MC, ≈ 0.77 at high m_tt |
| 6 | Correction-file caching and the review cleanup (carried over from v1.2) | Open | — | Code only | None |
| 7 | PDF uncertainty: std/mean → Hessian (v1.2 item 8) | Open | — | MC | PDF uncertainty ~10× larger |

## Samples (DAS, checked 2026-10-22)

- No 2025 TTto4Q or Z′ MC: TTto4Q exists only in `RunIII2024Summer24` (NanoAODv15 + ext1/ext2) and a low-PU
  `RunIII2026LowPUSummer26` campaign; `Run3Winter25` has QCD (HT and pT binned) and a few other processes, no
  TTto4Q and no Z′. Item 1 therefore runs the Summer24 files with 2025 corrections, as `--iov 2026` does.
- Z′ → tt̄ 2022/2023: `Run3Summer22(EE)` and `Run3Summer23(BPix)` exist in NanoAODv12 and MiniAODv4 (not v15,
  so no GloParTv3 in NanoAOD); the analysis uses the 2024 signal as a placeholder for 2022/23.
- RS (KK) gluon and the leptophobic Z′ DM mediator (`RSGluonToTT_M-*`, `ZprimeDMToTTbarResoIncl_MZp*_Mchi10_V1`):
  Run 2 only (UL16/17/18 and older); no 13.6 TeV samples under any name tried. Using them in Run 3 needs a new MC
  request.

## Validation log

| Date | Test | Result |
|---|---|---|
| 2026-10-22 | 2024 vs 2025 AK8 JME payloads (correctionlib, \|η\| 0 / 1 / 1.8 / 2.3, pT 0.5 / 1 / 2 TeV) | MC L1L2L3 and JES Total identical; JER SF differs (item 1) |
