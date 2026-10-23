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

Last updated: 2026-10-23 (extrapolation factor 2.0 adopted; nominal fit validated; 2022/23 T&P)

---

## Summary

| # | Change | Status | Commit | Affects | Physics effect |
|---|---|---|---|---|---|
| 1 | Real 2025 MC run (`--iov 2025`) instead of `scale_iov` (2024 × 1.00382); reverses v1.2 item 12 | Done, produced | `97fe15a` | 2025 TTbar + Z′ | 2025 MC gets its own JER SF (`Summer24Prompt25_JRV2`). The scaled files carry 2024's (`Summer24Prompt24_JRV2`): AK8 SF at \|η\| 2.3 is 1.36 / 1.42 (2024) vs 1.17 / 1.17 (2025) at 1 / 2 TeV, and within 1–3% for \|η\| < 1.8. MC JEC (L1L2L3) and the JES Total uncertainty are identical between the two payloads (checked at \|η\| 0–2.3, pT 0.5–2 TeV), so JES is unchanged. Fit side: JER for 2025 becomes its own nuisance (`jer25`) |
| 2 | 2025 pileup weights from the central LUM 2025 file; reverses v1.2 item 13 | Done | `97fe15a` | 2025 MC | `data/corrections/puWeights/2025_Summer24/puWeights.json.gz` = LUM `Run3-25Prompt-Summer24-NanoAODv15` 2026-06-05 (preliminary) `puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz`, correction `Collisions25_goldenJSON`, md5 `3fd35249…`. 2025/2024 weight at nTrueInt 20 / 40 / 60: 1.05 / 0.65 / 1.15 (data ⟨μ⟩ 51.8) |
| 3 | Top-tag SFs v1.3 from the high-pT T&P re-measurement (`toptag-sf-derivation` `ff241db`…`6cc18cf`, copied unchanged to `data/toptag/ttag_sf_v1.3.json`) | Done, produced | `97fe15a` | TTbar + Z′, 2024–2026 | Four pT bins 400–480 / 480–600 / 600–800 / 800–1200 GeV; tight ≈ 0.81 flat; band 1.18 / 1.25 / 1.52 ± 0.14 / 1.06 +0.65/−0.57 (interim 1.50 ± 0.30 and 1.00 ± 0.50 above 600 GeV). Consumer changes in `weights.py`: per-IOV `pt_edges`, loop over all bins (adds `ttag_pt4`), the JSON's extrapolation factors above 1.2 TeV instead of a fixed 2 (tight down ×3.4, up ×1.25, band ×1.74), drop `antitag_hipt` (replaced by the binned band SF), drop the `flep` weights (the v1.3 SFs include the f_lep < 0.59 efficiency). As implemented: one run writes both schemes, per-bin `ttag_pt1–4` (tight and band of a bin together) and the eigenvectors of the T&P covariance `ttag_eig1–8` (a fit uses one set); the extrapolation above 1.2 TeV is its own nuisance `ttag_extrap_tag` / `ttag_extrap_band` carrying √(f² − 1) × the last-bin deviation (adds in quadrature to the T&P total), so the deficit decision (item 4) is taken in the fit; v1.2 tables keep the doubling. bgestimation `67e187a`: configs `Run3v13pb*` / `Run3v13eig*` |
| 4 | Check of the tight-tag deficit above 1.2 TeV (1 tight probe where 6.6–7.6 are expected; tight SF < 0.44 at 95% CL from 40 probes) | Done (decision) | bgestimation `ecbf3fb` | Item 3's extrapolation | T&P follow-up (2026-10-23): 2 tight probes seen vs 7.4 expected (p = 0.022), degenerate with the probe composition, so the central SF stays and the deficit goes into the extrapolation width. The follow-up recommends an extrapolation factor of 2.0 on the last-bin deviation (`toptag-sf-derivation` `2d8eb72`) instead of the JSON's tight ×3.39 down / ×1.25 up and band ×1.74. Adopted as nominal by Aritra (2026-10-23) on the fit side, without a re-skim: config `Run3v13pbrsdtfbxfqdcxd` scales `ttag_extrap_tag` by 0.53 and `ttag_extrap_band` by 1.22, which gives factor 2.0 on the tight down side (−0.19, SF 0.60 at −1σ with the bin-4 uncertainty) and on the band (+1.13 / −0.99); the tight up side becomes +0.05 instead of +0.21 because a shape SIGMA is symmetric (exact templates at the next re-skim from a factor-2.0 SF JSON). The JSON's factors stay as the robustness variant (`Run3v13pbrsdtfbxfqdc`) |
| 5 | Event-level top-pT residual weight (TWiki data/NNLO exp(0.0416 − 0.0003 pT) on TOP-23-008, held at 1 TeV) with its up/down (no residual / residual²) | Done as a variation | `97fe15a` | TTbar | Nominal unchanged (TOP-23-008); variation `toppt_rsd`: Up = × residual, Down = × residual². bgestimation `make_ttbar_residual.py --from-template` takes Up as the residual model's nominal (no pT_eff approximation) |
| 6 | Correction-file caching and the review cleanup (carried over from v1.2) | Open | — | Code only | None |
| 7 | PDF uncertainty: std/mean → Hessian (v1.2 item 8) | Open | — | MC | PDF uncertainty ~10× larger |
| 8 | Provenance `git_sha` from the `COMMIT` file of a `git archive` export (batch-job copies have no `.git`) | Done | `d84dc75` | Output metadata | None |
| 9 | Fit side (bgestimation): JES and JER per year (2024 and 2025 split; JME default for a single total JES source and for JER: uncorrelated across years; all JES sources and the JER SF uncertainty are identical in the 2024/2025/2026 payloads, only the JER SF central value differs) | Done | bgestimation `67e187a` | Fit | On v1.2.1 inputs the split was ill-conditioned (identical 2024/2025 templates, model study #33); on v1.3 inputs the decorrelated fits pass the gate with it |
| 10 | `plots/make2Drootfiles.py`: systematic labels from the MC inputs only (data is nominal-only; the v1.2.1 data next to v1.3 MC still lists `ttag_band_hipt`) | Done | `b1a73ae` | Fit-input step | None (the v1.3 post step failed without it) |

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
| 2026-10-23 | Unit tests: `tests/test_ttag_weights_v13.py` (4 bins and binned band SF; no doubling and the extrapolation nuisance in quadrature; band extrapolation only on band jets beyond 1.2 TeV; eigen shifts reproduce the covariance diagonal; unrequested nuisances keep their nominal; v1.2 table unchanged; aliased IOVs; residual held at 1 TeV and from last-copy tops) + full suite with the v1.2 tests pinned to the v1.2 table | 100 pass, 4 skipped |
| 2026-10-23 | CMS central inputs (survey): no central Run-3 top-tag SFs (JMAR: Run-2 UL ParticleNet only); JES/JER correlation per the JERC docs; 2025 pileup file = the vendored one | Our T&P stays; JES/JER per year (item 9) |
| 2026-10-23 | Production (lxplus condor, `d84dc75` from EOS, workday, coffea-dask 2026.4.0 container): TTbar 473 file splits, Z′ 315 (7 IOVs × 3 widths × 15 masses) | 2023–2026 complete on the first pass; 17 TTbar splits (2022 Summer22/22EE files) and 1 Z′ job failed on xrootd reads via the global redirector and were resubmitted with a redirector fallback (FNAL, then INFN) |
| 2026-10-23 | Post-processing (`post_v13.sh` per IOV on lxplus condor, coffea-dask 2026.5.0 image because the v1.2.1 data outputs are numpy-2 pickles): merge + `make2Drootfiles.py` | 47 fit-input files per IOV in `2dAlphabetInputs/v1.3/twodalphabet_tight<yy>_v1.3`; staged `v1.3/run3_combined` (246 files) with the residual from the `TOPPT_RSD` templates and the smooth tail |
| 2026-10-23 | v1.3 vs v1.2.1 fit inputs (nominal integrals, all IOVs) | TTbar Pass −2%, Fail +0.5–1.6%; Z′ Pass −1%, Fail +4–6% (band SF above 800 GeV 1.00 → 1.06); 2025 a further −1% (Cen) / −2.5% (Fwd) from its own JER SF and pileup. All v1.3 templates present; every template the 18 v1.3 fit configs use exists (`check_cfg_templates.py`) |
| 2026-10-23 | Signal top-tag uncertainty at 4 TeV (central Pass) | `ttag_pt4` +30/−24% and `ttag_extrap_tag` +20/−61% (v1.2.1: `ttag_pt3` +36/−29%): the extrapolation's tight down side (SF 0.44 at −1σ above 1.2 TeV, item 4) roughly doubles the expected limits above 3.5 TeV; factor 2.0 adopted (item 4) |
| 2026-10-23 | v1.3 nominal fit (candidate B, tag SFs decorrelated tt̄/signal, per-bin `ttag_pt1–4`, factor 2.0; model study #43) | 42/42 masked b-only fits pass (1.2–8 TeV). Expected reach w1 / w10 / w30 4.78 / 6.21 / 7.55 TeV (v1.2.1: 4.97 / 6.53 / > 8; JSON factors: 4.63 / 5.99 / 7.29). Expected limits vs v1.2.1: 0.82–0.88× at 1.4–1.6 TeV, 1.3–1.5× (w1) at 2–2.5 TeV from the wider bin-4 measurement (unchanged without the extrapolation nuisance), 1.33–1.70× (w1) above 3 TeV. Masked GoF p = 0.13 (200 toys). Signal injection at the expected limit recovers 0.97 / 0.95 / 0.94 of the injected signal (median, 2 / 4 / 6 TeV); at r = 0 the median fitted r is within ±0.12 of the expected limit |
| 2026-10-23 | 2022 and 2023 top-tag T&P, each against its own MC (`toptag-sf-derivation` `4547742`) | Tight SF ratio to v1.3: 1.018 ± 0.033 (2022), 1.045 ± 0.043 (2023); band below 800 GeV 0.97 ± 0.08 / 1.04 ± 0.09; χ² 18.7 / 24 over 2022–2025 and the sub-eras agree, so one Run-3 table holds. Above 1.2 TeV: 4 tight probes of 54 where 8.99 are expected (p = 0.055), so factor 2.0 (item 4) stands. Fit-side year transfer `ttag_yr22/23` sized from the ratios (lnN 1.066 / 1.086 instead of the provisional 1.25) under test (model study #45) |
