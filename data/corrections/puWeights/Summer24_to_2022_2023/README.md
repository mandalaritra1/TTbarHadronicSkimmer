# Pileup weights: 2022/2023 data on Summer24 MC

Used for the 2022/2023 Z′ signal, which is the 2024 Summer24 sample standing in (there is no
NanoAODv15 Z′ for 2022/2023). The central LUM weights for 2022/2023 divide by the Summer22/23 MC
profile. On Summer24 MC their mean weight is 0.72 (2022preEE), 1.39 (2022postEE), 1.36 (2023preBPix)
and 1.40 (2023postBPix), which biased the 2022/23 Z′ normalisation in the first v1.2.1 production
(changelog item 37). No central 2022/23-on-Summer24 product exists, so these weights are derived
from central inputs.

## Method (`scripts/derive_pu_summer24_to_2022_2023.py`, 2026-10-02)

w(n) = data_IOV(n) / MC_Summer24(n), both normalised, 99 bins of nTrueInt (0–99, clamped).

- **Data:** DQM/LUM certification pileup histograms (golden JSON), 69.2 mb nominal, 72.4 mb up,
  66.0 mb down. These are `inputs/pileupHistogram-Cert_Collisions{2022,2023}_*-13p6TeV-*ub-99bins.root`
  from https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/{BCD,EFG} and
  Collisions23/PileUp/{BC,D}.
- **MC:** `Pileup_nTrueInt` of 1,388,000 Summer24 NanoAODv15 events (Z′ W10 1/3/5 TeV, W30 2 TeV, W1
  4 TeV, TTto4Q; 3 files each), mean 45.45. The profile implied by the central 2024 weights (data24 /
  w24) has mean 45.43 and agrees within 6% bin by bin in the populated range. The profile is stored in
  `derivation_summary.json`.
- Bins without MC get weight 1.

Mean weight on the Summer24 profile: 1.000 for all four IOVs (nominal, up and down within 0.2%).

## Validation against the central weights (`scripts/validate_pu_summer24_method.py`)

Two central products use the same MC: 2024 data on Summer24 (LUM Run3-24CDEReprocessingFGHIPrompt-Summer24,
`puWeights_CDEFGHI`, correction `Collisions24_CDEFGHI_goldenJSON`) and 2025 data on Summer24 (LUM
Run3-25Prompt-Summer24, `puWeights_2025pp_Golden_Summer24_25ns_69200ub`, `Collisions25_goldenJSON`).
Rebuilding them with this method from the 2024/2025 certification histograms (2024 inputs vendored here)
gives:

| case | mean weight on Summer24, method / central | method/central in the populated bins: weighted mean, max deviation |
|---|---|---|
| 2024 nominal (up, down) | 1.000 / 1.002 (1.000 / 1.003, 1.000 / 1.001) | 0.5%, 6.4% at n = 20 |
| 2025 nominal (up, down) | 1.000 / 1.002 (1.000 / 1.004, 1.000 / 1.001) | 0.5%, 6.4% at n = 20 |

Only the sparsely populated low-pileup bins (n ≲ 12) differ by more, and they carry a negligible share of
Summer24 events.

## Use

`corrections.GetPUSF(events, IOV, mc_campaign="Summer24")` with a 2022/2023 IOV uses
`{IOV}_data_on_Summer24` from `puWeights.json.gz`. `weights.Run3WeightManager.build_weights` sets the
campaign from the input file name (`RunIII2024Summer24` in the path).
