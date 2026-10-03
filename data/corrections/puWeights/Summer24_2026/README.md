# Pileup weights: 2026 data on Summer24 MC

2026 MC is the 2024 Summer24 MC, so 2026 needs data26 / MC_Summer24 weights. As of 2026-10-03 there
is no central 2026 product: no LUM correction, no certification pileup histogram, and no pileup
JSON for Collisions26. Until then this replaces the 2024 weights that 2026 borrowed before.

## Method (`scripts/derive_pu_2026_summer24.py`, 2026-10-03)

w(n) = data26(n) / MC_Summer24(n), both normalised, 99 bins of nTrueInt (0–99, clamped). Bins
without MC get weight 1.

- **Data profile.** The central histograms come from pileupCalc ("true" mode) on a per-LS pileup
  JSON. Each lumisection adds a Gaussian N(μ, σ) weighted by its luminosity, with μ and σ the mean
  and rms of the per-bunch luminosity × σ_inel. Without a 2026 pileup JSON this is rebuilt from
  brilcalc:
  - μ(LS) = k × brilcalc avgpu, from `brilcalc lumi --byls --minBiasXsec 69200 --normtag
    normtag_BRIL.json -i Cert_Collisions2026_401624_403937_golden.json`. Here k = 1.0051, the 2025
    pileup-JSON ⟨μ⟩ over the brilcalc ⟨μ⟩: normtag_BRIL runs 0.5% low.
  - σ(LS) = r × μ(LS), averaged over the 10/30/50/70/90% quantiles of the 2025 bunch spread r =
    rms/mean: 0.053, 0.061, 0.068, 0.076, 0.090.
  - Up and down use 72.4 and 66.0 mb (μ and σ scale with the cross section).
- **MC profile:** the measured Summer24 profile from `../Summer24_to_2022_2023/derivation_summary.json`.

2026 eras B+D: 25.79 fb⁻¹ in the golden JSON (era A included, era C not in it). Mean pileup 54.4
(up 56.9, down 51.9).

## Validation on 2025 (central `dataPileupHistogram-2025pp_Golden-{66000,69200,72400}ub.root`)

Weights on the Summer24 profile, in bins holding > 0.1% of Summer24:

| case | weighted mean of the bin deviation | max | ⟨PU⟩ mine / central |
|---|---|---|---|
| pileupCalc reimplementation on the 2025 pileup JSON | 0.03% | 0.05% | 51.78 / 51.77 |
| brilcalc approximation (used for 2026), nominal | 3.1% | 7.3% | 51.78 / 51.77 |
| brilcalc approximation, up / down | 3.2% / 3.1% | 6.5% / 10.3% | 54.18 / 54.16, 49.39 / 49.38 |

The approximation error, about 3%, is small next to the pileup systematic. In 2026 the up/down
weights move by 38% / 59% (weighted mean |w/w_nom − 1|). A single median spread instead of the
quantile average gave 3.5% and a 32% maximum. The mean weight on Summer24 is 1.000 (nominal, up
and down).

## Inputs

- `inputs/brilcalc_byls_2026_golden403937_69200ub.csv.gz`: the brilcalc per-LS output used for 2026.
- `inputs/dataPileupHistogram-2025pp_Golden-*ub.root`: the central 2025 histograms used for the validation.
- Not vendored, because of their size:
  - the 2025 pileup JSON, `/eos/user/c/cmsdqm/www/CAF/certification/Collisions25/PileUp/pileup_JSON-2025_Golden.txt`
    (md5 28c42df1721b806f6d8e9c876eff462b);
  - the 2025 brilcalc per-LS csv, lxplus `~/work/ttbarhadronic/lumi_2026/bril_golden2025_byls.csv`
    (md5 bdd841b6721f93a1212c55ae00ac2db6).

## Use

`corrections.GetPUSF(events, "2026")` uses `2026_data_on_Summer24` from `puWeights.json.gz`. Replace
it with the central product once LUM publishes one for 2026 (also re-derive when the golden JSON
grows).
