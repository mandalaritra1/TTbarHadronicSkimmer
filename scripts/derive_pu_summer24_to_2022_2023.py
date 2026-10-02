#!/usr/bin/env python3
"""Pileup weights for Summer24 MC reweighted to 2022/2023 data.

The 2022/2023 Z' signal is the 2024 Summer24 sample standing in (no NanoAODv15 signal for
those years). The central LUM POG weights for 2022/2023 are data22/23 / MC22/23 and do not
preserve the normalization on Summer24 MC (mean 0.72-1.40). This script builds
data_IOV(n) / MC_Summer24(n) from central inputs:

- data: the DQM/LUM certification pileup histograms (golden JSON, 69.2 mb nominal, 72.4 mb
  up, 66.0 mb down, 99 bins), https://cms-service-dqmdc.web.cern.ch/CAF/certification/
  Collisions22/PileUp/{BCD,EFG}, Collisions23/PileUp/{BC,D};
- MC: the Summer24 true-pileup profile, measured from Summer24 NanoAOD (Pileup_nTrueInt of
  the Z' and TTbar samples) and cross-checked against the profile implied by the central 2024
  weights (data24 / w24, LUM Run3-24CDEReprocessingFGHIPrompt-Summer24 puWeights_CDEFGHI).

Writes a correctionlib JSON with one correction per IOV, same schema as the LUM files
(inputs NumTrueInteractions, weights in {nominal, up, down}).

Usage (LPC container, needs xrootd):
  python scripts/derive_pu_summer24_to_2022_2023.py --data-dir pu_derive --out data/corrections/puWeights/Summer24_to_2022_2023/puWeights.json.gz
"""
import argparse
import gzip
import json
import os
import sys

import correctionlib
import correctionlib.schemav2 as cs
import numpy as np
import uproot

NBINS = 99
ERAS = {
    "2022preEE": "Cert_Collisions2022_355100_357900_eraBCD_GoldenJson",
    "2022postEE": "Cert_Collisions2022_359022_362760_eraEFG_GoldenJson",
    "2023preBPix": "Cert_Collisions2023_366403_369802_eraBC_GoldenJson",
    "2023postBPix": "Cert_Collisions2023_369803_370790_eraD_GoldenJson",
}
XS = {"nominal": 69200, "up": 72400, "down": 66000}
CORR_NAME = "{iov}_data_on_Summer24"


def read_hist(path):
    f = uproot.open(path)
    key = [k for k, c in f.classnames().items() if c.startswith("TH1")][0]
    h = f[key]
    vals, edges = h.values(), h.axis().edges()
    out = np.zeros(NBINS)
    n = min(NBINS, len(vals))
    out[:n] = vals[:n]
    assert np.allclose(edges[: n + 1], np.arange(n + 1)), f"unexpected binning in {path}"
    return out


def mc_profile_measured(files, max_events):
    counts = np.zeros(NBINS)
    total = 0
    for fn in files:
        try:
            arr = uproot.open(fn, timeout=120)["Events"]["Pileup_nTrueInt"].array(library="np", entry_stop=max_events)
        except Exception as e:  # a bad replica must not stop the derivation
            print("  skip", fn.split("/")[-1], type(e).__name__)
            continue
        counts += np.histogram(arr, bins=NBINS, range=(0, NBINS))[0]
        total += len(arr)
    return counts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--files-per-sample", type=int, default=3)
    ap.add_argument("--max-events", type=int, default=200000)
    args = ap.parse_args()

    # --- Summer24 MC profile, measured ---
    red = "root://cmsxrootd.fnal.gov/"
    files = []
    for js, keys in (("ZPrime10.json", ("1000", "3000", "5000")), ("ZPrime30.json", ("2000",)),
                     ("ZPrime1.json", ("4000",)), ("TTbar.json", ("inclusive",))):
        m = json.load(open(os.path.join(args.repo, "data/nanoAOD", js)))["2024"]
        for k in keys:
            key = [x for x in m if x == k or x.startswith(k)][0]
            v = m[key]
            lst = v if isinstance(v, list) else v["files"]
            files += [red + f for f in lst[: args.files_per_sample]]
    mc_meas, n_mc = mc_profile_measured(files, args.max_events)
    print(f"Summer24 measured profile: {n_mc} events from {len(files)} files, mean nTrueInt "
          f"{np.average(np.arange(NBINS) + 0.5, weights=mc_meas):.2f}")

    # --- Summer24 MC profile implied by the central 2024 weights ---
    lum24 = correctionlib.CorrectionSet.from_file(
        os.path.join(args.repo, "data/corrections/puWeights/2024_Summer24/puWeights.json.gz"))
    w24 = lum24["Collisions24_CDEFGHI_goldenJSON"].evaluate(np.arange(NBINS) + 0.5, "nominal")
    d24 = read_hist(os.path.join(args.data_dir, "dataPileupHistogram-2024CDEFGHI_Golden-69200ub.root"))
    mc_lum = np.where(w24 > 0, d24 / np.where(w24 > 0, w24, 1), 0.0)
    pm, pl = mc_meas / mc_meas.sum(), mc_lum / mc_lum.sum()
    core = (pm > 1e-3) & (pl > 1e-3)
    print(f"measured vs LUM-implied Summer24 profile: max |ratio-1| in the core = "
          f"{np.max(np.abs(pm[core] / pl[core] - 1)):.3f}; mean nTrueInt LUM-implied "
          f"{np.average(np.arange(NBINS) + 0.5, weights=pl):.2f}")

    # Use the measured profile (it is what the reweighted events actually have); bins without
    # MC get weight 1 (no events there).
    mc = pm
    corrections, summary = [], {}
    for iov, stem in ERAS.items():
        content = {}
        for var, xs in XS.items():
            d = read_hist(os.path.join(args.data_dir, f"pileupHistogram-{stem}-13p6TeV-{xs}ub-{NBINS}bins.root"))
            d = d / d.sum()
            w = np.where(mc > 0, d / np.where(mc > 0, mc, 1), 1.0)
            content[var] = w
        mean_nom = float(np.sum(content["nominal"] * mc))
        summary[iov] = {v: round(float(np.sum(content[v] * mc)), 4) for v in content}
        print(f"{iov}: mean weight on Summer24 nominal/up/down = {summary[iov]}, "
              f"data mean nTrueInt {np.average(np.arange(NBINS) + 0.5, weights=d):.1f}")
        assert abs(mean_nom - 1) < 0.01
        corrections.append(cs.Correction(
            name=CORR_NAME.format(iov=iov),
            description=(f"Pileup weight for Summer24 MC reweighted to {iov} data ({stem}, golden JSON, "
                         f"69.2/72.4/66.0 mb for nominal/up/down) divided by the Summer24 true-pileup profile "
                         f"measured from {n_mc} Summer24 NanoAOD events. Derived by "
                         f"scripts/derive_pu_summer24_to_2022_2023.py."),
            version=1,
            inputs=[cs.Variable(name="NumTrueInteractions", type="real", description="Pileup_nTrueInt"),
                    cs.Variable(name="weights", type="string", description="nominal, up, or down")],
            output=cs.Variable(name="weight", type="real"),
            data=cs.Category(nodetype="category", input="weights", content=[
                cs.CategoryItem(key=var, value=cs.Binning(
                    nodetype="binning", input="NumTrueInteractions", edges=list(np.arange(NBINS + 1.0)),
                    content=[float(x) for x in content[var]], flow="clamp"))
                for var in ("nominal", "up", "down")]),
        ))
    cset = cs.CorrectionSet(schema_version=2, description=(
        "Pileup weights: 2022/2023 data on Summer24 MC (the 2024 Z' signal standing in for 2022/2023)."),
        corrections=corrections)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        f.write(cset.model_dump_json(exclude_unset=True))
    json.dump({"summer24_profile_measured": [float(x) for x in pm], "n_events": int(n_mc),
               "mean_weight_on_summer24": summary},
              open(os.path.join(os.path.dirname(args.out), "derivation_summary.json"), "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
