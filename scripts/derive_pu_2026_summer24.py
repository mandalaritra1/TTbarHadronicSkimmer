#!/usr/bin/env python3
"""Pileup weights for Summer24 MC reweighted to 2026 data (no central 2026 product exists yet).

The central data pileup histograms come from pileupCalc ("true" mode) run on a per-lumisection
pileup JSON ([ls, recorded lumi, rms bunch lumi, mean bunch lumi]). For 2026 there is neither
the histogram nor the pileup JSON, so the data profile is built from brilcalc instead:

  mu(LS)    = k * brilcalc --byls --minBiasXsec 69200 avgpu (lumi-weighted over bunches), with
              k = <mu> of the 2025 pileup JSON / <mu> of brilcalc on 2025 (1.0049: normtag_BRIL
              runs 0.5% low against the pileup JSON)
  sigma(LS) = r * mu(LS), averaged over r = the 10/30/50/70/90% quantiles of the lumi-weighted
              bunch-to-bunch spread rms/mean in the 2025 pileup JSON

and each lumisection adds a Gaussian N(mu, sigma) weighted by its recorded lumi, integrated over
the 99 unit bins, exactly as pileupCalc does. Up/down use 72.4/66.0 mb (mu and sigma scale with
the cross section).

Validation, both on 2025 where the central histogram exists (dataPileupHistogram-2025pp_Golden):
  1. the pileupCalc reimplementation on the 2025 pileup JSON against the central histogram;
  2. the brilcalc approximation for 2025 against the central histogram.
Both are compared as weights on the Summer24 profile, the quantity that is applied.

MC = the Summer24 true-pileup profile measured from NanoAOD (derivation_summary.json of the
2022/2023 derivation). Bins without MC get weight 1.

  python scripts/derive_pu_2026_summer24.py --inputs <dir> \
      --out data/corrections/puWeights/Summer24_2026/puWeights.json.gz
"""
import argparse
import gzip
import json
import os
import sys
from math import erf, sqrt

import correctionlib.schemav2 as cs
import numpy as np
import uproot

NBINS = 99
XS = {"nominal": 69200., "up": 72400., "down": 66000.}
EDGES = np.arange(NBINS + 1.0)
CORR_NAME = "2026_data_on_Summer24"
_erf = np.vectorize(erf)


def gauss_bins(mu, sigma):
    """pileupCalc 'true' mode: Gaussian area per unit bin, normalised inside [0, 99]."""
    if sigma <= 0:
        p = np.zeros(NBINS)
        p[min(int(mu), NBINS - 1)] = 1.0
        return p
    cdf = 0.5 * (1 + _erf((EDGES - mu) / (sqrt(2) * sigma)))
    p = np.diff(cdf)
    s = p.sum()
    return p / s if s > 0 else p


def profile_from_pileup_json(path, golden, xs):
    pj = json.load(open(path))
    h = np.zeros(NBINS)
    for run, lss in pj.items():
        good = golden.get(int(run))
        if good is None:
            continue
        for ls, lumi, rms, avg in lss:
            if int(ls) in good and lumi > 0:
                h += lumi * gauss_bins(avg * xs, rms * xs)
    return h / h.sum()


def rel_spread(path, golden, qs=(0.1, 0.3, 0.5, 0.7, 0.9)):
    pj = json.load(open(path))
    r, w = [], []
    for run, lss in pj.items():
        good = golden.get(int(run))
        if good is None:
            continue
        for ls, lumi, rms, avg in lss:
            if int(ls) in good and lumi > 0 and avg > 0:
                r.append(rms / avg)
                w.append(lumi)
    r, w = np.array(r), np.array(w)
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(r[o][np.searchsorted(cw, q)]) for q in qs]


def read_byls(path):
    """brilcalc --byls csv: run:fill, ls:ls, time, beamstatus, E, delivered, recorded, avgpu, source."""
    mu, lumi = [], []
    for line in open(path):
        p = line.strip().split(",")
        if line.startswith("#") or len(p) < 9 or ":" not in p[0]:
            continue
        try:
            rec, pu = float(p[6]), float(p[7])
        except ValueError:
            continue
        if rec > 0 and 0 < pu < 120:   # a few 2024-style unphysical avgpu values are dropped
            mu.append(pu)
            lumi.append(rec)
    return np.array(mu), np.array(lumi)


def profile_from_byls(mu69, lumi, rq, xs, k=1.0):
    h = np.zeros(NBINS)
    scale = k * xs / 69200.
    for m, l in zip(mu69, lumi):
        h += l * np.mean([gauss_bins(m * scale, r * m * scale) for r in rq], axis=0)
    return h / h.sum()


def read_hist(path):
    f = uproot.open(path)
    key = [k for k, c in f.classnames().items() if c.startswith("TH1")][0]
    v = f[key].values()
    out = np.zeros(NBINS)
    out[: min(NBINS, len(v))] = v[:NBINS]
    return out / out.sum()


def golden_sets(path):
    return {int(r): set(l for a, b in v for l in range(a, b + 1)) for r, v in json.load(open(path)).items()}


def weights(d, mc):
    return np.where(mc > 0, d / np.where(mc > 0, mc, 1), 1.0)


def compare(label, mine, ref, mc):
    wm, wr = weights(mine, mc), weights(ref, mc)
    core = mc > 1e-3
    r = wm[core] / wr[core]
    out = {"mean_w_mine": round(float(np.sum(wm * mc)), 4), "mean_w_ref": round(float(np.sum(wr * mc)), 4),
           "weighted_mean_abs_dev": round(float(np.sum(mc[core] * np.abs(r - 1)) / np.sum(mc[core])), 4),
           "max_abs_dev": round(float(np.max(np.abs(r - 1))), 4),
           "max_dev_at_n": int(np.arange(NBINS)[core][np.argmax(np.abs(r - 1))]),
           "mean_pu_mine": round(float(np.sum(mine * (EDGES[:-1] + 0.5))), 2),
           "mean_pu_ref": round(float(np.sum(ref * (EDGES[:-1] + 0.5))), 2)}
    print(f"{label}: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="dir with the 2025 pileup JSON, 2025 central histograms, brilcalc csvs")
    ap.add_argument("--golden-2025", default="data/corrections/goldenJsons/Cert_Collisions2025_391658_398903_Golden.json")
    ap.add_argument("--golden-2026", default="data/corrections/goldenJsons/Cert_Collisions2026_401624_403937_golden.json")
    ap.add_argument("--summary", default="data/corrections/puWeights/Summer24_to_2022_2023/derivation_summary.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    I = lambda f: os.path.join(args.inputs, f)

    mc = np.array(json.load(open(args.summary))["summer24_profile_measured"])
    g25, g26 = golden_sets(args.golden_2025), golden_sets(args.golden_2026)
    rq = rel_spread(I("pileup_JSON-2025_Golden.txt"), g25)
    print(f"2025 bunch spread rms/mean, 10/30/50/70/90% quantiles: {[round(x, 4) for x in rq]}")

    val = {}
    mu25, l25 = read_byls(I("bril_golden2025_byls.csv"))
    mu26, l26 = read_byls(I("bril_golden403937_byls.csv"))
    print(f"brilcalc: 2025 {l25.sum()/1e9:.2f}/fb <mu> {np.average(mu25, weights=l25):.2f}; "
          f"2026 {l26.sum()/1e9:.2f}/fb <mu> {np.average(mu26, weights=l26):.2f}")
    exact_nom = profile_from_pileup_json(I("pileup_JSON-2025_Golden.txt"), g25, XS["nominal"])
    k = float(np.sum(exact_nom * (EDGES[:-1] + 0.5)) / np.average(mu25, weights=l25))
    print(f"mu scale k (2025 pileup JSON / brilcalc) = {k:.4f}")
    content = {}
    for var, xs in XS.items():
        ref = read_hist(I(f"dataPileupHistogram-2025pp_Golden-{int(xs)}ub.root"))
        exact = profile_from_pileup_json(I("pileup_JSON-2025_Golden.txt"), g25, xs)
        approx = profile_from_byls(mu25, l25, rq, xs, k)
        val[f"2025_{var}_pileupcalc_reimpl_vs_central"] = compare(f"2025 {var} pileupCalc reimplementation vs central", exact, ref, mc)
        val[f"2025_{var}_brilcalc_approx_vs_central"] = compare(f"2025 {var} brilcalc approximation vs central", approx, ref, mc)
        d26 = profile_from_byls(mu26, l26, rq, xs, k)
        content[var] = weights(d26, mc)
        print(f"2026 {var}: mean weight on Summer24 {np.sum(content[var] * mc):.4f}, data <PU> {np.sum(d26 * (EDGES[:-1] + 0.5)):.2f}")
    core = mc > 1e-3
    for var in ("up", "down"):
        dev = np.sum(mc[core] * np.abs(content[var][core] / content["nominal"][core] - 1)) / np.sum(mc[core])
        print(f"2026 {var} vs nominal (the pileup systematic): weighted mean |w/w_nom - 1| = {dev:.3f}")

    corr = cs.Correction(
        name=CORR_NAME,
        description=("Pileup weight for Summer24 MC reweighted to 2026 data (golden JSON "
                     "Cert_Collisions2026_401624_403937, eras B+D). No central 2026 product: the data profile is "
                     "pileupCalc-style ('true' mode) from brilcalc per-LS avgpu (normtag_BRIL) with a Gaussian bunch "
                     f"spread (2025 pileup JSON quantiles {[round(x, 3) for x in rq]}) and mu scale k = {k:.4f}; 69.2/72.4/66.0 mb. Validated "
                     "on 2025 against the central histogram. Divided by the measured Summer24 profile. "
                     "Derived by scripts/derive_pu_2026_summer24.py."),
        version=1,
        inputs=[cs.Variable(name="NumTrueInteractions", type="real", description="Pileup_nTrueInt"),
                cs.Variable(name="weights", type="string", description="nominal, up, or down")],
        output=cs.Variable(name="weight", type="real"),
        data=cs.Category(nodetype="category", input="weights", content=[
            cs.CategoryItem(key=var, value=cs.Binning(
                nodetype="binning", input="NumTrueInteractions", edges=list(EDGES),
                content=[float(x) for x in content[var]], flow="clamp"))
            for var in ("nominal", "up", "down")]),
    )
    cset = cs.CorrectionSet(schema_version=2, description="Pileup weights: 2026 data on Summer24 MC.", corrections=[corr])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        f.write(cset.model_dump_json(exclude_unset=True))
    json.dump({"bunch_spread_rel_2025_quantiles_10_30_50_70_90": rq, "mu_scale_k": k,
               "mean_weight_on_summer24": {v: round(float(np.sum(content[v] * mc)), 4) for v in content},
               "validation_2025": val},
              open(os.path.join(os.path.dirname(args.out), "derivation_summary.json"), "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
