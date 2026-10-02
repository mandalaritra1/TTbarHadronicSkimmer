#!/usr/bin/env python3
"""Validate the Summer24 pileup-weight method against the central LUM weights.

The method (scripts/derive_pu_summer24_to_2022_2023.py) is w(n) = data(n) / MC_Summer24(n) with
the data profile from the certification pileup histograms and the Summer24 profile measured
from NanoAOD. Two central products use the same MC: 2024 data on Summer24 (LUM
Run3-24CDEReprocessingFGHIPrompt-Summer24, puWeights_CDEFGHI) and 2025 data on Summer24 (LUM
Run3-25Prompt-Summer24, puWeights_2025pp_Golden_Summer24_25ns_69200ub). Rebuilding both with the
method and comparing bin by bin (weighted by the Summer24 profile) tests the method.
"""
import argparse
import json
import sys

import correctionlib
import numpy as np
import uproot

NBINS = 99
X = np.arange(NBINS) + 0.5


def read_hist(path):
    f = uproot.open(path)
    key = [k for k, c in f.classnames().items() if c.startswith("TH1")][0]
    v = f[key].values()
    out = np.zeros(NBINS)
    out[: min(NBINS, len(v))] = v[:NBINS]
    return out / out.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="derivation_summary.json (measured Summer24 profile)")
    ap.add_argument("--cases", nargs="+", required=True,
                    help="label:data_hist_prefix:central_json:correction_name (prefix + {66000,69200,72400}ub.root)")
    args = ap.parse_args()
    mc = np.array(json.load(open(args.summary))["summer24_profile_measured"])
    for case in args.cases:
        label, prefix, cjson, cname = case.split(":")
        cen = correctionlib.CorrectionSet.from_file(cjson)[cname]
        print(f"== {label}  (central {cname})")
        for var, xs in (("nominal", 69200), ("up", 72400), ("down", 66000)):
            d = read_hist(f"{prefix}{xs}ub.root")
            mine = np.where(mc > 0, d / np.where(mc > 0, mc, 1), 1.0)
            c = cen.evaluate(X, var)
            sel = mc > 1e-3                       # bins holding 99.x% of Summer24
            r = mine[sel] / c[sel]
            wdev = np.sum(mc[sel] * np.abs(r - 1)) / np.sum(mc[sel])
            print(f"  {var:7s} mean weight on Summer24: mine {np.sum(mine * mc):.4f}  central {np.sum(c * mc):.4f} | "
                  f"mine/central in the core: weighted mean |dev| {wdev:.3f}, max {np.max(np.abs(r - 1)):.3f} "
                  f"(n = {int(X[sel][np.argmax(np.abs(r - 1))])})")
        nom = cen.evaluate(X, "nominal")
        d = read_hist(f"{prefix}69200ub.root")
        mine = np.where(mc > 0, d / np.where(mc > 0, mc, 1), 1.0)
        print("   n:      " + " ".join(f"{int(n):6d}" for n in (10, 20, 30, 40, 45, 50, 60, 70)))
        print("   mine:   " + " ".join(f"{mine[n]:6.3f}" for n in (10, 20, 30, 40, 45, 50, 60, 70)))
        print("   central:" + " ".join(f"{nom[n]:6.3f}" for n in (10, 20, 30, 40, 45, 50, 60, 70)))


if __name__ == "__main__":
    sys.exit(main())
