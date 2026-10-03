#!/usr/bin/env python3
"""Merge outputs of one MC sample that was run as file splits into the output of a single run.

Each part was normalised in postprocess with its own lumi*xsec/sumw_i and its own theory
normalisation (q2/pdf/isr/fsr slices x sumw_nominal_i/sumw_var_i). Adding the parts is wrong;
this script undoes each part's normalisation, adds the raw histograms and counters, and
normalises once with the global sumw and theory sums, which is what one run over all files
gives (coffea chunks each file on its own, so the splits see the same chunks).

Single-sample outputs only (not the grouped signal layout: use merge_signal_masses.py there).

    python scripts/merge_split_mc.py -o TTbar_2026_inclusive_topvsqcd.coffea parts/p*/TTbar_2026_inclusive_topvsqcd.coffea
"""
import argparse
import copy
import sys

import hist
import numpy as np
from coffea.processor import defaultdict_accumulator
from coffea.util import load, save

SUMMED = ("cutflow", "cutflow_unweighted", "cutflow_weighted", "cutflow_weighted2", "weights", "systematics",
          "truthstudy", "sumw_by_dataset", "sumw2_by_dataset", "sumw_theory", "flat_theory_variations")
REBUILT = ("normalization", "theory_norm", "cutflow_scaled", "cutflow_weighted_scaled", "cutflow_weighted2_scaled")
STATIC = ("cutflow_table_steps", "analysisCategories", "sample_metadata")


def _scale_slices(h, factors, inverse=False):
    """Multiply (or divide) the systematic=<syst> slice of a weight-storage hist by factors[syst]."""
    if "systematic" not in h.axes.name:
        return
    isys = list(h.axes.name).index("systematic")
    view = h.view(flow=True)
    for syst, f in factors.items():
        if syst not in list(h.axes["systematic"]):
            continue
        idx = [slice(None)] * len(h.axes)
        idx[isys] = h.axes["systematic"].index(syst)
        f = 1.0 / f if inverse else f
        view["value"][tuple(idx)] *= f
        view["variance"][tuple(idx)] *= f * f


def _theory_factors(sums):
    nominal = float(sums.get("all|nominal", 0.0))
    out = {}
    for syst in (f"{n}{d}" for n in ("q2", "pdf", "isr", "fsr") for d in ("Up", "Down")):
        varied = float(sums.get(f"all|{syst}", 0.0))
        if nominal and varied:
            out[syst] = nominal / varied
    return out


def merge(parts):
    first = parts[0]
    if "ntuple" in first:
        raise SystemExit("ntuple outputs are not supported")
    shas = {p["provenance"].get("git_sha") for p in parts}
    if len(shas) != 1:
        raise SystemExit(f"parts come from different commits: {shas}")
    norms = [p["normalization"] for p in parts]
    for k in ("sample", "subsample", "year", "xsec_pb", "lumi_pb", "applied"):
        if len({str(n.get(k)) for n in norms}) != 1:
            raise SystemExit(f"parts disagree on normalization '{k}': {[n.get(k) for n in norms]}")
    if not first["normalization"].get("applied"):
        raise SystemExit("parts were not normalised (data?): nothing to undo; add them instead")
    for k in STATIC:
        if any(p[k] != first[k] for p in parts[1:]):
            raise SystemExit(f"'{k}' differs between parts")

    merged = {}
    for key, val in first.items():
        if isinstance(val, hist.Hist):
            acc = None
            for p in parts:
                h = p[key].copy()
                _scale_slices(h, p["theory_norm"], inverse=True)
                h = h * (1.0 / p["normalization"]["scale_factor"])
                acc = h if acc is None else acc + h
            merged[key] = acc
        elif key in SUMMED:
            acc = copy.deepcopy(val)
            for p in parts[1:]:
                acc = acc + p[key]
            merged[key] = acc
        elif key in STATIC:
            merged[key] = val
        elif key == "provenance":
            merged[key] = dict(val, merged_from=len(parts))
        elif key in REBUILT:
            continue
        else:
            raise SystemExit(f"don't know how to merge '{key}' ({type(val).__name__})")

    cut = merged["cutflow"]
    sumw, sumw2 = float(cut["sumw"]), float(cut["sumw2"])
    n0 = first["normalization"]
    s = n0["lumi_pb"] * float(n0["xsec_pb"]) / sumw
    theory = _theory_factors(merged["sumw_theory"])
    for key, val in merged.items():
        if isinstance(val, hist.Hist):
            merged[key] = val * s
            _scale_slices(merged[key], theory)
    merged["normalization"] = dict(n0, sumw_raw=sumw, sumw2_raw=sumw2, scale_factor=s, merged_from=len(parts))
    merged["theory_norm"] = theory
    merged["cutflow_scaled"] = {k: float(v) * (s * s if k == "sumw2" else s) for k, v in cut.items()}
    merged["cutflow_weighted_scaled"] = {k: float(v) * s for k, v in merged.get("cutflow_weighted", {}).items()}
    merged["cutflow_weighted2_scaled"] = {k: float(v) * s * s for k, v in merged.get("cutflow_weighted2", {}).items()}
    missing = set(first) - set(merged)
    if missing:
        raise SystemExit(f"keys lost in the merge: {sorted(missing)}")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    merged = merge([load(f) for f in args.inputs])
    save(merged, args.output)
    n = merged["normalization"]
    print(f"wrote {args.output}: {len(args.inputs)} parts, sumw {n['sumw_raw']:.6g}, scale {n['scale_factor']:.6g}")


if __name__ == "__main__":
    sys.exit(main())
