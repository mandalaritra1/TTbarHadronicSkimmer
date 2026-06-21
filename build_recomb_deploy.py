#!/usr/bin/env python
"""Build the deployable GloParTv3-recombination tagger config for the analysis.

Combines the production per-pT logistic fit (logistic_weights.json) with per-pT
working-point thresholds derived at the **standard CMS AK8 top-tagger targeted QCD
mis-tag** for each WP name (consistent with the official nomenclature):

    verytight 0.1% | tight 0.5% | medium 1.0% | loose 2.5% | veryloose 5.0%

i.e. the recombined score is calibrated so that in EVERY pT bin the QCD mis-tag
equals the WP's target (flat vs pT). Thresholds are signal-independent (defined
purely on QCD). To run the analysis at 0.5% mis-tag, use --ttagWP tight.

Outputs (small, tracked, shipped with code):
  data/recomb/recomb_deploy_<iov>.json   — logistic params + per-pT WP thresholds
                                           (recomb tagger)
  data/recomb/baseline_deploy_<iov>.json — per-pT WP thresholds for the baseline
                                           TopvsQCD ratio, same target mis-tags, so
                                           baseline and recomb are directly
                                           comparable at each --ttagWP.
Loaded by TTbarResProcessor(topTagger='recomb'|'baseline', recomb_weights=...).

Run: .venv/bin/python build_recomb_deploy.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "python"))
import glopart_recomb as gr  # noqa: E402

IOV = "2024"
LOGISTIC_WEIGHTS = "outputs/glopart_recomb/prod/logistic_weights.json"
QCD_GLOB = "outputs/glopart_recomb/ntuples_night/ntuple_QCD_*.npz"
OUT = f"data/recomb/recomb_deploy_{IOV}.json"
OUT_BASE = f"data/recomb/baseline_deploy_{IOV}.json"

# Standard CMS AK8 top-tagger WP nomenclature -> targeted QCD mis-tag efficiency.
# The recomb per-pT thresholds are derived to hit these exactly, per pT bin.
WP_MISTAG = {
    "verytight": 0.001,
    "tight":     0.005,
    "medium":    0.010,
    "loose":     0.025,
    "veryloose": 0.050,
}


def main():
    lw = gr.load_logistic_weights(LOGISTIC_WEIGHTS)
    pt_edges = np.asarray(lw["pt_edges"], dtype=np.float64)
    params = lw["params_per_bin"]
    nbins = len(pt_edges) - 1

    # load full-stats QCD background (label 0), xs-weighted
    cols = {}
    files = sorted(glob.glob(QCD_GLOB))
    if not files:
        sys.exit(f"no QCD ntuples at {QCD_GLOB}")
    for f in files:
        d = np.load(f)
        for k in list(gr.RAW_HEADS) + ["pt", "weight"]:
            cols.setdefault(k, []).append(d[k])
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    heads = {h: cols[h] for h in gr.RAW_HEADS}
    pt, w = cols["pt"], cols["weight"]
    print(f"QCD background: {len(pt):,} jets from {len(files)} bins")

    # scores on the full QCD sample
    base = gr.baseline_topvsqcd(heads)
    Xeng = gr.build_features_eng(heads)
    recomb = np.zeros(len(pt))
    idx = gr.pt_bin_index(pt, pt_edges)
    for b in range(nbins):
        m = idx == b
        if np.any(m):
            recomb[m] = 1.0 / (1.0 + np.exp(-gr.apply_logistic(Xeng[m], params[b])))

    def derive_wp(score, label):
        """Per-WP, per-pT-bin threshold giving each WP's TARGET QCD mis-tag."""
        out = {}
        print(f"\n{'WP':>10} {'ptbin':>13} {'target':>7} {label:>12}")
        for name, target in WP_MISTAG.items():
            per_bin = {}
            for b in range(nbins):
                m = idx == b
                wb = w[m]
                if wb.sum() <= 0:
                    per_bin[str(b)] = 1.0
                    continue
                thr = gr.weighted_quantile(score[m], 1.0 - target, wb)
                per_bin[str(b)] = float(thr)
                print(f"{name:>10} {pt_edges[b]:5.0f}-{pt_edges[b+1]:<6.0f} "
                      f"{100*target:6.2f}% {thr:12.4f}")
            out[name] = per_bin
        return out

    note = ("per-pT thresholds at the standard CMS targeted QCD mis-tag "
            "(verytight 0.1% / tight 0.5% / medium 1.0% / loose 2.5% / veryloose 5.0%)")

    # --- recomb (logistic) deploy: per-pT logistic params + per-pT WP thresholds ---
    recomb_deploy = {
        "tagger": "glopart_recomb_logistic",
        "iov": IOV,
        "transform": "eng",
        "eps": gr.EPS,
        "feature_names": list(gr.ENG_FEATURE_NAMES),
        "pt_edges": [float(e) for e in pt_edges],
        "params": {
            str(b): {"mu": params[b]["mu"].tolist(),
                     "sd": params[b]["sd"].tolist(),
                     "beta": params[b]["beta"].tolist()}
            for b in params
        },
        "wp": derive_wp(recomb, "recomb thr"),
        "wp_mistag": WP_MISTAG,
        "provenance": {"logistic_weights": LOGISTIC_WEIGHTS, "qcd_glob": QCD_GLOB, "note": note},
    }
    # --- baseline (TopvsQCD ratio) deploy: per-pT WP thresholds only (no params) ---
    baseline_deploy = {
        "tagger": "glopart_TopvsQCD_baseline_ptbinned",
        "iov": IOV,
        "pt_edges": [float(e) for e in pt_edges],
        "wp": derive_wp(base, "base thr"),
        "wp_mistag": WP_MISTAG,
        "provenance": {"qcd_glob": QCD_GLOB, "note": note},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for path, deploy in [(OUT, recomb_deploy), (OUT_BASE, baseline_deploy)]:
        with open(path, "w") as f:
            json.dump(deploy, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
