#!/usr/bin/env python
"""Driver for the GloParTv3 per-pT recombination top-tag study (local, ntuple).

Reads the lightweight per-jet skims in ``data/skim/`` and, for each boosted pT
window, compares three top-taggers at a fixed **0.5% QCD mis-tag**:

  * baseline   : TopvsQCD = (TopbWqq+TopbWq)/(TopbWqq+TopbWq+QCD)  (to beat)
  * lda        : ntuple-free Gaussian LDA on the 7 log-score heads   (cross-check)
  * logistic   : class-balanced IRLS on the 9-D engineered log-scores (PRIMARY)

Everything is split by event parity: parity==0 jets FIT the recombination, the
held-out parity==1 jets are used for ALL reported metrics (no train/test leak).
Each QCD pT window uses its OWN pT-binned sample (the skim weights are 1, so
concatenating pT-binned QCD would mix cross-sections); signal = all ZPrime
masses concatenated and sliced by pT (a matched boosted top is signal regardless
of mediator mass).

Outputs (under ``outputs/glopart_recomb/``):
  * ``recomb_metrics.json``       — per-window sig-eff, ROC arrays, decorrelation
  * ``logistic_weights.json``     — per-pT logistic params (mu, sd, beta)
  * ``lda_weights.json``          — per-pT LDA (w, b)
and prints a summary table + the mass-decorrelation gate verdict.

Run (local skims, the validated verdict):
    .venv/bin/python run_glopart_recomb.py

Run (full-stats ntuples from run_toptag_wp.py --recomb-ntuple, finer pT bins):
    .venv/bin/python run_glopart_recomb.py --ntuple-dir outputs/glopart_recomb/ntuples_2024 \
        --pt-edges 800 1000 1200 1500 2000 3000 --out-dir outputs/glopart_recomb/full
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "python"))
import glopart_recomb as gr  # noqa: E402

SKIM_DIR = "data/skim"
OUT_DIR = "outputs/glopart_recomb"
TARGET_MISTAG = 0.005
FIT_PARITY, EVAL_PARITY = 0, 1

# pT window -> dedicated QCD pT-binned skim. Order defines the saved pt-bin index.
WINDOWS = [
    (800.0, 1000.0, "QCD_PT800to1000"),
    (1000.0, 1500.0, "QCD_PT1000to1500"),
    (1500.0, 2000.0, "QCD_PT1500to2000"),
]
PT_EDGES = [800.0, 1000.0, 1500.0, 2000.0]

# m_SD bin edges for the decorrelation gate (finer through the top-mass window).
MSD_EDGES = np.array([30, 60, 90, 105, 120, 135, 150, 165, 180, 200, 230, 270, 320],
                     dtype=np.float64)


def load_skims(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(SKIM_DIR, pattern))):
        ds = os.path.basename(f)[len("skim_"):-len(".npz")]
        d = np.load(f)
        out[ds] = {k: d[k] for k in d.files}
    return out


def subset(sample, mask):
    """Return a dict with every column of ``sample`` masked by ``mask``."""
    return {k: v[mask] for k, v in sample.items()}


def load_ntuples(ntuple_dir):
    """Load full-stats fit ntuples (from run_toptag_wp.py --recomb-ntuple).

    Splits every file by the ``label`` column and concatenates into one weighted
    signal (label==1: matched tops from TTbar/ZPrime) and one weighted background
    (label==0: inclusive QCD) sample. This is the xs-weighted counterpart of the
    per-window local skims, so QCD pT-binned samples combine into one mixture.
    """
    files = sorted(glob.glob(os.path.join(ntuple_dir, "ntuple_*.npz")))
    if not files:
        sys.exit(f"no ntuple_*.npz under {ntuple_dir}/ (run run_toptag_wp.py --recomb-ntuple)")
    sig_parts, bkg_parts = [], []
    for f in files:
        d = {k: v for k, v in np.load(f).items()}
        lab = d["label"]
        if np.any(lab == 1):
            sig_parts.append(subset(d, lab == 1))
        if np.any(lab == 0):
            bkg_parts.append(subset(d, lab == 0))

    def cat(parts):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None

    sig, bkg = cat(sig_parts), cat(bkg_parts)
    if sig is None or bkg is None:
        sys.exit("ntuples missing a signal (label==1) or background (label==0) class")
    print(f"loaded {len(files)} ntuple file(s): {len(sig['pt'])} signal jets, "
          f"{len(bkg['pt'])} QCD jets\n")
    return sig, bkg


def build_bins(args):
    """Return (bins, pt_edges, out_dir) where bins is a list of (lo, hi, sig, bkg).

    Two input modes:
      * default (local skims): each window uses its dedicated QCD pT-binned skim;
      * ``--ntuple-dir`` (full stats): one weighted QCD + one weighted signal
        sample, sliced into the (possibly finer) ``--pt-edges`` windows.
    """
    if args.ntuple_dir:
        sig_all, bkg_all = load_ntuples(args.ntuple_dir)
        edges = args.pt_edges
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            bins.append((lo, hi,
                         subset(sig_all, (sig_all["pt"] >= lo) & (sig_all["pt"] < hi)),
                         subset(bkg_all, (bkg_all["pt"] >= lo) & (bkg_all["pt"] < hi))))
        return bins, edges, args.out_dir

    qcd = load_skims("skim_QCD_*.npz")
    sigfiles = load_skims("skim_ZPrime*.npz")
    if not qcd or not sigfiles:
        sys.exit(f"no skims found under {SKIM_DIR}/ (have you run the skim step?)")
    cols = next(iter(sigfiles.values())).keys()
    sig_all = {k: np.concatenate([d[k] for d in sigfiles.values()]) for k in cols}
    print(f"loaded {len(qcd)} QCD samples, {len(sigfiles)} signal samples "
          f"({len(sig_all['pt'])} signal jets)\n")
    bins = []
    for lo, hi, qname in WINDOWS:
        bins.append((lo, hi,
                     subset(sig_all, (sig_all["pt"] >= lo) & (sig_all["pt"] < hi)),
                     subset(qcd[qname], (qcd[qname]["pt"] >= lo) & (qcd[qname]["pt"] < hi))))
    return bins, list(PT_EDGES), args.out_dir


def heads_dict(sample):
    """Just the 7 raw head columns (what the feature builders consume)."""
    return {h: sample[h] for h in gr.RAW_HEADS}


def fit_window(sig, bkg):
    """Fit all three taggers on parity==0 and score parity==1 jets.

    Returns a dict of held-out score arrays for sig/bkg plus the fitted objects.
    """
    sfit, seval = sig["parity"] == FIT_PARITY, sig["parity"] == EVAL_PARITY
    bfit, beval = bkg["parity"] == FIT_PARITY, bkg["parity"] == EVAL_PARITY
    sw, bw = sig["weight"], bkg["weight"]

    # --- baseline (no fit) ---
    base_s = gr.baseline_topvsqcd(heads_dict(sig))
    base_b = gr.baseline_topvsqcd(heads_dict(bkg))

    # --- LDA from 7-head log-score moments (parity-0 only) ---
    X7s = gr.build_features(heads_dict(sig), "logscore")
    X7b = gr.build_features(heads_dict(bkg), "logscore")
    w_lda, b_lda = gr.moments_to_lda(
        gr.moment_vec(X7s[sfit], sw[sfit]),
        gr.moment_vec(X7b[bfit], bw[bfit]),
    )
    lda_s = gr.apply_score(X7s, w_lda, b_lda)
    lda_b = gr.apply_score(X7b, w_lda, b_lda)

    # --- logistic on 9-D engineered log-scores (parity-0 only) ---
    Es = gr.build_features_eng(heads_dict(sig))
    Eb = gr.build_features_eng(heads_dict(bkg))
    Xfit = np.r_[Es[sfit], Eb[bfit]]
    yfit = np.r_[np.ones(sfit.sum()), np.zeros(bfit.sum())]
    wfit = np.r_[sw[sfit], bw[bfit]]
    params = gr.fit_logistic(Xfit, yfit, wfit)
    log_s = gr.apply_logistic(Es, params)
    log_b = gr.apply_logistic(Eb, params)

    return {
        "eval_sig": seval, "eval_bkg": beval, "sw": sw, "bw": bw,
        "scores": {
            "baseline": (base_s, base_b),
            "lda": (lda_s, lda_b),
            "logistic": (log_s, log_b),
        },
        "lda": (w_lda, b_lda),
        "logistic_params": params,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ntuple-dir", default=None,
                    help="fit from full-stats ntuples in this dir (run_toptag_wp.py "
                         "--recomb-ntuple output) instead of the local data/skim/*.npz")
    ap.add_argument("--pt-edges", type=float, nargs="+",
                    default=[800.0, 1000.0, 1500.0, 2000.0],
                    help="pT window edges for --ntuple-dir mode (finer bins need stats)")
    ap.add_argument("--out-dir", default=OUT_DIR, help="output dir for metrics + weights")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bins, pt_edges, out_dir = build_bins(args)

    metrics = {"target_mistag": TARGET_MISTAG, "fit_parity": FIT_PARITY,
               "eval_parity": EVAL_PARITY, "msd_edges": MSD_EDGES.tolist(),
               "pt_edges": list(pt_edges), "windows": []}
    lda_per_bin, log_per_bin = {}, {}
    methods = ["baseline", "lda", "logistic"]

    hdr = f"{'pT window':>14} | {'n_sig':>6} {'n_qcd':>6} | " + \
          " ".join(f"{m:>9}" for m in methods) + " |  decorr chi2/ndf (base/log)"
    print(hdr)
    print("-" * len(hdr))

    for pb, (lo, hi, sig, bkg) in enumerate(bins):
        if len(sig["pt"]) == 0 or len(bkg["pt"]) == 0:
            print(f"{lo:5.0f}-{hi:<6.0f}  | skipped (empty signal or background)")
            continue
        res = fit_window(sig, bkg)
        seval, beval, sw, bw = res["eval_sig"], res["eval_bkg"], res["sw"], res["bw"]

        wrow = {"lo": lo, "hi": hi, "center": 0.5 * (lo + hi),
                "n_sig_eval": int(seval.sum()), "n_bkg_eval": int(beval.sum()),
                "sigeff": {}, "threshold": {}, "roc": {}, "decorr": {}}

        msd_b = bkg["msd"][beval]
        for m in methods:
            ss, sb = res["scores"][m]
            ss_e, sb_e = ss[seval], sb[beval]
            sw_e, bw_e = sw[seval], bw[beval]
            eff, thr = gr.sigeff_at_mistag_unbinned(ss_e, sb_e, TARGET_MISTAG, sw_e, bw_e)
            fpr, tpr = gr.roc_unbinned(ss_e, sb_e, sw_e, bw_e)
            wrow["sigeff"][m] = eff
            wrow["threshold"][m] = thr
            wrow["roc"][m] = {"mistag": fpr.tolist(), "sigeff": tpr.tolist()}
            # decorrelation: QCD mis-tag vs m_SD at this method's 0.5% threshold
            cen, mis, err = gr.mistag_vs_value(msd_b, sb_e, thr, MSD_EDGES, bw_e)
            flat = gr.flatness(cen, mis, err)
            wrow["decorr"][m] = {"centers": cen.tolist(), "mistag": mis.tolist(),
                                 "err": err.tolist(), "flatness": flat}

        metrics["windows"].append(wrow)
        lda_per_bin[pb] = res["lda"]
        log_per_bin[pb] = res["logistic_params"]

        eff_str = " ".join(f"{wrow['sigeff'][m]:9.3f}" for m in methods)
        cb = wrow["decorr"]["baseline"]["flatness"]["chi2_per_ndf"]
        cl = wrow["decorr"]["logistic"]["flatness"]["chi2_per_ndf"]
        print(f"{lo:5.0f}-{hi:<6.0f}  | {wrow['n_sig_eval']:6d} {wrow['n_bkg_eval']:6d} | "
              f"{eff_str} |  {cb:6.1f} / {cl:6.1f}")

    # --- persist ---
    mpath = os.path.join(out_dir, "recomb_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    gr.save_logistic_weights(os.path.join(out_dir, "logistic_weights.json"),
                             pt_edges, log_per_bin,
                             metadata={"target_mistag": TARGET_MISTAG})
    gr.save_weights(os.path.join(out_dir, "lda_weights.json"), pt_edges, lda_per_bin,
                    metadata={"target_mistag": TARGET_MISTAG})

    # --- summary ---
    print("\n=== summary (held-out sig-eff @ %.1f%% QCD mis-tag) ===" % (100 * TARGET_MISTAG))
    gains = []
    for w in metrics["windows"]:
        g = w["sigeff"]["logistic"] - w["sigeff"]["baseline"]
        gains.append(g)
        print(f"  {w['lo']:.0f}-{w['hi']:.0f} GeV: baseline {w['sigeff']['baseline']:.3f}"
              f"  logistic {w['sigeff']['logistic']:.3f}  (logistic {g:+.3f})")
    print(f"  mean logistic gain over baseline: {np.mean(gains):+.3f}")

    print("\n=== mass-decorrelation gate (mistag-vs-m_SD flatness, chi2/ndf) ===")
    print("  lower / closer-to-baseline = score does NOT sculpt the QCD mass peak")
    ok = True
    for w in metrics["windows"]:
        cb = w["decorr"]["baseline"]["flatness"]["chi2_per_ndf"]
        cl = w["decorr"]["logistic"]["flatness"]["chi2_per_ndf"]
        sb = w["decorr"]["baseline"]["flatness"]["slope"]
        sl = w["decorr"]["logistic"]["flatness"]["slope"]
        verdict = "OK" if cl <= 3.0 * max(cb, 1.0) else "CHECK — possible sculpting"
        if "CHECK" in verdict:
            ok = False
        print(f"  {w['lo']:.0f}-{w['hi']:.0f} GeV: baseline chi2/ndf {cb:6.1f} "
              f"(slope {sb:+.1e}) | logistic {cl:6.1f} (slope {sl:+.1e})  -> {verdict}")
    print(f"\n  GATE: {'PASS' if ok else 'NEEDS REVIEW'}")
    print(f"\nwrote {mpath} and weights to {out_dir}/")


if __name__ == "__main__":
    main()
