#!/usr/bin/env python
"""Plots for the GloParTv3 per-pT recombination study.

Consumes ``outputs/glopart_recomb/recomb_metrics.json`` (written by
``run_glopart_recomb.py``) and produces three figures:

  1. money plot   : signal eff @ 0.5% QCD mis-tag vs AK8 pT (baseline/LDA/logistic)
  2. ROC overlay  : signal-top eff vs QCD mis-tag, baseline vs logistic, per window
  3. decorr gate  : QCD mis-tag vs jet m_SD at the 0.5% WP (baseline vs logistic)

Figures are written to ``plots/images/glopart_recomb/`` and, if the
research-notes vault exists, mirrored into its ``attachments/`` so the findings
note embeds live PNGs.

Run:  .venv/bin/python plot_glopart_recomb.py
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except Exception:
    pass

METRICS = "outputs/glopart_recomb/recomb_metrics.json"
OUT_DIR = "plots/images/glopart_recomb"
VAULT_ATT = os.path.expanduser("~/Projects/research-notes/attachments")
MIRROR_VAULT = False  # set by --vault; off by default so quick checks don't clobber the vault

METHODS = {"baseline": ("Baseline TopvsQCD", "o-", "C0"),
           "lda": ("LDA (log-score)", "s--", "C1"),
           "logistic": ("Logistic (recomb)", "D-", "C2")}


def savefig(fig, name):
    """Save under plots/images and mirror into the vault attachments if present."""
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = [os.path.join(OUT_DIR, name)]
    if MIRROR_VAULT and os.path.isdir(VAULT_ATT):
        paths.append(os.path.join(VAULT_ATT, name))
    for p in paths:
        fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return paths


def money_plot(M):
    centers = [w["center"] for w in M["windows"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    for key, (lab, st, c) in METHODS.items():
        y = [w["sigeff"][key] for w in M["windows"]]
        ax.plot(centers, y, st, color=c, label=lab, lw=2, ms=10)
    base = [w["sigeff"]["baseline"] for w in M["windows"]]
    logi = [w["sigeff"]["logistic"] for w in M["windows"]]
    for x, b, l in zip(centers, base, logi):
        ax.annotate(f"+{100*(l-b):.1f}%", (x, l), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=13, color="C2")
    ax.set_xlabel(r"AK8 $p_T$ [GeV]")
    ax.set_ylabel(f"Signal eff @ {100*M['target_mistag']:.1f}% QCD mis-tag")
    ax.set_title("GloParTv3 per-$p_T$ recombination (2024)", fontsize=15)
    ax.legend(fontsize=13)
    ax.grid(alpha=0.3)
    return savefig(fig, "glopart_recomb_sigeff_vs_pt.png")


def roc_plot(M):
    fig, axes = plt.subplots(1, len(M["windows"]), figsize=(6 * len(M["windows"]), 6),
                             sharey=True)
    if len(M["windows"]) == 1:
        axes = [axes]
    for ax, w in zip(axes, M["windows"]):
        for key in ("baseline", "logistic"):
            lab, _, c = METHODS[key]
            r = w["roc"][key]
            ax.plot(r["mistag"], r["sigeff"], "-", color=c, lw=2, label=lab)
        ax.axvline(M["target_mistag"], color="grey", ls=":", lw=1.5,
                   label=f"{100*M['target_mistag']:.1f}% mis-tag")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("QCD mis-tag eff")
        ax.set_title(rf"${w['lo']:.0f}<p_T<{w['hi']:.0f}$ GeV", fontsize=13)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Signal (top) eff")
    axes[0].legend(fontsize=11, loc="lower right")
    fig.suptitle("ROC: baseline vs logistic recombination (2024)", fontsize=15)
    fig.tight_layout()
    return savefig(fig, "glopart_recomb_roc_by_pt.png")


def decorr_plot(M):
    fig, axes = plt.subplots(1, len(M["windows"]), figsize=(6 * len(M["windows"]), 6),
                             sharey=True)
    if len(M["windows"]) == 1:
        axes = [axes]
    for ax, w in zip(axes, M["windows"]):
        for key in ("baseline", "logistic"):
            lab, _, c = METHODS[key]
            d = w["decorr"][key]
            cen = np.array(d["centers"])
            mis = np.array(d["mistag"])
            err = np.array(d["err"])
            chi2 = d["flatness"]["chi2_per_ndf"]
            ax.errorbar(cen, 100 * mis, yerr=100 * err, fmt="o-", color=c, lw=1.8,
                        ms=6, capsize=2, label=f"{lab} ($\\chi^2$/ndf {chi2:.1f})")
        ax.axhline(100 * M["target_mistag"], color="grey", ls=":", lw=1.5)
        ax.set_xlabel(r"jet $m_{SD}$ [GeV]")
        ax.set_title(rf"${w['lo']:.0f}<p_T<{w['hi']:.0f}$ GeV", fontsize=13)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("QCD mis-tag [%] at 0.5% WP")
    axes[0].legend(fontsize=10, loc="upper right")
    fig.suptitle("Mass-decorrelation gate: QCD mis-tag vs $m_{SD}$ (2024)", fontsize=15)
    fig.tight_layout()
    return savefig(fig, "glopart_recomb_decorr_vs_msd.png")


def main():
    global METRICS, OUT_DIR, MIRROR_VAULT
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=METRICS, help="recomb_metrics.json to plot")
    ap.add_argument("--out-dir", default=OUT_DIR, help="dir for the PNGs")
    ap.add_argument("--vault", action="store_true", help="also mirror PNGs into research-notes/attachments")
    args = ap.parse_args()
    METRICS, OUT_DIR, MIRROR_VAULT = args.metrics, args.out_dir, args.vault
    if not os.path.exists(METRICS):
        raise SystemExit(f"{METRICS} not found — run run_glopart_recomb.py first")
    with open(METRICS) as f:
        M = json.load(f)
    for fn in (money_plot, roc_plot, decorr_plot):
        for p in fn(M):
            print("wrote", p)


if __name__ == "__main__":
    main()
