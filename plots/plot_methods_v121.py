#!/usr/bin/env python3
"""Method illustrations for the v1.2.1 skimmer (no analysis data involved):

  toppt_sf.png    per-top top-pT scale factor: the TOP-23-008 recipe now applied to the gen
                  last-copy top pT vs the Run-2 formula v1.1 applied to reco AK8 pT
  pu_weights.png  pileup weights vs true pileup: central 2024, derived 2026 and 2022/23 on
                  Summer24 MC, and the 2023 weights that v1 used on Summer24 MC by mistake

  .venv/bin/python plots/plot_methods_v121.py [--outdir plots/out/methods_v121]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import correctionlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from corrections import top_pt_sf  # noqa: E402

PU = os.path.join(ROOT, "data", "corrections", "puWeights")


def stamp(fig, inputs):
    if os.environ.get("TTBAR_NO_STAMP"):
        return
    try:
        ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], capture_output=True,
                             text=True, timeout=10, cwd=HERE).stdout.strip()
    except Exception:
        ver = "unknown"
    fig.text(0.99, 0.005, f"{datetime.date.today().isoformat()}  |  TTbarHadronicSkimmer {ver}  |  {inputs}",
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")


def plot_toppt(outdir):
    pt = np.linspace(0, 3000, 601)
    new = top_pt_sf(pt)
    old = np.exp(0.0615 - 0.0005 * pt)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_between(pt, 2 * new - 1, np.ones_like(pt), color="#5790fc", alpha=0.25,
                    label="New: uncertainty (no reweighting / 2w − 1)")
    ax.plot(pt, new, color="#5790fc", lw=3, label="New: NNLO QCD + NLO EW / POWHEG × 13→13.6 TeV\n(gen last-copy top $p_T$, held above 2 / 1 TeV)")
    ax.plot(pt, old, color="#e42536", lw=3, ls="--", label="v1.1: Run-2 data/POWHEG fit, exp(0.0615 − 0.0005 $p_T$)\n(reco AK8 $p_T$, uncapped, no uncertainty)")
    ax.axvspan(0, 400, color="0.9", zorder=0)
    ax.text(200, 0.08, "below the\nAK8 cut", ha="center", fontsize=13, color="0.4")
    ax.axhline(1, color="0.5", lw=1)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Top-quark $p_T$ [GeV]")
    ax.set_ylabel("Per-top scale factor")
    ax.legend(loc="upper right", fontsize=13, frameon=False)
    hep.cms.label("Private work", data=False, ax=ax, loc=0, fontsize=16, rlabel="13.6 TeV")
    stamp(fig, "python/corrections.py top_pt_sf; v1.1 pTReweighting (git 4b2a2b6^)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, "toppt_sf." + ext), dpi=150)
    plt.close(fig)


def pu_curve(path, name):
    cs = correctionlib.CorrectionSet.from_file(path)
    n = np.arange(0, 99) + 0.5
    return n, {v: np.asarray(cs[name].evaluate(n, v)) for v in ("nominal", "up", "down")}


def plot_pu(outdir):
    prof = np.asarray(json.load(open(os.path.join(PU, "Summer24_to_2022_2023", "derivation_summary.json")))["summer24_profile_measured"])
    prof = prof / prof.sum()
    curves = [
        ("2024, central (LUM)", os.path.join(PU, "2024_Summer24", "puWeights.json.gz"), "Collisions24_CDEFGHI_goldenJSON", "#e42536", "-"),
        ("2026, derived (brilcalc per LS)", os.path.join(PU, "Summer24_2026", "puWeights.json.gz"), "2026_data_on_Summer24", "#7a21dd", "-"),
        ("2022 postEE on Summer24, derived", os.path.join(PU, "Summer24_to_2022_2023", "puWeights.json.gz"), "2022postEE_data_on_Summer24", "#5790fc", "-"),
        ("2023 D central (÷ Summer23BPix MC):\nv1's proxy for Summer24, wrong", os.path.join(PU, "2023_Summer23BPix", "puWeights.json.gz"), "Collisions2023_369803_370790_eraD_GoldenJson", "0.35", "--"),
    ]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for label, path, name, col, ls in curves:
        n, w = pu_curve(path, name)
        mean = float(np.sum(prof * w["nominal"][: len(prof)]))
        ax.plot(n, w["nominal"], color=col, lw=2.5, ls=ls, label=f"{label}  " + r"$\langle w\rangle$" + f" = {mean:.2f}")
        if ls == "-":
            ax.fill_between(n, w["down"], w["up"], color=col, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e3)
    ax.axhline(1, color="0.5", lw=1)
    ax.set_ylabel("Weight (data ÷ MC)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), fontsize=12, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9)
    ax.text(0.98, 0.97, "bands: $\\sigma_{inel}$ = 66.0 / 72.4 mb\n" + r"$\langle w\rangle$" + " on the Summer24 MC profile",
            transform=ax.transAxes, ha="right", va="top", fontsize=12, color="0.35",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    bx.step(np.arange(len(prof)) + 0.5, prof, where="mid", color="0.2", lw=2)
    bx.set_ylabel("Summer24 MC")
    bx.set_xlabel("True number of interactions")
    bx.set_xlim(0, 99)
    hep.cms.label("Private work", data=True, ax=ax, loc=0, fontsize=16, rlabel="13.6 TeV")
    stamp(fig, "data/corrections/puWeights/{2024_Summer24,Summer24_2026,Summer24_to_2022_2023,2023_Summer23BPix}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, "pu_weights." + ext), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "plots", "out", "methods_v121"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    hep.style.use("CMS")
    plot_toppt(args.outdir)
    plot_pu(args.outdir)
    print("wrote", args.outdir)


if __name__ == "__main__":
    main()
