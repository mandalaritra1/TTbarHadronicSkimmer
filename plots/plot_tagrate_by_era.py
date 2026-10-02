#!/usr/bin/env python3
"""Leading-jet top-tag rate in data per era, against the mean pileup.

Tag rate = cutflow_unweighted['tag_jet0'] / ['ttbarcand']: the fraction of tt̄ candidates whose
higher-score AK8 jet passes the tight GloParTv3 WP (0.9284 in every IOV). The candidates are QCD
dominated, so this is mostly the QCD mistag rate. Data only, no Pass-region yields.

Mean pileup: lumi-weighted brilcalc avgpu (69.2 mb, normtag_BRIL, our golden JSONs, per DAS era run
range; LS with avgpu > 120 dropped) for 2024-2026, computed 2026-10-02 on lxplus in
~/work/ttbarhadronic/lumi_2026/. 2022/2023: mean of the certification pileup histogram per sub-IOV
(same value for every era of a sub-IOV; that method gives 50.0 for 2024 against 49.3 from brilcalc).
"""
import argparse
import datetime
import glob
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from coffea.util import load

MEAN_PU = {
    "2022preEE_C": 34.5, "2022preEE_D": 34.5,
    "2022postEE_E": 41.8, "2022postEE_F": 41.8, "2022postEE_G": 41.8,
    "2023preBPix_B": 44.6, "2023preBPix_C": 44.6, "2023postBPix_D": 44.8,
    "2024_C": 45.8, "2024_D": 44.0, "2024_E": 45.4, "2024_F": 50.2, "2024_G": 50.4, "2024_H": 51.1, "2024_I": 52.3,
    "2025_C": 49.2, "2025_D": 50.6, "2025_E": 52.9, "2025_F": 53.2, "2025_G": 52.0,
    "2026_B": 53.8, "2026_D": 54.6,
}
ORDER = list(MEAN_PU)
# right-panel label offsets (points), hand-placed where eras sit on top of each other
LABEL_OFFSETS = {"2022_E": (-7, 8), "2022_G": (-7, -12), "2023_B": (-8, -12), "2023_D": (8, 4),
                 "2025_E": (0, 9), "2025_D": (-10, 8), "2025_F": (7, -4), "2024_H": (-8, -12), "2024_I": (7, -12),
                 "2025_G": (7, 4), "2024_C": (8, 10), "2024_E": (10, -12), "2024_D": (-10, -2), "2026_D": (-7, 4)}
COLORS = {"2022": "#5790fc", "2023": "#f89c20", "2024": "#e42536", "2025": "#964a8b", "2026": "#7a21dd"}


def stamp(fig, inputs):
    if os.environ.get("TTBAR_NO_STAMP"):
        return
    try:
        ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], capture_output=True,
                             text=True, timeout=10, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        ver = "unknown"
    fig.text(0.99, 0.005, f"{datetime.date.today().isoformat()}  |  TTbarHadronicSkimmer {ver}  |  inputs: {inputs}",
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="outputs/tight_v1.2.1")
    ap.add_argument("--outdir", default="plots/out/tagrate_by_era")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for key in ORDER:
        f = os.path.join(args.indir, f"data_{key}_topvsqcd.coffea")
        if not os.path.exists(f):
            continue
        c = load(f)["cutflow_unweighted"]
        n, k = c["ttbarcand"], c["tag_jet0"]
        p = k / n
        rows.append((key, p, np.sqrt(p * (1 - p) / n), MEAN_PU[key], n, k))

    ref = [r for r in rows if r[0] in ("2024_C", "2024_D", "2024_E", "2024_F", "2024_G")]
    pref = sum(r[5] for r in ref) / sum(r[4] for r in ref)
    lines = ["| era | tt̄ candidates | leading jet tagged | tag rate | / 2024 C–G | mean pileup |", "|---|---|---|---|---|---|"]
    for key, p, e, pu, n, k in rows:
        lines.append(f"| {key} | {int(n)} | {int(k)} | {p:.5f} ± {e:.5f} | {p / pref:.3f} | {pu:.1f} |")
    open(os.path.join(args.outdir, "tagrate_by_era.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    hep.style.use("CMS")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    x = np.arange(len(rows))
    for i, (key, p, e, pu, n, k) in enumerate(rows):
        col = COLORS[key[:4]]
        ax.errorbar(i, p * 100, yerr=e * 100, fmt="o", color=col, ms=8)
        bx.errorbar(pu, p * 100, yerr=e * 100, fmt="o", color=col, ms=8)
        lab = key.replace("preEE", "").replace("postEE", "").replace("preBPix", "").replace("postBPix", "")
        dx, dy = LABEL_OFFSETS.get(lab, (7, 4))
        bx.annotate(lab.replace("_", " "), (pu, p * 100), textcoords="offset points", fontsize=10, color=col,
                    xytext=(dx, dy), ha="right" if dx < 0 else "center" if dx == 0 else "left")
    ax.set_xticks(x, [r[0].replace("_", " ") for r in rows], rotation=60, ha="right", fontsize=12)
    ax.set_ylabel("Leading-jet tag rate (%)")
    ax.axhline(pref * 100, color="0.5", ls="--", lw=1)
    ax.text(0.01, pref * 100, " 2024 C–G", transform=ax.get_yaxis_transform(), va="bottom", fontsize=12, color="0.4")
    ax2 = ax.twinx()
    ax2.plot(x, [r[3] for r in rows], "s", mfc="none", color="0.45", ms=7, label="mean pileup")
    ax2.set_ylabel("Mean pileup (69.2 mb)", color="0.35")
    ax2.set_ylim(25, 65)
    ax2.legend(loc="lower left", fontsize=12, frameon=False)
    for yr, col in COLORS.items():
        ax.plot([], [], "o", color=col, label=yr)
    ax.legend(loc="upper right", ncol=5, fontsize=12, frameon=False)
    ax.set_ylim(1.2, 2.3)
    bx.set_xlabel("Mean pileup (69.2 mb)")
    bx.set_ylabel("Leading-jet tag rate (%)")
    bx.set_ylim(1.2, 2.3)
    bx.set_xlim(32, 58)
    hep.cms.label("Private work", data=True, ax=ax, loc=0, fontsize=16, rlabel="")
    ax.set_title("GloParTv3 tight (0.9284), $t\\bar{t}$ candidates, data", loc="right", fontsize=14)
    stamp(fig, f"{args.indir}/data_*_topvsqcd.coffea (cutflow tag_jet0 / ttbarcand)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"tagrate_by_era.{ext}"))
    print("wrote", args.outdir)


if __name__ == "__main__":
    main()
