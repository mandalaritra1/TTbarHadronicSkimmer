"""Reproduce the two reference plots:

1. TTbar MC m_tt systematic variations (nominal/up/down + Syst/Nom ratio)
   for jer, jes, q2.  Prefiring is Run-2 only and absent from the Run-3 2024
   outputs, so it is omitted.
2. Z' (1% width) m_tt signal shapes overlaid for several masses.

Run:  .venv/bin/python plots/make_reference_plots.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import coffea.util as cu

hep.style.use("CMS")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "plots", "images", "reference")
os.makedirs(OUTDIR, exist_ok=True)

LUMI_FB = 112.7          # 2024 preliminary
ANACAT = 2               # 2tcen  (2-tag central signal region)


def get_mtt(hobj, systematic):
    """Return (centers, values) for ttbarmass in the chosen category/systematic."""
    h = hobj["ttbarmass"][{"systematic": systematic, "anacat": ANACAT}]
    return h.axes["ttbarmass"].centers, h.values()


# ---------------------------------------------------------------------------
# Plot 1: TTbar systematic variations
# ---------------------------------------------------------------------------
def plot_systematics():
    ttbar = cu.load(os.path.join(REPO, "outputs", "dy", "TTbar_2024_inclusive.coffea"))
    systs = [("jer",      "jer systematic variations"),
             ("jes",      "jes systematic variations"),
             ("pileup",   "pileup systematic variations"),
             ("pdf",      "pdf systematic variations"),
             ("q2",       "q2 systematic variations"),
             ("ttag_pt1", "top-tag pT systematic variations")]

    ncol = 3
    nrow = 2
    edges = ttbar["ttbarmass"].axes["ttbarmass"].edges
    _, nom = get_mtt(ttbar, "nominal")

    fig = plt.figure(figsize=(7 * ncol, 6 * nrow))
    # outer grid: one cell per systematic, generous spacing between cells
    outer = fig.add_gridspec(nrow, ncol, hspace=0.32, wspace=0.30)

    for k, (syst, label) in enumerate(systs):
        r, c = divmod(k, ncol)
        # nested main (height 3) + ratio (height 1) with tight spacing
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(inner[0])
        rax = fig.add_subplot(inner[1], sharex=ax)

        _, up = get_mtt(ttbar, syst + "Up")
        _, dn = get_mtt(ttbar, syst + "Down")

        hep.histplot(nom, edges, ax=ax, color="black", label="Nominal", histtype="step")
        hep.histplot(up,  edges, ax=ax, color="green", label="Up",   histtype="step")
        hep.histplot(dn,  edges, ax=ax, color="red",   label="Down", histtype="step")

        ax.set_xlim(800, 7000)
        ax.set_ylabel("Events / Bin")
        ax.legend(loc="upper right", fontsize=13)
        ax.text(0.05, 0.90, "MC TTbar", transform=ax.transAxes, fontsize=13, style="italic")
        ax.text(0.05, 0.83, label, transform=ax.transAxes, fontsize=12, style="italic")
        ax.text(0.05, 0.76, "2-tag central", transform=ax.transAxes, fontsize=12, fontweight="bold")
        hep.cms.label("Preliminary", data=False, lumi=f"{LUMI_FB:.1f}",
                      com=13.6, loc=0, ax=ax, fontsize=12)
        plt.setp(ax.get_xticklabels(), visible=False)

        with np.errstate(divide="ignore", invalid="ignore"):
            r_up = np.where(nom > 0, up / nom, np.nan)
            r_dn = np.where(nom > 0, dn / nom, np.nan)
        hep.histplot(r_up, edges, ax=rax, color="green", histtype="step")
        hep.histplot(r_dn, edges, ax=rax, color="red",   histtype="step")
        rax.axhline(1.0, color="black", ls="--", lw=1)
        rax.set_ylim(0.5, 1.5)
        rax.set_xlim(800, 7000)
        rax.set_ylabel("Syst/Nom", fontsize=13)
        rax.set_xlabel(r"$m_{t\bar{t}}$ [GeV]")

    out = os.path.join(OUTDIR, "mtt_systematic_variations.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Plot 2: Z' (1% width) signal shapes
# ---------------------------------------------------------------------------
def discover_masses():
    import re
    d = os.path.join(REPO, "outputs", "dy")
    masses = []
    for fn in os.listdir(d):
        m = re.fullmatch(r"ZPrime(\d+)_1_2024_\.coffea", fn)
        if m:
            masses.append(int(m.group(1)))
    return sorted(x for x in masses if x >= 1000)


def plot_signals():
    masses = discover_masses()
    cmap = plt.cm.turbo
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = None
    for j, m in enumerate(masses):
        f = os.path.join(REPO, "outputs", "dy", f"ZPrime{m}_1_2024_.coffea")
        o = cu.load(f)
        h = o["ttbarmass"][{"systematic": "nominal"}][{"anacat": sum}]
        edges = h.axes["ttbarmass"].edges
        vals = h.values()
        tev = m / 1000.0
        lbl = f"Z' 1% Width {tev:g} TeV"
        color = cmap(j / max(len(masses) - 1, 1))
        hep.histplot(vals, edges, ax=ax, color=color, label=lbl, histtype="step", lw=1.3)

    ax.set_xlim(800, 10000)
    ax.set_yscale("log")
    ax.set_ylim(1.0, None)
    ax.set_xlabel(r"$m_{t\bar{t}}$ [GeV]")
    ax.set_ylabel("Events / Bin")
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
    hep.cms.label("Preliminary", data=False, lumi=f"{LUMI_FB:.1f}",
                  com=13.6, loc=0, ax=ax, fontsize=16)
    out = os.path.join(OUTDIR, "zprime_signal_shapes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plot_systematics()
    plot_signals()
