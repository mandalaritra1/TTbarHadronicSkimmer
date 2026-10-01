#!/usr/bin/env python3
"""Leptonic tops in the all-hadronic selection: derive the jet-level leptonic-top cut c
and compare the current selection with option A and option D.

GloParTv3 has leptonic-top classes (TopbWev, TopbWmv, TopbWtauhv) that the tagger score
S = H / (H + QCD), H = TopbWqq + TopbWq, ignores. Per jet

    f_lep = L / (H + L),   L = TopbWev + TopbWmv + TopbWtauhv

    current  2DAlphabet Pass (both jets tight) and Fail (jet0 tight, jet1 in the band)
    A        current, and both jets also need f_lep < c (tight and band alike)
    C        current, and no isolated e/mu (mini-isolation lepton veto)
    D        A and C

c is the tightest cut that keeps >= 99% of truth fully merged hadronic tops in every
row (pT bin, tight-tagged and band jets alike): the largest per-row 99th percentile of
f_lep, rounded up (MC only; Pass-region data are never used).

Inputs are --ntuple-columns lepstudy outputs: MC rows = events with a tight-tagged jet0,
data rows = Fail region only. The full-statistics data Fail yield comes from the
production histograms (--data-full); the data slice gives the fraction each option keeps.

Outputs (in --outdir): lepstudy.md (all tables), flep_distributions.png,
eff_vs_c.png, summary.png.

Example:
    python plots/leptonic_top_study.py --indir outputs/lepstudy --outdir plots/out/lepstudy
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
from coffea import util

hep.style.use(hep.style.CMS)

TIGHT, LOW = 0.9284, 0.8571
CATS = ["atcen", "atfwd", "2tcen", "2tfwd"]
FAIL, PASS = (0, 1), (2, 3)
PT_EDGES = [400, 480, 600, 800, 1200, np.inf]
TARGET = 0.99
MIN_JETS = 2000         # rows with fewer reference jets (< 20 above q99) do not set c
WLEP = {11: "e", 13: "μ", 15: "τ→had", 1511: "τ→e", 1513: "τ→μ"}
SELECTIONS = ["current", "A", "C", "D"]
VETO_DEFAULT = dict(pt=30.0, iso=0.1)

SAMPLES = {
    "TTto4Q": "TTbar_2024_inclusive_*lepstudy.coffea",
    "TTtoLNu2Q": "TTbar_2024_TTtoLNu2Q_*lepstudy.coffea",
    "TTto2L2Nu": "TTbar_2024_TTto2L2Nu_*lepstudy.coffea",
    "Z′ 2 TeV": "ZPrime1_2024_2000_*lepstudy.coffea",
    "Z′ 4 TeV": "ZPrime1_2024_4000_*lepstudy.coffea",
    "Z′ 6 TeV": "ZPrime1_2024_6000_*lepstudy.coffea",
    "QCD": "QCD_2024_*lepstudy.coffea",
    "data": "data_2024_*lepstudy.coffea",
}
TT = ["TTto4Q", "TTtoLNu2Q", "TTto2L2Nu"]
ZP = ["Z′ 2 TeV", "Z′ 4 TeV", "Z′ 6 TeV"]


# ---------------------------------------------------------------- inputs
def load(indir, pattern):
    paths = sorted(glob.glob(os.path.join(indir, pattern)))
    if not paths:
        return None
    cols = {}
    for p in paths:
        for k, v in util.load(p)["ntuple"].items():
            cols.setdefault(k, []).append(np.asarray(v.value))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    for j in (0, 1):
        H = d[f"jet{j}_TopbWqq"] + d[f"jet{j}_TopbWq"]
        L = d[f"jet{j}_TopbWev"] + d[f"jet{j}_TopbWmv"] + d[f"jet{j}_TopbWtauhv"]
        d[f"jet{j}_S"] = H / (H + d[f"jet{j}_QCD"])
        d[f"jet{j}_flep"] = np.divide(L, H + L, out=np.zeros_like(L), where=(H + L) > 0)
    return d


def data_full_fail(pattern):
    """Full-statistics data yield per Fail category from the production histograms."""
    out = np.zeros(2)
    for p in sorted(glob.glob(pattern)):
        h = util.load(p)["ttbarmass"]
        if "systematic" in h.axes.name:
            h = h[{"systematic": "nominal"}]
        for i in FAIL:
            out[i] += h[{"anacat": i}].sum(flow=True).value
    return out


def jet_table(d):
    """Tight-tagged and band jets of one sample: jet0 of every row (tight), jet1 of
    Pass (tight) and Fail (band) rows."""
    n = len(d["anacat"])
    j1 = d["anacat"] >= 0
    region1 = np.isin(d["anacat"], PASS)          # True tight, False band
    return {
        "pt": np.concatenate([d["jet0_pt"], d["jet1_pt"][j1]]),
        "f": np.concatenate([d["jet0_flep"], d["jet1_flep"][j1]]),
        "tight": np.concatenate([np.ones(n, bool), region1[j1]]),
        "merge": np.concatenate([d["jet0_mergecat"], d["jet1_mergecat"][j1]]),
        "wlep": np.concatenate([d["jet0_wlep"], d["jet1_wlep"][j1]]),
        "w": np.concatenate([d["weight_nominal"], d["weight_nominal"][j1]]),
        "evt": np.concatenate([d["event"], d["event"][j1]]),
    }


def concat_jets(tables):
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def subset(tab, mask):
    return {k: v[mask] for k, v in tab.items()}


def lepton_veto(d, pt=30.0, iso=0.1):
    """Isolated lepton: medium muon |eta| < 2.4 or loose cut-based electron |eta| < 2.5,
    pT > pt, mini-isolation < iso (the ntuple keeps the two leading of each)."""
    veto = np.zeros(len(d["anacat"]), bool)
    for i in (0, 1):
        veto |= ((d[f"mu{i}_pt"] > pt) & ((d[f"mu{i}_id"] & 2) > 0)
                 & (d[f"mu{i}_miniiso"] >= 0) & (d[f"mu{i}_miniiso"] < iso))
        veto |= ((d[f"el{i}_pt"] > pt) & ((d[f"el{i}_id"] & 7) >= 2)
                 & (d[f"el{i}_miniiso"] >= 0) & (d[f"el{i}_miniiso"] < iso))
    return veto


def selections(d, c, veto=VETO_DEFAULT):
    base = d["anacat"] >= 0
    a = (d["jet0_flep"] < c) & (d["jet1_flep"] < c)
    v = lepton_veto(d, **veto)
    return {"current": base, "A": base & a, "C": base & ~v, "D": base & a & ~v}


def yields(d, mask, weighted=True):
    w = d["weight_nominal"] if weighted else np.ones(len(d["anacat"]))
    y = np.array([w[mask & (d["anacat"] == i)].sum() for i in range(4)])
    e = np.sqrt([(w[mask & (d["anacat"] == i)] ** 2).sum() for i in range(4)])
    return y, e


# ---------------------------------------------------------------- c derivation
def q99_table(jets, rng, nboot=200):
    had = (jets["merge"] == 3) & (jets["wlep"] == 0)
    rows = []
    for tight in (True, False):
        # per pT bin, plus all pT pooled (the band has few fully merged tops per bin)
        for lo, hi in list(zip(PT_EDGES[:-1], PT_EDGES[1:])) + [(PT_EDGES[0], np.inf)]:
            m = had & (jets["tight"] == tight) & (jets["pt"] >= lo) & (jets["pt"] < hi)
            f = jets["f"][m]
            if len(f) == 0:
                rows.append((tight, lo, hi, 0, np.nan, np.nan))
                continue
            q = np.quantile(f, TARGET)
            boot = [np.quantile(rng.choice(f, len(f)), TARGET) for _ in range(nboot)]
            rows.append((tight, lo, hi, len(f), q, np.std(boot)))
    return rows


def choose_c(rows, min_jets=MIN_JETS):
    """Tightest c with >= TARGET of the reference jets kept in every row: the largest
    q99 among rows with enough jets, rounded up to 2 significant figures."""
    qmax = max(q for (_, _, _, n, q, _) in rows if n >= min_jets)
    k = 1 - int(np.floor(np.log10(qmax)))
    return np.ceil(qmax * 10 ** k) / 10 ** k, qmax


def frac(sel, total, w=None):
    w = np.ones(len(sel)) if w is None else w
    n, k = w[total].sum(), w[total & sel].sum()
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    neff = n ** 2 / (w[total] ** 2).sum()
    return p, np.sqrt(max(p * (1 - p), 1.0 / neff) / neff)


# ---------------------------------------------------------------- plots
def stamp(fig, inputs):
    if os.environ.get("TTBAR_NO_STAMP"):
        return
    try:
        ver = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], capture_output=True,
                             text=True, timeout=10, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        ver = "unknown"
    try:
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    except Exception:
        pass
    fig.text(0.99, 0.002, f"{datetime.date.today().isoformat()}  |  TTbarHadronicSkimmer {ver}  |  inputs: {inputs}",
             ha="right", va="bottom", fontsize=7, color="0.45", family="monospace")


SEL_COLORS = {"current": "tab:blue", "A": "tab:orange", "C": "tab:red", "D": "tab:purple"}


def ascii_label(text):
    return text.replace("′", "'")       # the CMS font has no prime glyph


def plot_flep(jets_tt, jets_qcd, c, path, tag):
    classes = [
        ("hadronic top, fully merged", (jets_tt["merge"] == 3) & (jets_tt["wlep"] == 0), jets_tt, "k", "-"),
        ("hadronic top, partially merged", np.isin(jets_tt["merge"], (1, 2)) & (jets_tt["wlep"] == 0), jets_tt, "0.5", "-"),
        (r"leptonic top, W$\to$e$\nu$ in jet", jets_tt["wlep"] == 11, jets_tt, "tab:blue", "-"),
        (r"leptonic top, W$\to\mu\nu$ in jet", jets_tt["wlep"] == 13, jets_tt, "tab:orange", "-"),
        (r"leptonic top, W$\to\tau\nu$ in jet", np.isin(jets_tt["wlep"], (15, 1511, 1513)), jets_tt, "tab:green", "-"),
    ]
    if jets_qcd is not None:
        classes.append(("QCD jets", (jets_qcd["merge"] == 0) & (jets_qcd["wlep"] == 0), jets_qcd, "tab:red", "--"))
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), layout="constrained")
    # the band has ~10x fewer jets: coarser bins there
    for ax, tight, title, nbins in ((axes[0], True, "tight-tagged jets", 20), (axes[1], False, "antitag-band jets", 10)):
        bins = np.linspace(0, 1, nbins + 1)
        for label, m, src, color, ls in classes:
            sel = m & (src["tight"] == tight)
            if sel.sum() < 20:
                continue
            h, _ = np.histogram(src["f"][sel], bins=bins, weights=src["w"][sel])
            h = h / h.sum()
            ax.stairs(np.where(h > 0, h, np.nan), bins, color=color, ls=ls, lw=2, label=label)
        ax.axvline(c, color="k", ls=":", lw=1.5, label=f"cut c = {c:g}")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 30)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$f_{lep}$ = L / (H + L)")
        ax.set_ylabel(f"fraction of jets / {1 / nbins:g}")
        ax.text(0.97, 0.95, title, transform=ax.transAxes, ha="right", va="top", fontsize=20)
        if not tight:
            ax.text(0.97, 0.88, "limited MC statistics", transform=ax.transAxes, ha="right", va="top",
                    fontsize=14, color="0.4")
        hep.cms.label("Private Work", data=False, loc=2, ax=ax, rlabel="(13.6 TeV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=15, frameon=False)
    stamp(fig, tag)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_eff_vs_c(jets_tt, jets_qcd, c, path, tag):
    cs = np.logspace(-2, 0, 60)
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    curves = [
        ("hadronic top fully merged, tight", (jets_tt["merge"] == 3) & (jets_tt["wlep"] == 0) & jets_tt["tight"], jets_tt, "k", "-"),
        ("hadronic top fully merged, band", (jets_tt["merge"] == 3) & (jets_tt["wlep"] == 0) & ~jets_tt["tight"], jets_tt, "k", "--"),
        (r"leptonic top W$\to$e$\nu$ / $\mu\nu$", np.isin(jets_tt["wlep"], (11, 13)), jets_tt, "tab:blue", "-"),
        (r"leptonic top W$\to\tau\nu$", np.isin(jets_tt["wlep"], (15, 1511, 1513)), jets_tt, "tab:green", "-"),
    ]
    if jets_qcd is not None:
        curves.append(("QCD jets, tight", jets_qcd["tight"], jets_qcd, "tab:red", "--"))
    for label, m, src, color, ls in curves:
        if m.sum() == 0:
            continue
        f, w = src["f"][m], src["w"][m]
        eff = [w[f < x].sum() / w.sum() for x in cs]
        ax.plot(cs, eff, color=color, ls=ls, lw=2, label=label)
    ax.axvline(c, color="k", ls=":", lw=1.5, label=f"cut c = {c:g}")
    ax.axhline(TARGET, color="0.3", lw=1, ls="-.", label=f"{TARGET:.0%} of hadronic tops")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("cut value c (keep jets with $f_{lep}$ < c)")
    ax.set_ylabel("fraction of jets kept")
    hep.cms.label("Private Work", data=False, loc=2, ax=ax, rlabel="(13.6 TeV)")
    ax.legend(fontsize=12, loc="center", bbox_to_anchor=(0.55, 0.40))
    stamp(fig, tag)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(lep_ratio, keep, path, tag):
    fig, axes = plt.subplots(1, 2, figsize=(22, 9), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.6]})
    regions = ["Fail cen", "Fail fwd", "Pass cen", "Pass fwd"]
    x = np.arange(len(regions))
    wbar = 0.2
    for k, sel in enumerate(SELECTIONS):
        axes[0].bar(x + (k - 1.5) * wbar, [lep_ratio[sel][i] for i in range(4)], wbar,
                    color=SEL_COLORS[sel], label=sel)
    axes[0].set_xticks(x, regions)
    axes[0].set_ylabel("(TTtoLNu2Q + TTto2L2Nu)\n/ TTto4Q", fontsize=20)
    axes[0].set_ylim(0, 1.5 * max(max(v) for v in lep_ratio.values()))
    axes[0].legend(title="selection", fontsize=15, title_fontsize=15, ncol=2, loc="upper right")
    names = list(keep["A"].keys())
    x2 = np.arange(len(names))
    for k, sel in enumerate(SELECTIONS[1:]):
        axes[1].bar(x2 + (k - 1) * 0.27, [keep[sel][n] for n in names], 0.27,
                    color=SEL_COLORS[sel], label=sel)
    axes[1].set_xticks(x2, [ascii_label(n) for n in names], rotation=40, ha="right", fontsize=14)
    axes[1].set_ylabel("yield / current selection")
    axes[1].set_ylim(0, 1.45)
    axes[1].axhline(1, color="0.6", lw=1)
    axes[1].legend(title="selection", fontsize=15, title_fontsize=15, ncol=3, loc="upper right")
    for ax in axes:
        hep.cms.label("Private Work", data=False, loc=2, ax=ax, rlabel="(13.6 TeV)")
    stamp(fig, tag)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_mtt(S, c, path, tag):
    """Left: leptonic tt / TTto4Q vs m_tt in Fail, current and A. Right: Z' Pass m_tt / M
    for events A keeps and removes (unit-normalized to the current Pass yield)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), layout="constrained")
    edges = np.array([1000, 1500, 2000, 2500, 3000, 4000, 6000])
    for sel, color in (("current", SEL_COLORS["current"]), ("A", SEL_COLORS["A"])):
        num, den, err2 = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
        for k in TT:
            d = S.get(k)
            if d is None:
                continue
            m = selections(d, c)[sel] & np.isin(d["anacat"], FAIL)
            h, _ = np.histogram(d["ttbarmass"][m], bins=edges, weights=d["weight_nominal"][m])
            h2, _ = np.histogram(d["ttbarmass"][m], bins=edges, weights=d["weight_nominal"][m] ** 2)
            if k == "TTto4Q":
                den += h
            else:
                num += h
                err2 += h2
        r = np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0) & (num > 0))
        e = np.divide(np.sqrt(err2), den, out=np.full_like(num, np.nan), where=den > 0)
        centers = 0.5 * (edges[1:] + edges[:-1])
        axes[0].errorbar(centers, r, yerr=e, xerr=0.5 * np.diff(edges), fmt="o", color=color, ms=8, label=sel)
    axes[0].set_xlabel(r"$m_{t\bar{t}}$ [GeV]")
    axes[0].set_ylabel("leptonic tt / TTto4Q (Fail)")
    axes[0].set_ylim(0, 0.8)
    axes[0].legend(title="selection", fontsize=16, title_fontsize=16, loc="upper right")
    hep.cms.label("Private Work", data=False, loc=2, ax=axes[0], rlabel="(13.6 TeV)")
    bins = np.linspace(0.2, 1.3, 23)
    for k, mass, color in (("Z′ 2 TeV", 2000, "tab:blue"), ("Z′ 4 TeV", 4000, "tab:orange"), ("Z′ 6 TeV", 6000, "tab:green")):
        d = S.get(k)
        if d is None:
            continue
        sel = selections(d, c)
        base = sel["current"] & np.isin(d["anacat"], PASS)
        norm = d["weight_nominal"][base].sum()
        for part, m, ls in (("kept by A", sel["A"] & base, "-"), ("removed by A", base & ~sel["A"], "--")):
            h, _ = np.histogram(d["ttbarmass"][m] / mass, bins=bins, weights=d["weight_nominal"][m])
            axes[1].stairs(h / norm, bins, color=color, ls=ls, lw=2, label=f"{ascii_label(k)}, {part}")
    axes[1].set_xlabel(r"$m_{t\bar{t}}$ / $M_{Z'}$  (Pass)")
    axes[1].set_ylabel("fraction of current Pass yield / 0.05")
    axes[1].set_ylim(0, None)
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.25)
    axes[1].legend(fontsize=13, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    hep.cms.label("Private Work", data=False, loc=2, ax=axes[1], rlabel="(13.6 TeV)")
    stamp(fig, tag)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="outputs/lepstudy")
    ap.add_argument("--data-full", default="outputs/tight_v1.2/data_2024_*_topvsqcd.coffea",
                    help="production data outputs for the full-statistics Fail yield")
    ap.add_argument("--data-fraction", type=float, default=0.1, help="file fraction of the data slice")
    ap.add_argument("--c", type=float, default=None, help="override the derived cut value")
    ap.add_argument("--outdir", default="plots/out/lepstudy")
    ap.add_argument("--inputs-tag", default="lepstudy ntuples, 2024 MC slices + 10% data Fail")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(20261002)
    md = []

    S = {k: load(args.indir, p) for k, p in SAMPLES.items()}
    missing = [k for k, v in S.items() if v is None]
    if missing:
        print("missing samples:", missing)
    for k, d in S.items():
        if d is not None:
            dev = np.max(np.abs(d["jet0_S"] - d["jet0_tdisc"]))
            assert dev < 1e-4 and d["jet0_tdisc"].min() > TIGHT - 1e-6 or k == "data", (k, dev)

    # --- c from fully merged hadronic tops (TTto4Q, the hadronic top of TTtoLNu2Q, Z');
    # rows pooled over samples and per sample, so no single sample drops below 99%
    tabs = {k: jet_table(S[k]) for k in TT + ZP if S.get(k) is not None}
    sources = [k for k in tabs if k != "TTto2L2Nu"]       # no hadronic tops in dileptonic tt
    ref = concat_jets(list(tabs.values()))
    rows = q99_table(ref, rng)
    src_rows = {k: q99_table(tabs[k], rng, nboot=50) for k in sources}
    c, qmax = choose_c(rows + [r for k in sources for r in src_rows[k]])
    if args.c is not None:
        c = args.c
    md += ["## Cut value c", "",
           f"99th percentile of f_lep for truth fully merged hadronic tops, per row (pT bin × tight/band, "
           f"plus all-pT rows), pooled over TTto4Q, the hadronic top of TTtoLNu2Q and Z′ 2/4/6 TeV, and "
           f"per sample. Rows with < {MIN_JETS} jets (fewer than 20 above q99) do not set c. "
           f"Largest q99 over pooled and per-sample rows = {qmax:.4g} → **c = {c:g}** (rounded up).", "",
           "| jets | pT [GeV] | N (pooled) | q99 pooled | " + " | ".join(f"q99 {k}" for k in sources) + " |",
           "|---|---|---|---|" + "---|" * len(sources)]
    for i, (tight, lo, hi, n, q, e) in enumerate(rows):
        pt = "all" if (lo, hi) == (PT_EDGES[0], np.inf) else f"{lo:.0f}–{hi:.0f}"
        cells = []
        for k in sources:
            _, _, _, nk, qk, _ = src_rows[k][i]
            cells.append("–" if nk == 0 else (f"{qk:.3f}" if nk >= MIN_JETS else f"({qk:.2f}, {nk})"))
        md.append(f"| {'tight' if tight else 'band'} | {pt} | {n} | {q:.4g} ± {e:.2g} | " + " | ".join(cells) + " |")
    md += ["", "Per-sample entries in parentheses have fewer than the minimum jets (value, N)."]

    # held-out check: derive c on even event numbers, measure the retention on odd ones
    even = {k: tabs[k]["evt"] % 2 == 0 for k in sources}
    c_even, _ = choose_c(q99_table(subset(ref, ref["evt"] % 2 == 0), rng, nboot=1)
                         + [r for k in sources for r in q99_table(subset(tabs[k], even[k]), rng, nboot=1)],
                         min_jets=MIN_JETS // 2)
    worst = (1.0, "")
    for k in sources:
        odd = subset(tabs[k], ~even[k])
        had = (odd["merge"] == 3) & (odd["wlep"] == 0)
        for tight in (True, False):
            for lo, hi in zip(PT_EDGES[:-1], PT_EDGES[1:]):
                m = had & (odd["tight"] == tight) & (odd["pt"] >= lo) & (odd["pt"] < hi)
                if m.sum() >= MIN_JETS // 2:
                    r = np.mean(odd["f"][m] < c_even)
                    if r < worst[0]:
                        worst = (r, f"{k}, {'tight' if tight else 'band'} {lo:.0f}–{hi:.0f} GeV, N = {m.sum()}")
    md += ["", f"Held-out check: c derived on even event numbers = {c_even:g}; lowest retention of fully "
           f"merged hadronic tops on odd events (per sample and row, N ≥ {MIN_JETS // 2}) = "
           f"{worst[0]:.4f} ({worst[1]})."]

    # --- per-jet efficiencies at c
    lepj = ref
    qcdj = jet_table(S["QCD"]) if S.get("QCD") is not None else None
    md += ["", f"## Jets kept by f_lep < {c:g}", "",
           "| pT [GeV] | had. full tight | had. full band | had. partial | lep. e | lep. μ | lep. τ (all decays) | QCD tight (weighted) |",
           "|---|---|---|---|---|---|---|---|"]
    for lo, hi in list(zip(PT_EDGES[:-1], PT_EDGES[1:])) + [(400, np.inf)]:
        def eff(tab, m, weighted=False):
            ptm = (tab["pt"] >= lo) & (tab["pt"] < hi)
            p, e = frac(tab["f"] < c, m & ptm, tab["w"] if weighted else None)
            return "–" if np.isnan(p) else f"{p:.4f} ± {e:.4f}"
        cells = [
            eff(ref, (ref["merge"] == 3) & (ref["wlep"] == 0) & ref["tight"]),
            eff(ref, (ref["merge"] == 3) & (ref["wlep"] == 0) & ~ref["tight"]),
            eff(ref, np.isin(ref["merge"], (1, 2)) & (ref["wlep"] == 0)),
            eff(lepj, lepj["wlep"] == 11), eff(lepj, lepj["wlep"] == 13),
            eff(lepj, np.isin(lepj["wlep"], (15, 1511, 1513))),
            eff(qcdj, qcdj["tight"], weighted=True) if qcdj is not None else "–",
        ]
        md.append(f"| {'all' if (lo, hi) == (400, np.inf) else f'{lo:.0f}–{hi:.0f}'} | " + " | ".join(cells) + " |")

    # --- event yields per selection
    Y = {}
    for k, d in S.items():
        if d is None:
            continue
        for sel, m in selections(d, c).items():
            Y[(k, sel)] = yields(d, m, weighted=(k != "data"))
    md += ["", "## Event yields (MC: L-scaled to 109.95 fb⁻¹, Z′ to 1 pb; data: 10% slice, Fail only)", "",
           "| sample | selection | atcen | atfwd | 2tcen | 2tfwd |", "|---|---|---|---|---|---|"]
    for k in S:
        if S[k] is None:
            continue
        for sel in SELECTIONS:
            y, e = Y[(k, sel)]
            cells = [f"{y[i]:.4g} ± {e[i]:.2g}" if not (k == "data" and i in PASS) else "blinded" for i in range(4)]
            md.append(f"| {k} | {sel} | " + " | ".join(cells) + " |")

    # --- summary ratios
    lep_ratio, keep = {}, {sel: {} for sel in SELECTIONS}
    dfull = data_full_fail(args.data_full)
    md += ["", "## Summary", "",
           "| selection | lep. tt / TTto4Q: Fail cen | Fail fwd | Pass cen | Pass fwd | tt / data Fail cen | Fail fwd | data Fail kept cen | fwd |",
           "|---|---|---|---|---|---|---|---|---|"]
    have_data = S.get("data") is not None
    for sel in SELECTIONS:
        had = Y[("TTto4Q", sel)][0]
        lep = Y[("TTtoLNu2Q", sel)][0] + (Y[("TTto2L2Nu", sel)][0] if ("TTto2L2Nu", sel) in Y else 0)
        lep_ratio[sel] = lep / had
        kept = Y[("data", sel)][0][:2] / Y[("data", "current")][0][:2] if have_data else np.full(2, np.nan)
        tt_over_data = (had + lep)[:2] / (dfull * kept)
        md.append(f"| {sel} | " + " | ".join(f"{r:.3f}" for r in lep_ratio[sel])
                  + " | " + " | ".join(f"{r:.3f}" for r in tt_over_data)
                  + " | " + " | ".join(f"{r:.3f}" for r in kept) + " |")
    slice_check = Y[("data", "current")][0][:2] / args.data_fraction / dfull if have_data else np.full(2, np.nan)
    md += ["", f"Data slice check: (slice / {args.data_fraction:g}) / full = "
           f"{slice_check[0]:.3f} (cen), {slice_check[1]:.3f} (fwd). Full Fail data: {dfull[0]:.0f} / {dfull[1]:.0f}."]

    md += ["", "| sample | region | A / current | C / current | D / current |", "|---|---|---|---|---|"]
    for k in TT + ZP + ["QCD"]:
        if S.get(k) is None:
            continue
        for reg, idx in (("Fail", FAIL), ("Pass", PASS)):
            base = Y[(k, "current")][0][list(idx)].sum()
            r = {sel: Y[(k, sel)][0][list(idx)].sum() / base for sel in SELECTIONS[1:]}
            for sel in SELECTIONS[1:]:
                keep[sel][f"{k} {reg}"] = r[sel]
            md.append(f"| {k} | {reg} | {r['A']:.3f} | {r['C']:.3f} | {r['D']:.3f} |")
    if have_data:
        for sel in SELECTIONS[1:]:
            keep[sel]["data Fail"] = Y[("data", sel)][0][:2].sum() / Y[("data", "current")][0][:2].sum()
        md.append(f"| data (10%) | Fail | {keep['A']['data Fail']:.3f} | {keep['C']['data Fail']:.3f} | {keep['D']['data Fail']:.3f} |")

    # --- lepton-veto scan on top of A
    md += ["", "## Lepton-veto scan (on top of A; yields relative to current)", "",
           "| veto pT > | mini-iso < | lep. tt Fail | lep. tt Pass | TTto4Q Pass | Z′ 4 TeV Pass | data Fail |",
           "|---|---|---|---|---|---|---|"]
    for pt in (20.0, 30.0, 55.0):
        for iso in (0.05, 0.1, 0.2, 0.4):
            cells = []
            for k, idx in (("lep", FAIL), ("lep", PASS), ("TTto4Q", PASS), ("Z′ 4 TeV", PASS), ("data", FAIL)):
                names = ["TTtoLNu2Q", "TTto2L2Nu"] if k == "lep" else [k]
                num = den = 0.0
                for n in names:
                    d = S.get(n)
                    if d is None:
                        continue
                    m = selections(d, c, veto=dict(pt=pt, iso=iso))
                    num += yields(d, m["D"], weighted=(n != "data"))[0][list(idx)].sum()
                    den += yields(d, m["current"], weighted=(n != "data"))[0][list(idx)].sum()
                cells.append(f"{num / den:.3f}" if den > 0 else "–")
            md.append(f"| {pt:.0f} | {iso:g} | " + " | ".join(cells) + " |")

    with open(os.path.join(args.outdir, "lepstudy.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    ttj = concat_jets([jet_table(S[k]) for k in TT if S.get(k) is not None])
    plot_flep(ttj, qcdj, c, os.path.join(args.outdir, "flep_distributions.png"), args.inputs_tag)
    plot_eff_vs_c(ref, qcdj, c, os.path.join(args.outdir, "eff_vs_c.png"), args.inputs_tag)
    plot_summary(lep_ratio, keep, os.path.join(args.outdir, "summary.png"), args.inputs_tag)
    plot_mtt(S, c, os.path.join(args.outdir, "mtt_shapes.png"), args.inputs_tag)
    print("wrote", args.outdir)


if __name__ == "__main__":
    main()
