#!/usr/bin/env python3
"""Compare the Z' signal across Run-3 IOVs: the private 2022/2023 NanoAODv15 re-NANO
against the 2024 Summer24 production, through the same v1.2 selection.

Each input is a grouped signal .coffea (dataset axis = ZPrime<mass>_<width>), already
normalized in postprocess to lumi x 1 pb. The selection efficiency is therefore
    eff = yield / (lumi_pb * xsec_pb)
read from the nominal 'ttbarmass' histogram per analysis category. Shapes are
compared unit-normalized, with a ratio to the reference IOV (2024).

Outputs (in --outdir):
    efficiency_M<mass>.png             efficiency per category vs IOV
    <var>_M<mass>.png                  unit-normalized shapes + ratio to 2024
    efficiency.csv / efficiency.md     the numbers behind the efficiency plots

Example:
    python plots/compare_signal_years.py \\
        --input outputs/tight_v1.2/ZPrime1_2024_1400to2000_topvsqcd.coffea \\
        --input outputs/tight_v1.2/ZPrime1_2024_2500to4000_topvsqcd.coffea \\
        --input outputs/tight_v1.2/ZPrime1_2024_4500to7000_topvsqcd.coffea \\
        --input '.claude/worktrees/renano-v1.2/outputs/renano_v12/ZPrime1_202[23]*_weightsOnly.coffea' \\
        --masses 2000 4000 6000 --width 1 --outdir plots/out/signal_years
"""
import argparse
import csv
import datetime
import glob
import os
import subprocess
import warnings

import matplotlib

matplotlib.use("Agg")
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

warnings.filterwarnings("ignore")
hep.style.use(hep.style.CMS)

IOV_ORDER = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]
IOV_LABEL = {"2022preEE": "2022", "2022postEE": "2022EE", "2023preBPix": "2023",
             "2023postBPix": "2023BPix", "2024": "2024", "2025": "2025"}
REF_IOV = "2024"
PETROFF_6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
ANACATS = ["atcen", "atfwd", "2tcen", "2tfwd"]
CAT_GROUPS = {"at": [0, 1], "2t": [2, 3], "atcen": [0], "atfwd": [1], "2tcen": [2], "2tfwd": [3]}
CAT_TEXT = {"at": "1t + antitag (Fail)", "2t": "2t (Pass)"}

# (histogram, value axis, category group, x label, rebin, x range as a function of mass)
SHAPES = [
    ("ttbarmass", "ttbarmass", "2t", r"$m_{t\bar{t}}$ [GeV]", 2, lambda m: (800, 1.35 * m)),
    ("jetmsd", "jetmsd", "2t", r"Leading jet $m_{SD}$ [GeV]", 2, lambda m: (0, 300)),
    ("jet0_pt", "jetpt", "2t", r"Leading jet $p_T$ [GeV]", 1, lambda m: (300, 2000)),
    ("jet1_tdisc", "topscore", "all", "Second jet GloParTv3 top score", 1, lambda m: (0.8, 1.0)),
]


def load_inputs(patterns):
    """{(iov, mass, width): (output, dataset)} from grouped signal coffea files."""
    from coffea import util
    entries = {}
    for pat in patterns:
        paths = sorted(glob.glob(os.path.expanduser(pat)))
        if not paths:
            raise SystemExit(f"no file matches {pat}")
        for path in paths:
            out = util.load(path)
            for ds, norm in out["normalization"].items():
                mass, width = ds.replace("ZPrime", "").split("_")
                key = (norm["year"], int(mass), int(width))
                if key in entries:
                    raise SystemExit(f"{key} appears twice ({path})")
                if not norm.get("applied"):
                    raise SystemExit(f"{path} {ds}: normalization not applied ({norm.get('reason')})")
                entries[key] = (out, ds, path)
    return entries


def nominal(out, name, ds, cats):
    """Nominal histogram for one dataset, summed over the given anacat indices."""
    h = out[name][{"dataset": ds, "systematic": "nominal"}]
    idx = [i for i in range(len(ANACATS))] if cats == "all" else CAT_GROUPS[cats]
    return h[{"anacat": idx}].project(*[a for a in h.axes.name if a != "anacat"]) if len(idx) > 1 \
        else h[{"anacat": idx[0]}]


def efficiency(out, ds):
    norm = out["normalization"][ds]
    denom = norm["lumi_pb"] * norm["xsec_pb"]
    h = out["ttbarmass"][{"dataset": ds, "systematic": "nominal"}]
    res = {}
    for group, idx in CAT_GROUPS.items():
        v = sum(h[{"anacat": i}].sum(flow=True).value for i in idx)
        var = sum(h[{"anacat": i}].sum(flow=True).variance for i in idx)
        res[group] = (v / denom, np.sqrt(var) / denom)
    return res


# cumulative cutflow chain (cutflow_weighted keys); each step's conditional efficiency is
# step / previous step, antitag and 2t both relative to the tagged leading jet.
CUTFLOW_CHAIN = [("trigger", "analysis_events"), ("htCut", "trigger"), ("metfilter", "htCut"),
                 ("jetVetoMap", "metfilter"), ("jetkincut", "jetVetoMap"), ("twoFatJets", "jetkincut"),
                 ("dphi", "twoFatJets"), ("ttbarcand", "dphi"), ("tag_jet0", "ttbarcand"),
                 ("tag_2tag", "tag_jet0"), ("antitag", "tag_jet0")]
CUTFLOW_LABEL = {"trigger": "Trigger", "htCut": r"$H_T$", "metfilter": "MET filt.", "jetVetoMap": "Veto map",
                 "jetkincut": "AK8 kin.+ID", "twoFatJets": "2 AK8", "dphi": r"$\Delta\phi$",
                 "ttbarcand": "Subjets", "tag_jet0": "Tag jet 0", "tag_2tag": "Tag jet 1", "antitag": "Antitag jet 1"}


def conditional_cutflow(out):
    """{step: (step/previous, binomial stat. error)} from the weighted cutflow, the error from the
    unweighted counts; None unless the output holds one dataset."""
    if len(out["normalization"]) != 1:
        return None
    cw, cu = out["cutflow_weighted"], out["cutflow_unweighted"]
    res = {}
    for k, d in CUTFLOW_CHAIN:
        p = cw.get(k, 0.0) / cw[d] if cw.get(d) else np.nan
        n = cu.get(d, 0)
        res[k] = (p, np.sqrt(p * (1 - p) / n) if n else np.nan)
    return res


def plot_cutflow(entries, mass, width, iovs, outdir, inputs_tag):
    flows = {iov: conditional_cutflow(entries[(iov, mass, width)][0]) for iov in iovs}
    if any(f is None for f in flows.values()):
        return None, flows
    steps = [k for k, _ in CUTFLOW_CHAIN]
    fig, ax = plt.subplots(layout="constrained", figsize=(14, 10))
    hep.cms.label("Private Work", data=False, loc=2, ax=ax, rlabel="(13.6 TeV)")
    x = np.arange(len(steps))
    ref = flows[REF_IOV]
    lo, hi = 1, 1
    for i, iov in enumerate(iovs):
        if iov == REF_IOV:
            continue
        r = np.array([flows[iov][s][0] / ref[s][0] for s in steps])
        re = r * np.array([np.hypot(flows[iov][s][1] / flows[iov][s][0], ref[s][1] / ref[s][0]) for s in steps])
        ax.errorbar(x, r, yerr=re, marker="o", ms=9, lw=2, capsize=4, color=PETROFF_6[i % len(PETROFF_6)], label=IOV_LABEL[iov])
        lo, hi = min(lo, np.nanmin(r)), max(hi, np.nanmax(r))
    ax.axhline(1, color="black", lw=1.5)
    ax.set_xticks(x, [CUTFLOW_LABEL[s] for s in steps], rotation=45, ha="right")
    ax.set_ylabel("Step efficiency / 2024")
    span = hi - lo
    ax.set_ylim(lo - 0.15 * span, hi + 0.9 * span)
    ax.text(cms_anchor_x(ax), 0.80, f"Z' {mass / 1000:g} TeV, $\\Gamma/M$ = {width}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=22)
    ax.legend(loc="upper right", framealpha=0.92, ncol=2)
    stamp(fig, inputs_tag)
    path = os.path.join(outdir, f"cutflow_M{mass}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, flows


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


def cms_anchor_x(ax, fallback=0.05):
    for t in ax.texts:
        if t.get_text() == "CMS":
            return t.get_position()[0]
    return fallback


def plot_efficiency(entries, mass, width, iovs, outdir, inputs_tag):
    fig, ax = plt.subplots(layout="constrained")
    hep.cms.label("Private Work", data=False, loc=2, ax=ax, rlabel="(13.6 TeV)")
    x = np.arange(len(iovs))
    ymax = 0
    for (group, marker, color), dx in zip((("2t", "o", PETROFF_6[0]), ("at", "s", PETROFF_6[1])), (-0.08, 0.08)):
        vals = [efficiency(*entries[(iov, mass, width)][:2])[group] for iov in iovs]
        y = np.array([v for v, _ in vals]); e = np.array([u for _, u in vals])
        ax.errorbar(x + dx, y, yerr=e, fmt=marker, color=color, ms=10, capsize=4, label=CAT_TEXT[group])
        ref = y[iovs.index(REF_IOV)] if REF_IOV in iovs else None
        if ref is not None:
            ax.axhline(ref, color=color, ls=":", lw=1.5)
        ymax = max(ymax, float(np.max(y + e)))
    ax.set_xticks(x, [IOV_LABEL[i] for i in iovs])
    ax.set_xlim(-0.6, len(iovs) - 0.4)
    ax.set_ylabel("Selection efficiency")
    ax.set_ylim(0, ymax * 1.6)
    ax.text(cms_anchor_x(ax), 0.80, f"Z' {mass / 1000:g} TeV, $\\Gamma/M$ = {width}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=22)
    ax.legend(loc="upper right", framealpha=0.92)
    stamp(fig, inputs_tag)
    path = os.path.join(outdir, f"efficiency_M{mass}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_shape(entries, mass, width, iovs, spec, outdir, inputs_tag):
    name, axis, cats, xlabel, rebin, xr = spec
    fig, (ax, rax) = plt.subplots(2, 1, layout="constrained", sharex=True, gridspec_kw={"height_ratios": (3, 1)})
    hep.cms.label("Private Work", data=False, loc=2, ax=ax, rlabel="(13.6 TeV)")
    lo, hi = xr(mass)
    shapes = {}
    for iov in iovs:
        out, ds, _ = entries[(iov, mass, width)]
        h = nominal(out, name, ds, cats)
        if rebin > 1:
            h = h[::hist.rebin(rebin)]
        edges = h.axes[0].edges
        v, var = h.values(), h.variances()
        tot = h.values(flow=True).sum()  # unit area including under/overflow
        over = h.values(flow=True)[-1] / tot if tot > 0 else 0.0
        shapes[iov] = (edges, v / tot, np.sqrt(var) / tot, over) if tot > 0 else None
    ymax = 0
    ref = shapes.get(REF_IOV)
    for i, iov in enumerate(iovs):
        if shapes[iov] is None:
            continue
        edges, y, e, over = shapes[iov]
        is_ref = iov == REF_IOV
        color = "black" if is_ref else PETROFF_6[i % len(PETROFF_6)]
        label = IOV_LABEL[iov] + (" (ref.)" if is_ref else "")
        if over >= 0.005:
            label += f", {over:.0%} above {edges[-1]:g}"
        hep.histplot(y, edges, yerr=e, ax=ax, color=color, lw=3 if is_ref else 2, label=label)
        sel = (edges[:-1] >= lo) & (edges[1:] <= hi)
        ymax = max(ymax, float(np.max((y + e)[sel])) if sel.any() else 0)
        if ref is not None and not is_ref:
            with np.errstate(divide="ignore", invalid="ignore"):
                # ratio only where the reference holds >= 2% of its peak bin (low-stat edges masked)
                ok = ref[1] > 0.02 * ref[1].max()
                r = np.where(ok, y / ref[1], np.nan)
                re = np.where(ok, e / ref[1], np.nan)
            hep.histplot(r, edges, yerr=re, ax=rax, color=color, lw=2)
    rax.axhline(1, color="black", lw=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ymax * 1.55)
    ax.set_ylabel("Fraction of events")
    rax.set_ylim(0.3, 1.7)
    rax.set_ylabel("Ratio to 2024")
    rax.set_xlabel(xlabel)
    ax.set_xlabel("")
    region = CAT_TEXT.get(cats, "1t, 2t (all)")
    ax.text(cms_anchor_x(ax), 0.80, f"Z' {mass / 1000:g} TeV, $\\Gamma/M$ = {width}%\n{region}", transform=ax.transAxes,
            ha="left", va="top", fontsize=20)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=18)
    stamp(fig, inputs_tag)
    path = os.path.join(outdir, f"{name}_M{mass}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", required=True, help="coffea file or glob (repeatable)")
    ap.add_argument("--masses", type=int, nargs="+", default=[2000, 4000, 6000])
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--outdir", default="plots/out/signal_years")
    ap.add_argument("--inputs-tag", default="2024 v1.2 production + 2022/23 re-NANO v15 (v1.2, weights-only)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    entries = load_inputs(args.input)
    rows, cutflow_rows = [], []
    for mass in args.masses:
        iovs = [i for i in IOV_ORDER if (i, mass, args.width) in entries]
        if REF_IOV not in iovs:
            print(f"M{mass}: no {REF_IOV} reference, skipped"); continue
        for iov in iovs:
            out, ds, path = entries[(iov, mass, args.width)]
            eff = efficiency(out, ds)
            norm = out["normalization"][ds]
            rows.append({"mass": mass, "iov": iov, "sumw_raw": norm["sumw_raw"],
                         **{f"eff_{g}": eff[g][0] for g in CAT_GROUPS}, **{f"err_{g}": eff[g][1] for g in CAT_GROUPS}})
        print(plot_efficiency(entries, mass, args.width, iovs, args.outdir, args.inputs_tag))
        for spec in SHAPES:
            print(plot_shape(entries, mass, args.width, iovs, spec, args.outdir, args.inputs_tag))
        path, flows = plot_cutflow(entries, mass, args.width, iovs, args.outdir, args.inputs_tag)
        if path:
            print(path)
            cutflow_rows.append((mass, iovs, flows))

    with open(os.path.join(args.outdir, "efficiency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.outdir, "efficiency.md"), "w") as f:
        f.write("| M (GeV) | IOV | gen events (sumw) | eff 2t | eff at | 2t / 2024 | at / 2024 |\n|---|---|---|---|---|---|---|\n")
        def ratio(r, ref, g):
            q = r[f"eff_{g}"] / ref[f"eff_{g}"]
            return q, q * np.hypot(r[f"err_{g}"] / r[f"eff_{g}"], ref[f"err_{g}"] / ref[f"eff_{g}"])
        for mass in sorted({r["mass"] for r in rows}):
            ref = next(r for r in rows if r["mass"] == mass and r["iov"] == REF_IOV)
            for r in (r for r in rows if r["mass"] == mass):
                f.write(f"| {mass} | {IOV_LABEL[r['iov']]} | {r['sumw_raw']:.0f} | {r['eff_2t']:.4f} ± {r['err_2t']:.4f} | "
                        f"{r['eff_at']:.4f} ± {r['err_at']:.4f} | " + ("1 (ref.) | 1 (ref.) |\n" if r is ref else
                        f"{ratio(r, ref, '2t')[0]:.3f} ± {ratio(r, ref, '2t')[1]:.3f} | "
                        f"{ratio(r, ref, 'at')[0]:.3f} ± {ratio(r, ref, 'at')[1]:.3f} |\n"))
    if cutflow_rows:
        with open(os.path.join(args.outdir, "cutflow.md"), "w") as f:
            for mass, iovs, flows in cutflow_rows:
                f.write(f"\n### M = {mass} GeV: conditional step efficiency (step / previous step)\n\n")
                f.write("| step | " + " | ".join(IOV_LABEL[i] for i in iovs) + " |\n|---|" + "---|" * len(iovs) + "\n")
                for step, _ in CUTFLOW_CHAIN:
                    f.write(f"| {step} | " + " | ".join(f"{flows[i][step][0]:.4f} ± {flows[i][step][1]:.4f}" for i in iovs) + " |\n")
        print(open(os.path.join(args.outdir, "cutflow.md")).read())
    print(open(os.path.join(args.outdir, "efficiency.md")).read())


if __name__ == "__main__":
    main()
