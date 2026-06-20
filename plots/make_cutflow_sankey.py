#!/usr/bin/env python3
"""Render cutflow funnel/Sankey diagrams (SVG + PNG) for slides.

Standalone, white-background, light-mode styling: dark text, light fills, no
theme classes. Bar width is proportional to log10(events) so all stages stay
visible across many orders of magnitude; the retention % carries the true
magnitude of each cut.

Usage:
    ./coffea-dask/bin/python plots/make_cutflow_sankey.py
"""
from __future__ import annotations

import glob
import math
import subprocess
from pathlib import Path

import coffea.util as cutil

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "plots" / "images" / "cutflow"

# step order: key -> (label, sub for "total")
STEPS = [
    ("total", "Total events"),
    ("trigger", "HLT PFHT1050"),
    ("ht", "H_T > 1400 GeV"),
    ("met", "MET filters"),
    ("ak81", "≥ 1 AK8 jet"),
    ("ak82", "≥ 2 AK8 jets"),
    ("ttbar", "tt̄ candidate"),
]
# category boxes: key, label, family ("pass"/"fail")
CATS = [
    ("2tcen", "Pass · central", "pass"),
    ("2tfwd", "Pass · forward", "pass"),
    ("atcen", "Fail · central", "fail"),
    ("atfwd", "Fail · forward", "fail"),
]

# light-mode hex (50 fill / 600 stroke / 800 text)
BLUE = dict(fill="#E6F1FB", stroke="#185FA5", ribbon="#85B7EB")
TEAL = dict(fill="#E1F5EE", stroke="#0F6E56", text="#085041", ribbon="#5DCAA5")
CORAL = dict(fill="#FAECE7", stroke="#993C1D", text="#712B13", ribbon="#F0997B")
INK = "#2C2C2A"      # primary text
MUTE = "#5F5E5A"     # secondary text
FONT = "Helvetica, Arial, sans-serif"


def extract(o):
    if "cutflow_unweighted" in o:
        c = o["cutflow_unweighted"]
        m = {"total": "input_events", "trigger": "trigger", "ht": "htCut",
             "met": "metfilter", "ak81": "jetkincut", "ak82": "twoFatJets",
             "ttbar": "ttbarcand", "2tcen": "category_2tcen",
             "2tfwd": "category_2tfwd", "atcen": "category_atcen",
             "atfwd": "category_atfwd"}
    else:
        c = o["cutflow"]
        m = {"total": "all events 1", "trigger": "trigger", "ht": "htCut",
             "met": "metfilter", "ak81": "jetkincut", "ak82": "twoFatJets",
             "ttbar": "after_ttbarcandCuts", "2tcen": "2tcen", "2tfwd": "2tfwd",
             "atcen": "atcen", "atfwd": "atfwd"}
    return {k: float(c.get(v, 0.0)) for k, v in m.items()}


def fmt(x):
    if x >= 1e9:
        return f"{x/1e9:.2f}B"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    if x >= 1e4:
        return f"{int(round(x)):,}"
    return f"{int(round(x)):,}"


def pct(p):
    return f"{p:.1f}%" if p >= 1 else f"{p:.3f}%"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_svg(title, vals, total_sub=None):
    CX, BH, MAXW = 215.0, 24.0, 360.0
    tops = [72, 124, 176, 228, 280, 332, 384]

    counts = [vals[k] for k, _ in STEPS]
    catc = [vals[k] for k, _, _ in CATS]
    allc = [c for c in counts + catc if c > 0]
    lo = math.log10(min(allc)) - 0.4
    hi = math.log10(max(counts))
    scale = MAXW / (hi - lo)

    def w(c):
        return max(8.0, (math.log10(c) - lo) * scale)

    spans = []
    for c, t in zip(counts, tops):
        ww = w(c)
        spans.append((CX - ww / 2, CX + ww / 2, t))

    p = []
    p.append(f'<svg width="100%" viewBox="0 0 680 575" role="img" '
             f'xmlns="http://www.w3.org/2000/svg">')
    p.append(f'<title>{esc(title)}</title><desc>Cutflow funnel.</desc>')
    p.append('<rect x="0" y="0" width="680" height="575" fill="#ffffff"/>')
    p.append(f'<text x="40" y="42" font-family="{FONT}" font-size="18" '
             f'font-weight="500" fill="{INK}">{esc(title)}</text>')

    # connecting ribbons between consecutive bars
    for i in range(len(spans) - 1):
        l0, r0, t0 = spans[i]
        l1, r1, t1 = spans[i + 1]
        b0 = t0 + BH
        p.append(f'<polygon points="{l0:.1f},{b0} {r0:.1f},{b0} '
                 f'{r1:.1f},{t1} {l1:.1f},{t1}" fill="{BLUE["ribbon"]}" '
                 f'opacity="0.5"/>')

    # category ribbons from ttbar bar bottom windows to fixed boxes
    l6, r6, t6 = spans[-1]
    by = t6 + BH
    boxes = [(46, 184), (196, 334), (346, 484), (496, 634)]
    win = (r6 - l6) / 4.0
    for j, ((bx0, bx1), (_, _, fam)) in enumerate(zip(boxes, CATS)):
        ox0 = l6 + j * win
        ox1 = l6 + (j + 1) * win
        col = TEAL["ribbon"] if fam == "pass" else CORAL["ribbon"]
        p.append(f'<path d="M{ox0:.1f},{by} C{ox0:.1f},{by+31} {bx0},{by+31} '
                 f'{bx0},470 L{bx1},470 C{bx1},{by+31} {ox1:.1f},{by+31} '
                 f'{ox1:.1f},{by} Z" fill="{col}" opacity="0.4"/>')

    # funnel bars + right-side labels
    total = counts[0]
    prev = None
    for (k, label), c, (l, r, t) in zip(STEPS, counts, spans):
        p.append(f'<rect x="{l:.1f}" y="{t}" width="{(r-l):.1f}" height="{BH}" '
                 f'rx="3" fill="{BLUE["fill"]}" stroke="{BLUE["stroke"]}" '
                 f'stroke-width="0.8"/>')
        cy = t + 12
        p.append(f'<text x="412" y="{cy-2}" font-family="{FONT}" font-size="14" '
                 f'font-weight="500" fill="{INK}">{esc(label)}</text>')
        if prev is None:
            sub = total_sub if total_sub else f"{fmt(c)} input · {int(c):,}"
        else:
            sub = f"{fmt(c)} · {pct(c/prev*100)} kept"
        p.append(f'<text x="412" y="{cy+14}" font-family="{FONT}" font-size="12" '
                 f'fill="{MUTE}">{esc(sub)}</text>')
        prev = c

    # category boxes (fraction relative to the tt-bar candidate sample)
    cand = counts[-1]
    for (bx0, bx1), (k, label, fam), c in zip(boxes, CATS, catc):
        pal = TEAL if fam == "pass" else CORAL
        cx = (bx0 + bx1) / 2
        bw = bx1 - bx0
        p.append(f'<rect x="{bx0}" y="470" width="{bw}" height="56" rx="8" '
                 f'fill="{pal["fill"]}" stroke="{pal["stroke"]}" '
                 f'stroke-width="0.8"/>')
        p.append(f'<text x="{cx}" y="490" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="14" font-weight="500" '
                 f'fill="{pal["text"]}">{esc(label)}</text>')
        p.append(f'<text x="{cx}" y="508" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="14" fill="{pal["text"]}">'
                 f'{fmt(c)}</text>')
        cp = pct(c / cand * 100) if cand else "-"
        p.append(f'<text x="{cx}" y="521" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="12" fill="{pal["text"]}">'
                 f'{cp} of cand.</text>')

    # legend
    p.append(f'<rect x="46" y="548" width="12" height="12" rx="3" '
             f'fill="{TEAL["fill"]}" stroke="{TEAL["stroke"]}" stroke-width="0.8"/>')
    p.append(f'<text x="63" y="558" font-family="{FONT}" font-size="12" '
             f'fill="{MUTE}">Pass region (2 top-tags)</text>')
    p.append(f'<rect x="240" y="548" width="12" height="12" rx="3" '
             f'fill="{CORAL["fill"]}" stroke="{CORAL["stroke"]}" stroke-width="0.8"/>')
    p.append(f'<text x="257" y="558" font-family="{FONT}" font-size="12" '
             f'fill="{MUTE}">Fail region (antitag)</text>')
    p.append(f'<text x="430" y="558" font-family="{FONT}" font-size="12" '
             f'fill="{MUTE}">Bar width ∝ log₁₀(events)</text>')
    p.append('</svg>')
    return "\n".join(p)


def render(title, vals, stem, total_sub=None):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTDIR / f"{stem}.svg"
    png_path = OUTDIR / f"{stem}.png"
    svg_path.write_text(build_svg(title, vals, total_sub=total_sub))
    subprocess.run(["rsvg-convert", "-w", "1600", "-b", "white",
                    str(svg_path), "-o", str(png_path)], check=True)
    print(f"wrote {png_path}")


def main():
    # data: sum eras
    data = None
    for f in sorted(glob.glob(str(REPO / "outputs/dy/data_2024_*.coffea"))):
        d = extract(cutil.load(f))
        if data is None:
            data = {k: 0.0 for k in d}
        for k in d:
            data[k] += d[k]
    render("2024 data — hadronic cutflow", data, "cutflow_sankey_data_2024")

    ttbar = extract(cutil.load(str(REPO / "TTbar_2024_inclusive (1).coffea")))
    render("t t̄ (2024 MC) — hadronic cutflow", ttbar,
           "cutflow_sankey_ttbar_2024")

    # Signal: normalize each mass to a reference cross-section of 1 pb at the
    # 2024 integrated luminosity, so every mass starts at sigma*L before any
    # selection (1 pb x 104 fb^-1 = 104,000 events) and yields read as eff x 1pb.
    SIG_XSEC_PB = 1.0
    SIG_LUMI_FB = 109.95
    for mass, tev in [(2000, "2 TeV"), (4000, "4 TeV")]:
        zp = extract(cutil.load(
            str(REPO / f"outputs/dy/ZPrime{mass}_1_2024_.coffea")))
        ngen = zp["total"]
        scale = SIG_XSEC_PB * SIG_LUMI_FB * 1000.0 / ngen
        zp = {k: v * scale for k, v in zp.items()}
        render(f"Zʹ → t t̄, {tev} (2024 MC) — hadronic cutflow",
               zp, f"cutflow_sankey_zprime{mass}_2024",
               total_sub=f"{fmt(zp['total'])} · σ = 1 pb at {SIG_LUMI_FB:g} fb⁻¹")


if __name__ == "__main__":
    main()
