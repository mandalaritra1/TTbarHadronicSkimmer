#!/usr/bin/env python
"""Plot the GloParTv3 top-tag efficiency SF result from cut-and-count ntuples.

Two panels for the headline WP: (left) probe tagging efficiency eps vs pT for
data (bkg-subtracted) and MC (fully-merged top); (right) the data/MC scale factor
vs pT with stat-error bars. Uses toptag_sf_cutcount.compute_sf so the plotted
numbers are exactly what feeds weights.py.

    python plot_sf_result.py --ntuple-dir outputs/nt_p0 --wp tight \
        --lumi-fb 1.86 --out sf_result.png
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from toptag_sf_cutcount import load_ntuples, compute_sf, DEFAULT_PT_EDGES

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    _HEP = True
except Exception:
    _HEP = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ntuple-dir', required=True)
    ap.add_argument('--wp', default='tight')
    ap.add_argument('--lumi-fb', type=float, default=None)
    ap.add_argument('--iov', default='2024')
    ap.add_argument('--out', default='sf_result.png')
    args = ap.parse_args()

    nt = load_ntuples(args.ntuple_dir)
    if not nt:
        sys.exit(f'no ntuples in {args.ntuple_dir}')
    results = compute_sf(nt, pt_edges=DEFAULT_PT_EDGES)
    recs = results[args.wp]

    lo = np.array([r['pt_lo'] for r in recs])
    hi = np.array([r['pt_hi'] if np.isfinite(r['pt_hi']) else lo[-1] + 200
                   for r in recs])
    ctr = 0.5 * (lo + hi)
    xerr = 0.5 * (hi - lo)
    eff_mc = np.array([r['eff_mc'] for r in recs])
    eff_mc_e = np.array([r['eff_mc_err'] for r in recs])
    eff_da = np.array([r['eff_data'] for r in recs])
    eff_da_e = np.array([r['eff_data_err'] for r in recs])
    sf = np.array([r['sf'] for r in recs])
    sf_e = np.array([r['sf_err'] for r in recs])

    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    ax[0].errorbar(ctr, eff_mc, xerr=xerr, yerr=eff_mc_e, marker='s', ms=9,
                   lw=2, capsize=4, color='#3366cc', label=r'$\epsilon_{MC}$ (t$\bar{t}$, full-merged)')
    ax[0].errorbar(ctr, eff_da, xerr=xerr, yerr=eff_da_e, marker='o', ms=9,
                   lw=2, capsize=4, color='k', label=r'$\epsilon_{data}$ (bkg-subtracted)')
    ax[0].set_xlabel('probe $p_{T}$ [GeV]')
    ax[0].set_ylabel(f'top-tag efficiency ({args.wp} WP)')
    ax[0].set_ylim(0, 1.15)
    ax[0].legend(loc='lower right', fontsize=15)
    ax[0].grid(alpha=0.3)

    ax[1].axhline(1.0, color='gray', ls='--', lw=1)
    ax[1].errorbar(ctr, sf, xerr=xerr, yerr=sf_e, marker='o', ms=10, lw=2,
                   capsize=4, color='#c33')
    ax[1].set_xlabel('probe $p_{T}$ [GeV]')
    ax[1].set_ylabel(f'data / MC scale factor ({args.wp} WP)')
    # Adaptive y-range so every point + its error bar and the SF=1 reference
    # line are on-scale (a fixed [0.5,1.5] silently clips low-SF stat-limited
    # bins, which then read as missing data).
    finite = np.isfinite(sf) & np.isfinite(sf_e)
    lo_y = float(np.min((sf - sf_e)[finite])) if finite.any() else 0.0
    hi_y = float(np.max((sf + sf_e)[finite])) if finite.any() else 1.5
    ax[1].set_ylim(max(0.0, lo_y - 0.15), min(2.0, max(hi_y, 1.0) + 0.18))
    ax[1].grid(alpha=0.3)
    for x, y, e in zip(ctr, sf, sf_e):
        ax[1].annotate(f'{y:.3f}\n$\\pm${e:.3f}', (x, y), textcoords='offset points',
                       xytext=(0, 14), ha='center', fontsize=12)

    lab = 'Preliminary'
    if _HEP:
        yr = args.iov
        for a in ax:
            hep.cms.label(lab, ax=a, data=True, year=yr, com=13.6,
                          lumi=args.lumi_fb, fontsize=15)
    fig.suptitle('GloParTv3 top-tag efficiency SF (semileptonic $t\\bar{t}$ '
                 f'tag-and-probe, $\\mu$; P0)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'saved {args.out}')
    for r in recs:
        h = 'inf' if not np.isfinite(r['pt_hi']) else f"{r['pt_hi']:.0f}"
        print(f"  [{r['pt_lo']:.0f},{h}] SF={r['sf']:.3f}+-{r['sf_err']:.3f} "
              f"eps_data={r['eff_data']:.3f} eps_mc={r['eff_mc']:.3f}")


if __name__ == '__main__':
    main()
