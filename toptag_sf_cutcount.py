#!/usr/bin/env python
"""Cut-and-count GloParTv3 top-tag efficiency scale factors from probe ntuples.

Consumes the per-probe ntuples written by ``run_toptag_sf.py``
(``sf_ntuple_<dataset>.npz``, one row per tag-and-probe probe, with a ``weight``
column already normalized to ``xsec*lumi/sumw`` for MC and ``1`` for data) and
produces the merge-category-corrected, background-subtracted efficiency SF

    SF(wp, pt) = eps_data^{full} / eps_MC^{full}

for the fully-merged-top category (CMS DP-2025/010 method, P0 scope: muon
channel, tight WP the headline but all five WPs computed).

Method (per WP, per probe-pT bin)
---------------------------------
Let ``pass`` = probe clears the WP threshold, ``tot`` = all probes in the bin.
Datasets are grouped: ``data`` (weight 1), ``ttbar`` signal MC, ``other`` MC
(single-top, W+jets, ...). ttbar is further split by gen ``merge_cat``:
``full`` (merge_cat == 3, fully-merged top) vs ``nonfull`` (0/1/2).

    eps_MC^full   = S_tt(full,pass) / S_tt(full,tot)
    eps_data^full = [ N_data(pass) - S_other(pass) - S_tt(nonfull,pass) ]
                  / [ N_data(tot)  - S_other(tot)  - S_tt(nonfull,tot)  ]

i.e. from data we subtract the non-ttbar MC prediction and the ttbar
*non-fully-merged* MC prediction, isolating the fully-merged-top efficiency that
the analysis actually applies its tag to. ``S`` are weighted sums; ``N_data`` are
raw counts (weight 1). This closes by construction: if data == total MC then
eps_data^full == eps_MC^full and SF == 1 (verify with ``--closure``).

Uncertainties are the weighted-binomial (normal-approx) efficiency errors,
propagated through the background subtraction; MC-stat and data-stat only (P0 --
no systematics yet).

Usage
-----
    coffea-dask/bin/python toptag_sf_cutcount.py \
        --ntuple-dir outputs/toptag_sf/ntuples_2024 --iov 2024 \
        --out outputs/toptag_sf/sf_2024.json

    # self-test on synthetic ntuples (no files needed), asserts closure:
    coffea-dask/bin/python toptag_sf_cutcount.py --selftest
"""

import os
import sys
import glob
import json
import argparse

import numpy as np

WP_NAMES = ['very_tight', 'tight', 'medium', 'loose', 'very_loose']
MC_FULL = 3   # merge_cat code for fully-merged top (matches the processor)

# Consumer binning: weights.py ttag_scale_factors uses probe-pT edges [400,480,600]
# -> three effective SF bins. Last edge stands in for +inf.
DEFAULT_PT_EDGES = [400.0, 480.0, 600.0, np.inf]


# ---------------------------------------------------------------------------
# dataset grouping
# ---------------------------------------------------------------------------
def classify_dataset(ds):
    """Map an ntuple dataset name to a group: 'data' | 'ttbar' | 'other'."""
    low = ds.lower()
    if low.startswith('data') or 'muon' in low or 'singlemu' in low:
        return 'data'
    if low.startswith('tt') or 'ttbar' in low or 'ttto' in low:
        return 'ttbar'
    return 'other'   # single-top, W+jets, ... -> subtracted background


def load_ntuples(ntuple_dir, subsamples=None):
    """Load every ``sf_ntuple_*.npz`` into {dataset: {field: np.ndarray}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ntuple_dir, 'sf_ntuple_*.npz'))):
        ds = os.path.basename(path)[len('sf_ntuple_'):-len('.npz')]
        if subsamples and not any(s in ds for s in subsamples):
            continue
        with np.load(path) as f:
            out[ds] = {k: f[k] for k in f.files}
    return out


# ---------------------------------------------------------------------------
# weighted efficiency + error helpers
# ---------------------------------------------------------------------------
def _wsum(w, mask=None):
    """(sum w, sum w^2) over mask -- effective-count bookkeeping."""
    if mask is not None:
        w = w[mask]
    return float(np.sum(w)), float(np.sum(w * w))


def _eff_weighted(sp, sp2, st, st2):
    """Weighted efficiency eps=Sp/St and its normal-approx variance.

    var(eps) = [ Sp2*(1-eps)^2 + (St2-Sp2)*eps^2 ] / St^2  (weighted binomial).
    """
    if st <= 0:
        return 0.0, 0.0
    eff = sp / st
    var = (sp2 * (1.0 - eff) ** 2 + max(st2 - sp2, 0.0) * eff ** 2) / (st * st)
    return eff, var


def _bin_indices(pt, edges):
    """Return the per-probe pT-bin index (0..nbin-1); pT<edges[0] -> bin 0."""
    return np.clip(np.digitize(pt, edges[1:-1], right=False),
                   0, len(edges) - 2)


# ---------------------------------------------------------------------------
# core measurement
# ---------------------------------------------------------------------------
def compute_sf(ntuples, pt_edges=DEFAULT_PT_EDGES, wps=WP_NAMES,
               closure=False):
    """Return a nested results dict: {wp: [per-pt-bin record]}.

    ``closure=True`` replaces the data counts by the summed MC prediction
    (ttbar + other), so every SF must come back 1.0 within float tolerance.
    """
    nbin = len(pt_edges) - 1
    groups = {'data': [], 'ttbar': [], 'other': []}
    for ds, cols in ntuples.items():
        groups[classify_dataset(ds)].append(cols)

    def accumulate(group_cols, wp, need_full=None):
        """Weighted (Sp,Sp2,St,St2) per pT bin for a list of datasets.

        need_full: None -> all probes; True -> merge_cat==FULL only;
        False -> merge_cat!=FULL only.
        """
        sp = np.zeros(nbin); sp2 = np.zeros(nbin)
        st = np.zeros(nbin); st2 = np.zeros(nbin)
        for cols in group_cols:
            w = cols['weight'].astype(np.float64)
            ib = _bin_indices(cols['probe_pt'], pt_edges)
            passwp = cols[f'pass_{wp}'].astype(bool)
            sel = np.ones(len(w), dtype=bool)
            if need_full is True:
                sel = cols['merge_cat'] == MC_FULL
            elif need_full is False:
                sel = cols['merge_cat'] != MC_FULL
            for b in range(nbin):
                inb = sel & (ib == b)
                st[b] += np.sum(w[inb]); st2[b] += np.sum(w[inb] ** 2)
                inbp = inb & passwp
                sp[b] += np.sum(w[inbp]); sp2[b] += np.sum(w[inbp] ** 2)
        return sp, sp2, st, st2

    results = {}
    for wp in wps:
        # ttbar fully-merged (the MC efficiency numerator/denominator)
        f_sp, f_sp2, f_st, f_st2 = accumulate(groups['ttbar'], wp, need_full=True)
        # ttbar non-fully-merged (subtracted from data)
        nf_sp, nf_sp2, nf_st, nf_st2 = accumulate(groups['ttbar'], wp,
                                                  need_full=False)
        # non-ttbar background (subtracted from data)
        o_sp, o_sp2, o_st, o_st2 = accumulate(groups['other'], wp)

        if closure:
            # data := ttbar(all) + other ; use weighted sums as the "counts"
            a_sp, a_sp2, a_st, a_st2 = accumulate(groups['ttbar'], wp)
            d_sp = a_sp + o_sp; d_sp2 = a_sp2 + o_sp2
            d_st = a_st + o_st; d_st2 = a_st2 + o_st2
        else:
            d_sp, d_sp2, d_st, d_st2 = accumulate(groups['data'], wp)

        recs = []
        for b in range(nbin):
            eff_mc, var_mc = _eff_weighted(f_sp[b], f_sp2[b], f_st[b], f_st2[b])
            # background-subtracted data numerator / denominator
            num = d_sp[b] - o_sp[b] - nf_sp[b]
            den = d_st[b] - o_st[b] - nf_st[b]
            num2 = d_sp2[b] + o_sp2[b] + nf_sp2[b]   # variances add in quadrature
            den2 = d_st2[b] + o_st2[b] + nf_st2[b]
            eff_dat, var_dat = _eff_weighted(num, num2, den, den2)
            sf = eff_dat / eff_mc if eff_mc > 0 else 0.0
            # relative errors add in quadrature for the ratio
            rel = 0.0
            if eff_dat > 0 and eff_mc > 0:
                rel = np.sqrt(var_dat / eff_dat ** 2 + var_mc / eff_mc ** 2)
            sf_err = abs(sf) * rel
            recs.append({
                'pt_lo': float(pt_edges[b]),
                'pt_hi': float(pt_edges[b + 1]),
                'eff_mc': eff_mc, 'eff_mc_err': float(np.sqrt(var_mc)),
                'eff_data': eff_dat, 'eff_data_err': float(np.sqrt(var_dat)),
                'sf': sf, 'sf_err': sf_err,
                'n_tt_full_tot': f_st[b], 'n_data_tot': d_st[b],
            })
        results[wp] = recs
    return results


def sf_table_for_weights(results, wp='tight', lead_bin=1.0):
    """Extract {nominal, up, down} for one WP in weights.py's 4-bin layout.

    weights.py::_add_ttag_pt_weights uses ``ptbins = [0, 400, 480, 600]`` and only
    reads array indices 1,2,3 (its loop is ``range(1, 4)``); index 0 is the
    <400 GeV filler and is never applied. So we prepend ``lead_bin`` (1.0) and map
    the three measured bins ([400,480], [480,600], [600,inf]) to indices 1,2,3.
    Assumes the results were computed with DEFAULT_PT_EDGES (the 3 measured bins).
    """
    recs = results[wp]
    assert len(recs) == 3, (
        f"weights.py expects 3 measured pT bins (->indices 1,2,3); got {len(recs)}")
    nom = [lead_bin] + [r['sf'] for r in recs]
    up = [lead_bin] + [r['sf'] + r['sf_err'] for r in recs]
    dn = [lead_bin] + [max(r['sf'] - r['sf_err'], 0.0) for r in recs]
    return {'nominal': nom, 'up': up, 'down': dn}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def print_report(results, iov, headline_wp='tight'):
    print(f"\n=== GloParTv3 top-tag efficiency SF (IOV {iov}) ===")
    print("merge-category-corrected, background-subtracted; stat-only errors\n")
    for wp in results:
        star = '  <-- headline' if wp == headline_wp else ''
        print(f"WP = {wp}{star}")
        print(f"  {'pT bin':>13s} {'eps_MC':>14s} {'eps_data':>16s} "
              f"{'SF':>16s}   {'N_tt_full':>10s} {'N_data':>10s}")
        for r in results[wp]:
            hi = 'inf' if not np.isfinite(r['pt_hi']) else f"{r['pt_hi']:.0f}"
            print(f"  [{r['pt_lo']:.0f},{hi:>4s}] "
                  f"{r['eff_mc']:.3f}+-{r['eff_mc_err']:.3f} "
                  f"{r['eff_data']:.3f}+-{r['eff_data_err']:.3f} "
                  f"{r['sf']:.3f}+-{r['sf_err']:.3f} "
                  f"{r['n_tt_full_tot']:10.1f} {r['n_data_tot']:10.1f}")
        print()


# ---------------------------------------------------------------------------
# synthetic self-test (closure + a known-SF injection)
# ---------------------------------------------------------------------------
def _synth_ntuples(seed=0, sf_true=0.90):
    """Build fake ntuples: ttbar (full+nonfull) MC, single-top MC, and pseudo-data.

    Pseudo-data is an exact copy of the MC templates (same pT, weights, gen
    composition) with ONLY the fully-merged-top pass decisions redrawn at
    ``eff_full * sf_true`` and the gen merge truth hidden (merge_cat := 0, as in
    real data). This makes the background subtraction exact, so the estimator must
    recover SF == sf_true (and closure at sf_true == 1).
    """
    rng = np.random.RandomState(seed)
    eff_full, eff_other = 0.60, 0.15

    def mk_mc(n, frac_full, w):
        pt = rng.uniform(400, 900, n)
        merge = np.where(rng.rand(n) < frac_full, MC_FULL, 0).astype(np.int8)
        eff = np.where(merge == MC_FULL, eff_full, eff_other)
        passd = (rng.rand(n) < eff).astype(np.int8)
        return {'probe_pt': pt.astype(np.float32), 'merge_cat': merge,
                'D': rng.rand(n).astype(np.float32),
                'weight': np.full(n, w, dtype=np.float32),
                '_pass': passd}

    def finalize(cols):
        out = {k: v for k, v in cols.items() if not k.startswith('_')}
        for name in WP_NAMES:
            out[f'pass_{name}'] = cols['_pass']
        return out

    tt = mk_mc(20000, 0.7, w=1.3)
    st = mk_mc(4000, 0.0, w=0.8)   # single-top: all non-full background

    # pseudo-data: copy templates, redraw ONLY fully-merged pass at eff*sf_true,
    # then hide the gen merge truth.
    data_parts = []
    for cols in (tt, st):
        d = {k: v.copy() for k, v in cols.items()}
        full = d['merge_cat'] == MC_FULL
        redraw = (rng.rand(len(full)) < min(eff_full * sf_true, 1.0)).astype(np.int8)
        d['_pass'] = np.where(full, redraw, d['_pass']).astype(np.int8)
        d['merge_cat'] = np.zeros(len(full), dtype=np.int8)  # data has no truth
        # keep the copied MC weight so weighted pseudo-data == MC prediction and
        # the background subtraction is exact (real data is weight-1 with MC
        # normalized to match; the estimator only sees weighted sums either way).
        data_parts.append(d)
    data = {k: np.concatenate([p[k] for p in data_parts]) for k in data_parts[0]}

    return ({'TTbar_sig': finalize(tt), 'SingleTop_bkg': finalize(st),
             'Data_C': finalize(data)}, eff_full)


def selftest():
    print("[selftest] closure (data := MC): expect SF = 1.000 in every bin")
    nt, _ = _synth_ntuples(seed=1, sf_true=1.0)
    res = compute_sf(nt, closure=True)
    ok = True
    for wp, recs in res.items():
        for r in recs:
            if r['n_tt_full_tot'] > 0 and abs(r['sf'] - 1.0) > 1e-6:
                ok = False
                print(f"  FAIL {wp} [{r['pt_lo']:.0f}]: SF={r['sf']:.6f}")
    print("  closure", "PASS" if ok else "FAIL")

    print("\n[selftest] injected SF=0.90 on fully-merged pass-rate: "
          "expect recovered SF ~ 0.90")
    nt2, _ = _synth_ntuples(seed=2, sf_true=0.90)
    res2 = compute_sf(nt2, closure=False)
    recs = res2['tight']
    got = np.mean([r['sf'] for r in recs if r['n_tt_full_tot'] > 0])
    print(f"  recovered mean SF (tight) = {got:.3f} (target 0.90)")
    ok2 = abs(got - 0.90) < 0.05
    print("  injection", "PASS" if ok2 else "FAIL")
    return 0 if (ok and ok2) else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ntuple-dir', default='outputs/toptag_sf/ntuples_2024')
    ap.add_argument('--iov', default='2024')
    ap.add_argument('--out', default=None, help='write the SF table as JSON')
    ap.add_argument('--headline-wp', default='tight', choices=WP_NAMES)
    ap.add_argument('--pt-edges', default=None,
                    help='comma list, e.g. 400,480,600 (inf appended)')
    ap.add_argument('--subsample', action='append', default=[])
    ap.add_argument('--closure', action='store_true',
                    help='replace data by MC sum; every SF must be 1.0')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    edges = DEFAULT_PT_EDGES
    if args.pt_edges:
        edges = [float(x) for x in args.pt_edges.split(',')] + [np.inf]

    ntuples = load_ntuples(args.ntuple_dir, args.subsample or None)
    if not ntuples:
        sys.exit(f"no sf_ntuple_*.npz found in {args.ntuple_dir}")
    groups = {}
    for ds in ntuples:
        groups.setdefault(classify_dataset(ds), []).append(ds)
    print(f"loaded {len(ntuples)} ntuple(s) from {args.ntuple_dir}")
    for g, dss in groups.items():
        n = sum(len(ntuples[d]['probe_pt']) for d in dss)
        print(f"  {g:6s}: {n:>8d} probes  {dss}")

    results = compute_sf(ntuples, pt_edges=edges, closure=args.closure)
    print_report(results, args.iov, args.headline_wp)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        table = {args.iov: {wp: sf_table_for_weights(results, wp)
                            for wp in WP_NAMES}}
        payload = {'sf_table': table, 'detail': results,
                   'pt_edges': [e if np.isfinite(e) else None for e in edges],
                   'method': 'CMS DP-2025/010 merge-corrected cut-and-count',
                   'iov': args.iov}
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
