#!/usr/bin/env python3
"""Per-run leading-AK8 GloParTv3 top-tag rate in JetMET data (QCD-dominated).

Question: do the 2026 PromptReco CMSSW_16_0_6 runs (403441-403684), and especially
403567-403663, show a step in the GloParTv3 tagger output compared with neighbouring
runs? The semileptonic T&P sees the real-top tight efficiency drop there (0.73 of 2024
vs ~0.92 elsewhere) with ~900 probes per period; the all-hadronic QCD tag rate has
millions of jets.

Definitions, copied from the skimmer (not re-derived):
  score  = (TopbWqq + TopbWq) / (TopbWqq + TopbWq + QCD)    ttbarprocessor._tscore (v15 IOVs)
  f_lep  = L / (H + L), L = TopbWev + TopbWmv + TopbWtauhv  python/categories.leptonic_top_fraction
  tight  : score > 0.9284 and f_lep < 0.59                  _TAGGER_WPS['tight'], _FLEP_CUT
  band   : 0.8571 < score < 0.9284 and f_lep < 0.59         antitag window (low WP = 'medium')
  AK8 jet: NanoAOD pt > 400, |y| < 2.4, AK8PUPPI_Tight from the vendored jetid.json.gz
           (python/jets.baseline_masks; NanoAODv15 has no FatJet_jetId branch)
  event  : golden JSON, HLT_PFHT1050, >= 1 such jet; the leading-pT one is probed.
No JEC (NanoAOD pt): this is a per-run comparison.

Subcommands
  jobs   build per-job file manifests from data/nanoAOD/data.json
  run    process one manifest over xrootd, write per-run counts (.npz)
  merge  sum the job outputs into per-run / per-period tables (.csv, .json)
  plot   tag rate, band fraction and mean score vs run
"""
import argparse
import csv
import datetime
import glob
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

# --- skimmer definitions --------------------------------------------------------------
TIGHT = 0.9284          # ttbarprocessor._TAGGER_WPS['tight']['2026'/'2025']
BAND_LO = 0.8571        # ttbarprocessor._TAGGER_WPS['medium'] (antitag lower edge)
FLEP_CUT = 0.59         # ttbarprocessor._FLEP_CUT
PT_MIN = 400.0
Y_MAX = 2.4
TRIGGER = "HLT_PFHT1050"

PT_EDGES = np.array([400.0, 500.0, 600.0, 800.0, 1200.0, np.inf])
NPT = len(PT_EDGES) - 1
NS = 200                # score histogram bins on [0, 1]
NC = 9                  # scalar counters per (run, pt bin), see COUNTERS
COUNTERS = ["n_evt", "n_tight", "n_band", "n_tight_s", "n_band_s",
            "n_gt05", "sum_s_gt05", "sum_pt", "n_flep_fail"]

BRANCHES = [
    "run", "luminosityBlock", TRIGGER,
    "FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_mass",
    "FatJet_chHEF", "FatJet_neHEF", "FatJet_chEmEF", "FatJet_neEmEF", "FatJet_muEF",
    "FatJet_chMultiplicity", "FatJet_neMultiplicity",
    "FatJet_globalParT3_TopbWqq", "FatJet_globalParT3_TopbWq", "FatJet_globalParT3_QCD",
    "FatJet_globalParT3_TopbWev", "FatJet_globalParT3_TopbWmv",
    "FatJet_globalParT3_TopbWtauhv",
]

# --- run periods ------------------------------------------------------------------------
# Release periods from toptag-sf-derivation/results/outputs/drift2526_meta/
# release_history_2026.txt (ProcessHistory of the first file per run).
PERIODS_2026 = [
    ("B 16_0_2_patch1", 401844, 402261),
    ("B 16_0_4", 402262, 402513),
    ("D 16_0_6", 403441, 403684),
    ("D 16_0_7_patch1", 403685, 403867),
]
WINDOW = (403567, 403663)   # sub-window inside 16_0_6 (T&P 0.65)

GOLDEN = {
    "2026": "Cert_Collisions2026_401624_403937_golden.json",
    "2025": "Cert_Collisions2025_391658_398903_Golden.json",
}
ERAS = [("2026", "B"), ("2026", "D"), ("2025", "G")]


def period_of(year, run):
    if year == "2025":
        return "2025 G"
    for name, lo, hi in PERIODS_2026:
        if lo <= run <= hi:
            return name
    return "2026 other"


# --- jobs -------------------------------------------------------------------------------
def run_from_lfn(lfn):
    p = lfn.split("/")
    i = p.index("000") if "000" in p else None
    return int(p[i + 1] + p[i + 2]) if i is not None else -1


def cmd_jobs(a):
    with open(a.datajson) as f:
        d = json.load(f)
    os.makedirs(a.outdir, exist_ok=True)
    names = []
    for year, era in ERAS:
        lfns = sorted(d[year][era], key=lambda x: (run_from_lfn(x), x))
        for k in range(0, len(lfns), a.files_per_job):
            name = f"{year}{era}_{k // a.files_per_job:04d}"
            with open(os.path.join(a.outdir, name + ".json"), "w") as f:
                json.dump({"year": year, "era": era, "files": lfns[k:k + a.files_per_job]}, f)
            names.append(name)
        print(f"{year} {era}: {len(lfns)} files")
    with open(a.list, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"{len(names)} jobs -> {a.outdir}, list {a.list}")


# --- run --------------------------------------------------------------------------------
def load_golden(path):
    with open(path) as f:
        g = json.load(f)
    return {int(r): np.array(v, dtype=np.int64) for r, v in g.items()}


def golden_mask(runs, lumis, golden):
    ok = np.zeros(len(runs), dtype=bool)
    for r in np.unique(runs):
        m = runs == r
        rng = golden.get(int(r))
        if rng is None:
            continue
        lo, hi = rng[:, 0], rng[:, 1]
        idx = np.searchsorted(lo, lumis[m], side="right") - 1
        ok[m] = (idx >= 0) & (lumis[m] <= hi[np.clip(idx, 0, None)])
    return ok


def process_chunk(arr, golden, jetid, acc):
    import awkward as ak

    runs = ak.to_numpy(arr["run"]).astype(np.int64)
    lumis = ak.to_numpy(arr["luminosityBlock"]).astype(np.int64)
    gm = golden_mask(runs, lumis, golden)
    trig = ak.to_numpy(arr[TRIGGER]).astype(bool)
    for r in np.unique(runs[gm]):
        a = acc.setdefault(int(r), new_run_acc())
        a["n_golden"] += int(np.sum(gm & (runs == r)))
        a["n_trig"] += int(np.sum(gm & trig & (runs == r)))
    sel_evt = gm & trig
    if not np.any(sel_evt):
        return
    arr = arr[sel_evt]
    runs = runs[sel_evt]

    pt, eta, mass = arr["FatJet_pt"], arr["FatJet_eta"], arr["FatJet_mass"]
    counts = ak.to_numpy(ak.num(pt, axis=1))
    fl = lambda x: ak.to_numpy(ak.flatten(x, axis=1))
    ptf, etaf, mf = fl(pt).astype(np.float64), fl(eta).astype(np.float64), fl(mass).astype(np.float64)
    # rapidity as python/functions.getRapidity: 0.5 ln((E+pz)/(E-pz))
    pz = ptf * np.sinh(etaf)
    e = np.sqrt((ptf * np.cosh(etaf)) ** 2 + mf ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 0.5 * np.log((e + pz) / (e - pz))
    chm = fl(arr["FatJet_chMultiplicity"]).astype(np.int64)
    nem = fl(arr["FatJet_neMultiplicity"]).astype(np.int64)
    idok = jetid.evaluate(
        etaf, fl(arr["FatJet_chHEF"]).astype(np.float64), fl(arr["FatJet_neHEF"]).astype(np.float64),
        fl(arr["FatJet_chEmEF"]).astype(np.float64), fl(arr["FatJet_neEmEF"]).astype(np.float64),
        fl(arr["FatJet_muEF"]).astype(np.float64), chm, nem, chm + nem,
    ) != 0
    selj = (ptf > PT_MIN) & (np.abs(y) < Y_MAX) & idok

    # leading-pT selected jet per event (flat index), -1 if none
    evt_of_jet = np.repeat(np.arange(len(counts)), counts)
    ptsel = np.where(selj, ptf, -1.0)
    order = np.lexsort((-ptsel, evt_of_jet))        # by event, then pt descending
    first = np.zeros(len(counts), dtype=np.int64)
    first[1:] = np.cumsum(counts)[:-1]
    has = counts > 0
    lead = np.full(len(counts), -1, dtype=np.int64)
    lead[has] = order[first[has]]
    ok = has.copy()
    ok[has] = selj[lead[has]]
    if not np.any(ok):
        return
    li = lead[ok]
    r_ev = runs[ok]
    hqq = fl(arr["FatJet_globalParT3_TopbWqq"]).astype(np.float64)[li]
    hq = fl(arr["FatJet_globalParT3_TopbWq"]).astype(np.float64)[li]
    qcd = fl(arr["FatJet_globalParT3_QCD"]).astype(np.float64)[li]
    lep = (fl(arr["FatJet_globalParT3_TopbWev"]).astype(np.float64)[li]
           + fl(arr["FatJet_globalParT3_TopbWmv"]).astype(np.float64)[li]
           + fl(arr["FatJet_globalParT3_TopbWtauhv"]).astype(np.float64)[li])
    H = hqq + hq
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(H + qcd > 0, H / np.where(H + qcd > 0, H + qcd, 1.0), 0.0)
        flep = np.where(H + lep > 0, lep / np.where(H + lep > 0, H + lep, 1.0), 0.0)
    lpt = ptf[li]
    fok = flep < FLEP_CUT
    t_s = s > TIGHT
    b_s = (s > BAND_LO) & (s < TIGHT)
    g05 = s > 0.5
    ptb = np.clip(np.searchsorted(PT_EDGES, lpt, side="right") - 1, 0, NPT - 1)
    sb = np.clip((s * NS).astype(np.int64), 0, NS - 1)
    vals = np.stack([np.ones_like(s), t_s & fok, b_s & fok, t_s, b_s,
                     g05, np.where(g05, s, 0.0), lpt, ~fok], axis=1).astype(np.float64)

    ur, inv = np.unique(r_ev, return_inverse=True)
    for k, r in enumerate(ur):
        m = inv == k
        a = acc.setdefault(int(r), new_run_acc())
        key = ptb[m]
        for c in range(NC):
            a["cnt"][:, c] += np.bincount(key, weights=vals[m, c], minlength=NPT)
        hk = key * NS + sb[m]
        a["h_all"] += np.bincount(hk, minlength=NPT * NS).reshape(NPT, NS)
        a["h_flep"] += np.bincount(hk[fok[m]], minlength=NPT * NS).reshape(NPT, NS)


def new_run_acc():
    return {"n_golden": 0, "n_trig": 0, "cnt": np.zeros((NPT, NC)),
            "h_all": np.zeros((NPT, NS), dtype=np.int64),
            "h_flep": np.zeros((NPT, NS), dtype=np.int64)}


def add_acc(dst, src):
    for r, a in src.items():
        d = dst.setdefault(r, new_run_acc())
        for k in a:
            d[k] = d[k] + a[k]


def cmd_run(a):
    import uproot
    import correctionlib

    with open(a.manifest) as f:
        man = json.load(f)
    golden = load_golden(os.path.join(a.aux, GOLDEN[man["year"]]))
    jetid = correctionlib.CorrectionSet.from_file(os.path.join(a.aux, "jetid.json.gz"))["AK8PUPPI_Tight"]
    acc, ok_files, bad_files = {}, [], []
    t0 = time.time()
    for lfn in man["files"]:
        url = a.redirector.rstrip("/") + "//" + lfn.lstrip("/")
        done = False
        for attempt in range(a.retries + 1):
            facc = {}
            try:
                tf = time.time()
                with uproot.open(url, timeout=300) as f:
                    tree = f["Events"]
                    n = tree.num_entries
                    for arr in tree.iterate(BRANCHES, step_size=a.step, library="ak"):
                        process_chunk(arr, golden, jetid, facc)
                add_acc(acc, facc)
                ok_files.append(lfn)
                done = True
                print(f"ok {lfn} n={n} {time.time() - tf:.0f}s", flush=True)
                break
            except Exception as e:  # noqa: BLE001 - record and move on
                print(f"FAIL attempt {attempt} {lfn}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                time.sleep(10)
        if not done:
            bad_files.append(lfn)
    runs = np.array(sorted(acc), dtype=np.int64)
    np.savez_compressed(
        a.out, year=man["year"], era=man["era"], runs=runs,
        n_golden=np.array([acc[r]["n_golden"] for r in runs], dtype=np.int64),
        n_trig=np.array([acc[r]["n_trig"] for r in runs], dtype=np.int64),
        cnt=np.array([acc[r]["cnt"] for r in runs]).reshape(len(runs), NPT, NC),
        h_all=np.array([acc[r]["h_all"] for r in runs]).reshape(len(runs), NPT, NS),
        h_flep=np.array([acc[r]["h_flep"] for r in runs]).reshape(len(runs), NPT, NS),
        ok_files=np.array(ok_files), bad_files=np.array(bad_files),
    )
    print(f"done {len(ok_files)} ok, {len(bad_files)} failed, {len(runs)} runs, {time.time() - t0:.0f}s")
    return 0


# --- merge ------------------------------------------------------------------------------
def load_outputs(indir):
    tot = {}       # (year, run) -> acc
    ok_files, bad_files, nfiles = set(), set(), 0
    for p in sorted(glob.glob(os.path.join(indir, "*.npz"))):
        z = np.load(p)
        nfiles += 1
        year = str(z["year"])
        for i, r in enumerate(z["runs"]):
            d = tot.setdefault((year, int(r)), new_run_acc())
            d["n_golden"] += int(z["n_golden"][i])
            d["n_trig"] += int(z["n_trig"][i])
            d["cnt"] = d["cnt"] + z["cnt"][i]
            d["h_all"] = d["h_all"] + z["h_all"][i]
            d["h_flep"] = d["h_flep"] + z["h_flep"][i]
        ok_files |= set(z["ok_files"].tolist())
        bad_files |= set(z["bad_files"].tolist())
    return tot, ok_files, bad_files - ok_files, nfiles


def rate(k, n):
    """Binomial fraction and its uncertainty."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    f = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    e = np.sqrt(np.divide(f * (1 - f), n, out=np.zeros_like(k), where=n > 0))
    return f, e


def summarize(cnt_list, ref_cnt=None, h_all=None):
    """Totals over runs; with ref_cnt also the pT-spectrum-corrected tight/band ratio to ref.
    h_all (NPT, NS score histogram, all leading jets) gives the error on the mean score."""
    c = np.sum(cnt_list, axis=0)                 # (NPT, NC)
    n = c[:, 0].sum()
    out = {"n_evt": int(n)}
    for nm, j in (("tight", 1), ("band", 2), ("tight_s", 3), ("band_s", 4)):
        f, e = rate(c[:, j].sum(), n)
        out[nm] = float(f)
        out[nm + "_err"] = float(e)
    out["mean_s_gt05"] = float(c[:, 6].sum() / c[:, 5].sum()) if c[:, 5].sum() > 0 else 0.0
    out["mean_pt"] = float(c[:, 7].sum() / n) if n > 0 else 0.0
    if h_all is not None:
        h = np.asarray(h_all).sum(axis=0)
        ctr = (np.arange(NS) + 0.5) / NS
        m = ctr > 0.5
        k = h[m].sum()
        if k > 1:
            mu = np.sum(h[m] * ctr[m]) / k
            out["mean_s_gt05_err"] = float(np.sqrt(np.sum(h[m] * (ctr[m] - mu) ** 2) / k / k))
    if ref_cnt is not None:
        for nm, j in (("tight", 1), ("band", 2)):
            rref = np.divide(ref_cnt[:, j], ref_cnt[:, 0], out=np.zeros(NPT), where=ref_cnt[:, 0] > 0)
            exp = float(np.sum(c[:, 0] * rref))
            obs = float(c[:, j].sum())
            # stat error: Poisson on obs; reference error neglected when ref >> sample
            out[nm + "_obs_over_exp"] = obs / exp if exp > 0 else 0.0
            out[nm + "_obs_over_exp_err"] = np.sqrt(obs) / exp if exp > 0 else 0.0
    return out


def cmd_merge(a):
    tot, ok_files, bad_files, njobs = load_outputs(a.indir)
    os.makedirs(a.outdir, exist_ok=True)
    rel = {}
    if a.release_map and os.path.exists(a.release_map):
        for line in open(a.release_map):
            p = line.split()
            if len(p) >= 3 and p[1].isdigit():
                rel[int(p[1])] = " ".join(p[2:])
    keys = sorted(tot)
    # reference for the pT-corrected ratios: all 2026 runs outside 16_0_6
    zero = np.zeros((NPT, NC))
    ref = zero + sum([tot[k]["cnt"] for k in keys if k[0] == "2026"
                      and period_of(*k) not in ("D 16_0_6", "2026 other")], zero)
    ref25 = zero + sum([tot[k]["cnt"] for k in keys if k[0] == "2025"], zero)

    rows = []
    for (year, r) in keys:
        d = tot[(year, r)]
        s = summarize([d["cnt"]], ref if year == "2026" else ref25, d["h_all"])
        rows.append({"year": year, "run": r, "period": period_of(year, r),
                     "in_window": int(year == "2026" and WINDOW[0] <= r <= WINDOW[1]),
                     "release": rel.get(r, ""), "n_golden": d["n_golden"], "n_trig": d["n_trig"], **s})
    fields = list(rows[0].keys())
    with open(os.path.join(a.outdir, "tagrate_per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})

    groups = {}
    for (year, r) in keys:
        groups.setdefault(period_of(year, r), []).append((year, r))
        if year == "2026" and WINDOW[0] <= r <= WINDOW[1]:
            groups.setdefault("D 16_0_6 window 403567-403663", []).append((year, r))
        if year == "2026" and PERIODS_2026[2][1] <= r < WINDOW[0]:
            groups.setdefault("D 16_0_6 before window (403441-403566)", []).append((year, r))
        if year == "2026" and period_of(year, r) not in ("D 16_0_6", "2026 other"):
            groups.setdefault("2026 all except 16_0_6", []).append((year, r))
    per = {}
    for g, ks in groups.items():
        ry = ref25 if g == "2025 G" else ref
        per[g] = {"runs": len(ks), "first": ks[0][1], "last": ks[-1][1],
                  **summarize([tot[k]["cnt"] for k in ks], ry,
                              np.sum([tot[k]["h_all"] for k in ks], axis=0))}
        # run-to-run scatter: chi2/ndf of per-run tight rates about the group mean
        n = np.array([tot[k]["cnt"][:, 0].sum() for k in ks])
        t = np.array([tot[k]["cnt"][:, 1].sum() for k in ks])
        m = n > a.min_events
        if m.sum() > 1:
            p = t[m].sum() / n[m].sum()
            var = p * (1 - p) / n[m]
            per[g]["tight_chi2_ndf"] = float(np.sum((t[m] / n[m] - p) ** 2 / var) / (m.sum() - 1))
            per[g]["runs_used_chi2"] = int(m.sum())

    # pT-binned tight / band rates per group
    for g, ks in groups.items():
        c = np.sum([tot[k]["cnt"] for k in ks], axis=0)
        per[g]["pt_bins"] = {
            f"{PT_EDGES[b]:.0f}-{PT_EDGES[b + 1]:.0f}": {
                "n": int(c[b, 0]),
                "tight": float(c[b, 1] / c[b, 0]) if c[b, 0] else 0.0,
                "band": float(c[b, 2] / c[b, 0]) if c[b, 0] else 0.0}
            for b in range(NPT)}

    # score histograms per group (leading jets passing f_lep), for the shape comparison
    hists = {g: np.sum([tot[k]["h_flep"] for k in ks], axis=0).tolist() for g, ks in groups.items()}

    meta = {"date": str(datetime.date.today()), "njob_outputs": njobs,
            "files_ok": len(ok_files), "files_failed": sorted(bad_files),
            "definitions": {"tight": TIGHT, "band_lo": BAND_LO, "flep_cut": FLEP_CUT,
                            "pt_min": PT_MIN, "y_max": Y_MAX, "trigger": TRIGGER,
                            "jet_id": "AK8PUPPI_Tight (jetid.json.gz, correctionlib)",
                            "pt_edges": PT_EDGES.tolist()},
            "reference_for_obs_over_exp": "2026: all 2026 runs outside 16_0_6; 2025: all 2025 G"}
    with open(os.path.join(a.outdir, "tagrate_per_period.json"), "w") as f:
        json.dump({"meta": meta, "periods": per}, f, indent=1)
    with open(os.path.join(a.outdir, "score_hists_per_period.json"), "w") as f:
        json.dump({"score_edges": np.linspace(0, 1, NS + 1).tolist(),
                   "pt_edges": PT_EDGES.tolist(), "h_flep": hists}, f)

    print(f"{njobs} job outputs, {len(ok_files)} files ok, {len(bad_files)} failed")
    hdr = f"{'period':42s} {'runs':>4s} {'n_evt':>10s} {'tight':>16s} {'band':>16s} {'t obs/exp':>14s} {'<s>':>7s} {'chi2/ndf':>8s}"
    print(hdr)
    for g, v in per.items():
        print(f"{g:42s} {v['runs']:4d} {v['n_evt']:10d} "
              f"{v['tight']:.5f}+-{v['tight_err']:.5f} {v['band']:.5f}+-{v['band_err']:.5f} "
              f"{v.get('tight_obs_over_exp', 0):.3f}+-{v.get('tight_obs_over_exp_err', 0):.3f} "
              f"{v['mean_s_gt05']:.4f} {v.get('tight_chi2_ndf', float('nan')):8.2f}")


# --- plot -------------------------------------------------------------------------------
def git_sha(repo):
    try:
        sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", repo, "diff", "--quiet", "HEAD", "--"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use(hep.style.CMS)
    rows = list(csv.DictReader(open(os.path.join(a.outdir, "tagrate_per_run.csv"))))
    stamp = (f"{datetime.date.today().isoformat()}  |  skimmer {git_sha(a.repo)}"
             f"  |  scripts/qcd_tagrate_per_run.py, NanoAOD PromptReco JetMET")
    panels = [("tight", "Tight tag rate"), ("band", "Band fraction"),
              ("mean_s_gt05", "Mean score (s > 0.5)")]
    sets = []
    for year, fname, periods, rlab in (
            ("2026", "tagrate_per_run_2026", PERIODS_2026, "2026 B+D (13.6 TeV)"),
            ("2025", "tagrate_per_run_2025G", [("2025 G", 0, 10 ** 7)], "2025 G (13.6 TeV)")):
        rr = [r for r in rows if r["year"] == year and int(r["n_evt"]) >= a.min_events
              and r["period"] != "2026 other"]
        if rr:
            sets.append((year, fname, periods, rlab, rr))
    # common y ranges so the 2026 and 2025 G figures read on the same scale
    yr = {}
    for col, _ in panels:
        lo, hi = np.inf, -np.inf
        for *_, rr in sets:
            v = np.array([float(r[col]) for r in rr])
            e = np.array([float(r.get(col + "_err", 0) or 0) for r in rr])
            lo, hi = min(lo, np.min(v - e)), max(hi, np.max(v + e))
        yr[col] = (lo - 0.15 * (hi - lo), hi + 0.30 * (hi - lo))   # headroom for labels

    for year, fname, periods, rlab, rr in sets:
        runs = np.array([int(r["run"]) for r in rr])
        n = np.array([float(r["n_evt"]) for r in rr])
        x = np.arange(len(rr))
        fig, axs = plt.subplots(len(panels), 1, sharex=True, layout="constrained",
                                figsize=(21, 21))
        for ax, (col, lab) in zip(axs, panels):
            v = np.array([float(r[col]) for r in rr])
            e = np.array([float(r.get(col + "_err", 0) or 0) for r in rr])
            if year == "2026":
                in_w = (runs >= WINDOW[0]) & (runs <= WINDOW[1])
                if in_w.any():
                    ax.axvspan(x[in_w].min() - 0.5, x[in_w].max() + 0.5, color="#e42536",
                               alpha=0.15, lw=0, label="runs 403567-403663")
                for name, lo, hi in periods[1:]:
                    b = np.searchsorted(runs, lo)
                    if 0 < b < len(runs):
                        ax.axvline(b - 0.5, color="0.4", ls="--", lw=1.5)
            ax.errorbar(x, v, yerr=e if e.any() else None, fmt="o", ms=5, lw=1.2, color="k",
                        label="per run")
            for k, (name, lo, hi) in enumerate(periods):
                m = (runs >= lo) & (runs <= hi)
                if m.any():
                    mu = np.sum(v[m] * n[m]) / np.sum(n[m])       # event-weighted mean
                    ax.hlines(mu, x[m].min() - 0.4, x[m].max() + 0.4, color="#5790fc", lw=2.5,
                              label="period mean" if k == 0 else None)
            ax.set_ylim(*yr[col])
            ax.set_ylabel(lab)
            ax.yaxis.set_label_coords(-0.07, 0.5)
        for name, lo, hi in periods:
            m = (runs >= lo) & (runs <= hi)
            if m.any():
                axs[0].text(x[m].mean(), 0.96, name, transform=axs[0].get_xaxis_transform(),
                            ha="center", va="top", fontsize=20)
        axs[1].legend(loc="upper left", fontsize=18, ncol=3, framealpha=1)
        axs[-1].set_xlim(-1, len(x))
        step = max(1, int(np.ceil(len(x) / 25)))
        axs[-1].set_xticks(x[::step])
        axs[-1].set_xticklabels([str(r) for r in runs[::step]], rotation=70, fontsize=16)
        axs[-1].set_xlabel(f"Run (>= {a.min_events} selected events, in run order)", labelpad=8)
        hep.cms.label("Private work", data=True, loc=0, rlabel=rlab, ax=axs[0])
        fig.get_layout_engine().set(rect=(0, 0.02, 1, 0.98))
        fig.text(0.99, 0.002, stamp, ha="right", va="bottom", fontsize=9, color="0.45",
                 family="monospace")
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.outdir, f"{fname}.{ext}"), dpi=110)
        plt.close(fig)
        print("wrote", os.path.join(a.outdir, fname + ".png"))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    j = sp.add_parser("jobs")
    j.add_argument("--datajson", default=os.path.join(repo, "data/nanoAOD/data.json"))
    j.add_argument("--outdir", default="jobs")
    j.add_argument("--list", default="jobs_all.txt")
    j.add_argument("--files-per-job", type=int, default=5)
    r = sp.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--aux", default=".", help="dir with golden JSONs and jetid.json.gz")
    r.add_argument("--redirector", default="root://cms-xrd-global.cern.ch/")
    r.add_argument("--retries", type=int, default=1)
    r.add_argument("--step", type=int, default=250000)
    m = sp.add_parser("merge")
    m.add_argument("--indir", required=True)
    m.add_argument("--outdir", default=os.path.join(repo, "plots/out/tagrate_per_run_2026"))
    m.add_argument("--release-map", default=os.path.expanduser(
        "~/Projects/toptag-sf-derivation/results/outputs/drift2526_meta/release_history_2026.txt"))
    m.add_argument("--min-events", type=int, default=2000)
    p = sp.add_parser("plot")
    p.add_argument("--outdir", default=os.path.join(repo, "plots/out/tagrate_per_run_2026"))
    p.add_argument("--repo", default=repo)
    p.add_argument("--min-events", type=int, default=2000)
    a = ap.parse_args()
    return {"jobs": cmd_jobs, "run": cmd_run, "merge": cmd_merge, "plot": cmd_plot}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
