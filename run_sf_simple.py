#!/usr/bin/env python
"""Runner-free driver for TopTagSFProcessor (version-robust across coffea).

coffea dropped the classic ``processor.Runner`` executor API in the 2024/2025
line (it is back in 2026.x), so this driver bypasses it entirely: it opens each
file eagerly with ``NanoEventsFactory.from_root`` and calls ``proc.process``
directly, accumulating per-file outputs with ``coffea.processor.accumulate``.
That runs on the LCG_107 coffea (2025.1.0) on lxplus as well as the Mac's
2026.4.0. Same CLI surface (a subset) and same ntuple output as run_toptag_sf.py.

    source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
    python run_sf_simple.py --input-mode manifest -r root://cms-xrd-global.cern.ch/ \
        --sample TTbar --sample Data --sample SingleTop \
        --maxfiles 8 --out outputs/sf_lxplus.coffea --ntuple-outdir outputs/ntuples
"""
import os
import sys
import time
import json
import argparse
import traceback

sys.path.append(os.path.join(os.getcwd(), 'python'))

import numpy as np
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from toptag_sf_processor import TopTagSFProcessor
from run_toptag_sf import (
    build_sf_fileset, build_local_sf_fileset, save_sf_ntuples, _load_xsec_map,
)
from run_toptag_wp import LUMI_PB, default_redirector


def merge(a, b):
    """Recursively accumulate two processor outputs. Dicts recurse; leaf coffea
    accumulators (column_accumulator, defaultdict_accumulator) combine with `+`.
    Avoids depending on coffea.processor.accumulate, which moved across versions.
    """
    if isinstance(a, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(a[k], v) if k in a else v
        return out
    return a + b


# Events per read chunk. NanoAOD v15 files are ~600k events; eager whole-file
# reads peak at ~5.5 GB RSS, so slicing into ~150k-event chunks caps memory at
# ~1.5 GB/worker and keeps many parallel workers off the OOM killer.
CHUNK_EVENTS = int(os.environ.get('SF_CHUNK_EVENTS', 150_000))


def _num_entries(path):
    """Cheap metadata-only entry count for the Events tree."""
    import uproot
    return uproot.open({path: 'Events'}).num_entries


def open_events(path, metadata, entry_start=None, entry_stop=None):
    """Eager NanoEvents open of one entry range, tolerant of signature drift."""
    NanoAODSchema.warn_missing_crossrefs = False
    try:
        return NanoEventsFactory.from_root(
            {path: 'Events'}, schemaclass=NanoAODSchema, metadata=metadata,
            entry_start=entry_start, entry_stop=entry_stop,
            delayed=False).events()
    except TypeError:
        # older/newer signature: positional file + treepath, no delayed kwarg
        return NanoEventsFactory.from_root(
            path, treepath='Events', schemaclass=NanoAODSchema,
            entry_start=entry_start, entry_stop=entry_stop,
            metadata=metadata).events()


# Local staging directory for --stage-local (xrdcp bulk-copy, then local read).
STAGE_DIR = os.environ.get('SF_STAGE_DIR', '/tmp/%s/sfstage' % os.environ.get('USER', 'sf'))
STAGE_LOCAL = os.environ.get('SF_STAGE_LOCAL', '0') == '1'


def _stage_file(path):
    """xrdcp a remote xrootd path to a unique local temp; return local path.

    A random-access fsspec_xrootd read of these files is I/O-bound (~5 min/file
    of network round-trips); an xrdcp bulk transfer is ~60 s, and reading the
    local copy is CPU-bound. So staging first is far faster end to end.
    """
    import subprocess
    os.makedirs(STAGE_DIR, exist_ok=True)
    # unique local name keyed off the remote basename + a pid/counter suffix
    base = path.rstrip('/').split('/')[-1]
    local = os.path.join(STAGE_DIR, f"{os.getpid()}_{base}")
    subprocess.run(['xrdcp', '-f', '-s', path, local],
                   check=True, capture_output=True)
    return local


def _run_file(proc, path, md):
    """Process one already-openable path in bounded entry-chunks."""
    n = _num_entries(path)
    acc = None
    for start in range(0, n, CHUNK_EVENTS):
        stop = min(start + CHUNK_EVENTS, n)
        ev = open_events(path, md, entry_start=start, entry_stop=stop)
        out = proc.process(ev)
        acc = out if acc is None else merge(acc, out)
        del ev, out
    return acc


# module-level so ProcessPoolExecutor can pickle it
def _process_one(job):
    """Worker: (path, metadata, iov, channel, apply_pu, lumimask) -> output dict.

    Reads the file in bounded entry-chunks and folds the per-chunk processor
    outputs together, so peak memory is independent of the file's event count.
    With SF_STAGE_LOCAL=1 the remote file is xrdcp'd to local disk first (much
    faster than streaming random-access reads) and removed afterwards.
    """
    path, md, iov, channel, apply_pu, lumimask = job
    local = None
    try:
        proc = TopTagSFProcessor(iov=iov, channel=channel, apply_pu=apply_pu,
                                 apply_lumimask=lumimask)
        read_path = path
        if STAGE_LOCAL and path.startswith('root://'):
            local = _stage_file(path)
            read_path = local
        acc = _run_file(proc, read_path, md)
        return (md['dataset'], True, acc)
    except Exception as e:
        return (md['dataset'], False, f"{type(e).__name__}: {e}")
    finally:
        if local and os.path.exists(local):
            try:
                os.remove(local)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iov', default='2024')
    ap.add_argument('--channel', default='mu', choices=['mu'])
    ap.add_argument('--input-mode', choices=['manifest', 'local'], default='manifest')
    ap.add_argument('--rootdir', default=os.path.expanduser('~/Projects/rootfiles/ttbar_sl'))
    ap.add_argument('--sample', action='append', default=[])
    ap.add_argument('--subsample', action='append', default=[])
    ap.add_argument('--xsec-json', default='data/nanoAOD/xsec_pb.json')
    ap.add_argument('-r', '--redirector', default='root://cms-xrd-global.cern.ch/')
    ap.add_argument('--out', default=None)
    ap.add_argument('--ntuple-outdir', default=None)
    ap.add_argument('--maxfiles', type=int, default=None)
    ap.add_argument('--workers', type=int, default=1,
                    help='parallel file workers (ProcessPoolExecutor)')
    ap.add_argument('--no-lumimask', action='store_true')
    ap.add_argument('--apply-pu', action='store_true')
    ap.add_argument('--lumi', type=float, default=None,
                    help='effective integrated lumi (pb) for MC xsec norm; '
                         'defaults to the full-IOV LUMI_PB. Use the subset '
                         'effective lumi when processing a data fraction.')
    ap.add_argument('--lumi-from-data-frac', action='store_true',
                    help='auto-set MC lumi = (data files processed / total data '
                         'files in the full manifest) * full-IOV LUMI_PB. Assumes '
                         'uniform recorded-lumi per data file (good for the '
                         'unprescaled single-mu trigger). Overrides --lumi.')
    args = ap.parse_args()

    groups = args.sample or ['Data', 'TTbar']
    if args.input_mode == 'local':
        fileset = build_local_sf_fileset(args.rootdir, args.iov, args.maxfiles)
    else:
        fileset = build_sf_fileset(args.iov, groups, args.redirector, args.maxfiles)
    if args.subsample:
        fileset = {ds: s for ds, s in fileset.items()
                   if any(x in ds for x in args.subsample)}
    if not fileset:
        sys.exit('empty fileset')

    print(f"IOV {args.iov} datasets={len(fileset)} redirector={args.redirector}")
    proc = TopTagSFProcessor(iov=args.iov, channel=args.channel,
                             apply_pu=args.apply_pu,
                             apply_lumimask=not args.no_lumimask)

    tic = time.time()
    lumimask = not args.no_lumimask
    jobs = []
    for ds, spec in fileset.items():
        md = dict(spec['metadata']); md['dataset'] = ds
        for path in spec['files']:
            jobs.append((path, md, args.iov, args.channel, args.apply_pu, lumimask))

    # Fold each per-file output into a single running accumulator as it
    # arrives (incremental merge), so memory stays O(1) in the number of
    # files instead of holding all N per-file outputs until the end -- the
    # full 2024 run is ~7100 files and would otherwise OOM a shared node.
    output = None
    done = {}
    nfail = 0

    def _fold(ds, ok, res):
        nonlocal output, nfail
        if ok:
            output = res if output is None else merge(output, res)
            done[ds] = done.get(ds, 0) + 1
        else:
            nfail += 1
            print(f"  [skip] {ds}: {res}", flush=True)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                ds, ok, res = fut.result()
                _fold(ds, ok, res)
                if i % 10 == 0 or i == len(futs):
                    print(f"  ... {i}/{len(futs)} files done "
                          f"({sum(done.values())} ok, {nfail} fail)", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            _fold(*_process_one(j))
            if i % 10 == 0 or i == len(jobs):
                print(f"  ... {i}/{len(jobs)} files done "
                      f"({sum(done.values())} ok, {nfail} fail)", flush=True)
    for ds, spec in fileset.items():
        print(f"  {ds:26s} processed {done.get(ds, 0)}/{len(spec['files'])} file(s)")

    if output is None:
        sys.exit('no files processed successfully')
    dt = time.time() - tic

    out = args.out or f'outputs/toptag_sf_{args.iov}_simple.coffea'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    xsec_map = _load_xsec_map(args.xsec_json, args.iov)
    ntdir = args.ntuple_outdir or f'outputs/toptag_sf/ntuples_{args.iov}'
    full_lumi = LUMI_PB.get(args.iov)
    if args.lumi_from_data_frac:
        n_data_proc = sum(done.get(ds, 0) for ds, spec in fileset.items()
                          if not spec['metadata'].get('is_mc', True))
        full_data = build_sf_fileset(args.iov, ['Data'], args.redirector, None)
        n_data_total = sum(len(s['files']) for s in full_data.values())
        frac = n_data_proc / n_data_total if n_data_total else 0.0
        lumi_pb = frac * full_lumi
        print(f"effective lumi: {n_data_proc}/{n_data_total} data files "
              f"= {frac:.4f} x {full_lumi:.0f} pb = {lumi_pb:.1f} pb "
              f"({lumi_pb/1000:.3f} fb^-1)")
        norm_label = 'data-frac'
    else:
        lumi_pb = args.lumi if args.lumi is not None else full_lumi
        norm_label = 'override' if args.lumi is not None else 'full-IOV'
    print(f"MC normalized to lumi = {lumi_pb:.1f} pb ({norm_label})")
    saved = save_sf_ntuples(output, fileset, lumi_pb, xsec_map, ntdir)
    output.pop('ntuple', None)
    from coffea.util import save as coffea_save
    coffea_save(output, out)

    print(f"\nsaved {out}\nsaved {len(saved)} ntuple(s) to {ntdir}/")
    for ds, n, wlabel in saved:
        print(f"  sf_ntuple_{ds:24s} {n:>8d} probes  weight={wlabel}")
    print(f"elapsed {dt:.1f}s")
    for ds in fileset:
        print(f"  {ds:26s} probes={output['nevents'].get(ds,0):>7d} / "
              f"raw={output.get('nevents_raw',{}).get(ds,0):<9d} "
              f"sumw={output['sumw'].get(ds,0):.3e}")


if __name__ == '__main__':
    main()
