#!/usr/bin/env python
"""Run TopTagSFProcessor over NanoAOD v15 single-lepton + MC files.

Semileptonic-ttbar tag-and-probe for the GloParTv3 top-tag efficiency scale
factors (CMS DP-2025/010 methodology). Writes one skinny per-probe ntuple
(``sf_ntuple_<dataset>.npz``) per dataset -- the population the SF fit /
cut-and-count consumes -- plus a ``.coffea`` with bookkeeping (sumw, nevents,
cutflow). MC ntuples carry a per-row ``weight`` normalized to
``xsec_pb * lumi_pb / sumw``; data carries ``weight = 1``.

Sample groups (each backed by a manifest under ``data/nanoAOD/``):
  Data      -- SingleMuon/Muon single-lepton PD  ({year:{era:[LFN]}})
  TTbar     -- semileptonic + inclusive ttbar     ({year:{sub:{"files":[...]}}})
  WJets     -- W+jets background
  SingleTop -- single-top (tW, t-ch) background

Examples
--------
Local smoke test (2 files/dataset, Data+TTbar)::

    coffea-dask/bin/python run_toptag_sf.py --env local --test \
        --rootdir ~/Projects/rootfiles/ttbar_sl \
        --out outputs/toptag_sf_2024_local_smoke.coffea

Full LPC run, all groups::

    python run_toptag_sf.py --env lpc --sample Data --sample TTbar \
        --sample WJets --sample SingleTop \
        --out outputs/toptag_sf_2024_full.coffea
"""

import os
import sys
import glob
import time
import json
import argparse

sys.path.append(os.path.join(os.getcwd(), 'python'))

import numpy as np
from coffea import processor, util
from coffea.nanoevents import NanoAODSchema

from toptag_sf_processor import TopTagSFProcessor, SF_NTUPLE_FIELDS

# Reuse the WP run-script's Dask / manifest plumbing (module-level defs only).
from run_toptag_wp import (
    LUMI_PB, quiet_dask_worker_logs, _limit_files, _redirect, _dataset_name,
    _load_manifest, start_dask_client, close_dask, default_redirector,
)

class _BroadcastHeavyClient:
    """Client wrapper that replicates every ``submit``-ed future to all workers.

    ``DaskExecutor`` ships the pickled processor as a single ``heavy_input``
    future via ``client.submit``, and every chunk task depends on it. The
    scheduler then piles all chunks onto whichever worker holds that object
    (dependency locality), and work-stealing will not move a large object -- so a
    300-worker pool runs as a 1-worker pool. Chunks go through ``.map``; bare
    ``.submit`` is used only for heavy_input, so broadcasting every submitted
    future is safe. Ported from the committed fix in ttbaranalysis.py.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def submit(self, func, *fargs, **fkwargs):
        future = self._inner.submit(func, *fargs, **fkwargs)
        try:
            from distributed import wait as _dask_wait

            _dask_wait(future, timeout=300)
            self._inner.replicate(future)
        except Exception:
            pass
        return future


# Sample group -> manifest + MC flag. Missing manifests are skipped with a note.
GROUP_CONFIG = {
    'Data':      {'json': 'data/nanoAOD/data_singlemu.json', 'is_mc': False},
    'TTbar':     {'json': 'data/nanoAOD/TTbar_sl.json',       'is_mc': True},
    'WJets':     {'json': 'data/nanoAOD/WJets.json',          'is_mc': True},
    # pT(LNu) < 200 half of the W+jets ladder. VALIDATION only -- 82x the cross
    # section of 'WJets' (6545 vs 79.8 pb), so whether it contributes probes to
    # the pT>400 selection has to be measured, not argued. Do not add to a
    # production run until that measurement says it belongs.
    'WJetsLowPt': {'json': 'data/nanoAOD/WJets_lowpt.json',   'is_mc': True},
    'SingleTop': {'json': 'data/nanoAOD/SingleTop.json',      'is_mc': True},
}


def build_sf_fileset(iov, groups, redirector=None, maxfiles=None):
    fileset = {}
    for group in groups:
        cfg = GROUP_CONFIG.get(group)
        if cfg is None:
            print(f"  [skip] unknown sample group '{group}'")
            continue
        if not os.path.exists(cfg['json']):
            print(f"  [skip] {group}: manifest {cfg['json']} not found")
            continue
        try:
            part = _load_manifest(cfg['json'], iov, group, redirector,
                                  maxfiles, is_mc=cfg['is_mc'])
        except KeyError:
            print(f"  [skip] {group}: IOV {iov} not in {cfg['json']}")
            continue
        fileset.update(part)
    return fileset


def build_local_sf_fileset(rootdir, iov, maxfiles=None):
    """Scan ``<rootdir>/<iov>/{mc,data}/<SAMPLE>/*.root`` for a local smoke run."""
    fileset = {}
    for kind, is_mc in (('mc', True), ('data', False)):
        for sampledir in sorted(glob.glob(os.path.join(rootdir, iov, kind, '*'))):
            if not os.path.isdir(sampledir):
                continue
            name = os.path.basename(sampledir)
            files = _limit_files(
                sorted(glob.glob(os.path.join(sampledir, '*.root'))), maxfiles)
            if not files:
                continue
            ds = name if is_mc else f'Data_{name}'
            fileset[ds] = {
                'files': files,
                'metadata': {'sample': name, 'subsample': name, 'year': iov,
                             'is_mc': is_mc},
            }
    return fileset


def _load_xsec_map(path, iov):
    try:
        with open(path) as f:
            return json.load(f).get(iov, {})
    except Exception:
        return {}


def save_sf_ntuples(output, fileset, lumi_pb, xsec_map, outdir, suffix=''):
    """Write one per-probe ntuple npz per dataset, xsec*lumi/sumw-normalized.

    The processor stores RAW ``genweight``; here MC is scaled to
    ``xsec_pb * lumi_pb / sumw`` (falls back to raw when xsec/sumw missing, e.g.
    a --test run). Data keeps weight = 1.
    """
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for ds, cols in output.get('ntuple', {}).items():
        arrs = {f: acc.value for f, acc in cols.items()}
        n = len(arrs['D'])
        base = ds.split('__')[0]          # __jesUp etc. share the base's norm
        meta = fileset.get(base, {}).get('metadata', {})
        is_mc = meta.get('is_mc', True)
        xsec = meta.get('xsec_pb')
        if xsec is None:
            xrec = xsec_map.get(ds) or xsec_map.get(meta.get('subsample', ''))
            xsec = xrec.get('xsec_pb') if isinstance(xrec, dict) else xrec
        sumw = output.get('sumw', {}).get(base)
        if is_mc and xsec and sumw and lumi_pb:
            norm = float(xsec) * float(lumi_pb) / float(sumw)
            wlabel = 'xsec*lumi/sumw'
        else:
            norm = 1.0
            wlabel = 'raw' if is_mc else 'data(=1)'
        out = {f: arrs[f] for f in SF_NTUPLE_FIELDS if f != 'genweight'}
        out['weight'] = (arrs['genweight'] * np.float32(norm)).astype(np.float32)
        path = os.path.join(outdir, f"sf_ntuple_{ds}{suffix}.npz")
        np.savez_compressed(path, **out)
        saved.append((ds, n, wlabel))
    return saved


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iov', default='2024')
    ap.add_argument('--env', choices=['local', 'lpc', 'casa', 'winterfell'],
                    default='local')
    ap.add_argument('--channel', default='mu', choices=['mu'])
    ap.add_argument('--input-mode', choices=['auto', 'local', 'manifest'],
                    default='auto')
    ap.add_argument('--rootdir',
                    default=os.path.expanduser('~/Projects/rootfiles/ttbar_sl'))
    ap.add_argument('--sample', action='append', default=[],
                    help='sample group(s); default Data + TTbar. Repeatable.')
    ap.add_argument('--xsec-json', default='data/nanoAOD/xsec_pb.json')
    ap.add_argument('-r', '--redirector', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--ntuple-outdir', default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--maxworkers', type=int, default=100)
    ap.add_argument('--dask', action='store_true')
    ap.add_argument('-n', '--nocluster', action='store_true')
    ap.add_argument('--dask-memory', default='5GB')
    ap.add_argument('--chunksize', type=int, default=50000)
    ap.add_argument('--chunksize-dask', type=int, default=100000)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--maxfiles', type=int, default=None)
    ap.add_argument('--maxchunks', type=int, default=None)
    ap.add_argument('--apply-jec-syst', action='store_true',
                    help='MC only: also write __jesUp/__jesDown/__jerUp/__jerDown '
                         'ntuples with the full selection re-run per variation')
    ap.add_argument('--apply-jec', action='store_true',
                    help='apply JEC + JER (incl. JER smearing) to AK8 probes; MC only')
    ap.add_argument('--apply-pu', action='store_true',
                    help='multiply MC weight by nominal PU weight (needs vendored '
                         'PU json on workers). Off for P0.')
    ap.add_argument('--no-lumimask', action='store_true',
                    help='skip golden-JSON lumi masking on data')
    ap.add_argument('--subsample', action='append', default=[],
                    help='only datasets whose name contains this string')
    args = ap.parse_args()

    if args.test:
        if args.maxfiles is None:
            args.maxfiles = 2
        if args.maxchunks is None:
            args.maxchunks = 1

    out = args.out or f'outputs/toptag_sf_{args.iov}.coffea'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    NanoAODSchema.warn_missing_crossrefs = False

    groups = args.sample or ['Data', 'TTbar']
    input_mode = args.input_mode
    if input_mode == 'auto':
        input_mode = 'local' if args.env == 'local' else 'manifest'

    if input_mode == 'local':
        fileset = build_local_sf_fileset(args.rootdir, args.iov, args.maxfiles)
        input_label = f'local:{args.rootdir}'
    else:
        redirector = args.redirector or default_redirector(args.env)
        fileset = build_sf_fileset(args.iov, groups, redirector, args.maxfiles)
        input_label = f'manifest:{redirector}'

    if args.subsample:
        fileset = {ds: spec for ds, spec in fileset.items()
                   if any(s in ds for s in args.subsample)}

    if not fileset:
        sys.exit('No files found for the selected input mode / sample groups')

    print(f"IOV {args.iov} channel={args.channel} env={args.env} "
          f"input={input_label} datasets={len(fileset)}")
    for ds, spec in fileset.items():
        print(f"  {ds:26s} {len(spec['files']):4d} file(s)  "
              f"is_mc={spec['metadata'].get('is_mc')}")

    proc = TopTagSFProcessor(
        iov=args.iov, channel=args.channel,
        apply_pu=args.apply_pu, apply_jec=args.apply_jec,
        apply_jec_syst=args.apply_jec_syst,
        apply_lumimask=not args.no_lumimask,
    )

    tic = time.time()
    use_dask = args.dask or args.env in ('lpc', 'casa')
    client = cluster = None
    try:
        if use_dask:
            quiet_dask_worker_logs()
            client, cluster = start_dask_client(args, os.getcwd())
            if not args.nocluster:
                print(f"waiting for workers (pool target {args.maxworkers})...")
                client.wait_for_workers(1)
                print(f"  {len(client.scheduler_info()['workers'])} worker(s) up "
                      f"after {int(time.time() - tic)}s; pool still filling")
            runner = processor.Runner(
                metadata_cache={},
                executor=processor.DaskExecutor(
                    client=_BroadcastHeavyClient(client), retries=12,
                    treereduction=20),
                schema=NanoAODSchema, chunksize=args.chunksize_dask,
                maxchunks=args.maxchunks, skipbadfiles=True, savemetrics=True,
                xrootdtimeout=600,
            )
        else:
            runner = processor.Runner(
                executor=processor.FuturesExecutor(workers=args.workers),
                schema=NanoAODSchema, chunksize=args.chunksize,
                maxchunks=args.maxchunks, skipbadfiles=True, savemetrics=True,
            )
        output, metrics = runner(fileset, treename='Events',
                                 processor_instance=proc)
    finally:
        close_dask(client, cluster)

    dt = time.time() - tic
    output['datasets_metadata'] = {ds: spec['metadata']
                                   for ds, spec in fileset.items()}
    output['run_info'] = {
        'iov': args.iov, 'channel': args.channel, 'env': args.env,
        'input_mode': input_mode, 'input': input_label,
        'lumi_pb': LUMI_PB.get(args.iov),
        'method': 'CMS DP-2025/010 semileptonic ttbar T&P',
    }

    xsec_map = _load_xsec_map(args.xsec_json, args.iov)
    ntuple_outdir = args.ntuple_outdir or f'outputs/toptag_sf/ntuples_{args.iov}'
    saved = save_sf_ntuples(output, fileset, LUMI_PB.get(args.iov),
                            xsec_map, ntuple_outdir)
    output.pop('ntuple', None)   # drop bulky column accumulators from the .coffea

    util.save(output, out)
    print(f"\nsaved {out}")
    print(f"saved {len(saved)} probe ntuple(s) to {ntuple_outdir}/")
    for ds, n, wlabel in saved:
        print(f"  sf_ntuple_{ds:24s} {n:>9d} probes  weight={wlabel}")
    print(f"elapsed {dt:.1f}s, events/s = {metrics['entries']/max(dt,1e-9):.0f}")
    for ds in fileset:
        nsel = output['nevents'].get(ds, 0)
        nraw = output.get('nevents_raw', {}).get(ds, 0)
        print(f"  {ds:26s} probes={nsel:>8d} / raw={nraw:<10d} "
              f"sumw={output['sumw'].get(ds, 0):.3e}")


if __name__ == '__main__':
    main()
