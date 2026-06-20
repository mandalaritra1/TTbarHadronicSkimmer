#!/usr/bin/env python
"""Run TopTagWPProcessor over NanoAOD v15 files and save histograms.

This produces the histogram accumulator consumed by ``plot_toptag_wp.py`` to
derive the GloParTv3 top-tagging working points. No ntuples are written.

Examples
--------
Local smoke test over local files on a laptop::

    coffea-dask/bin/python run_toptag_wp.py --env local --test \
        --out outputs/toptag_wp_2024_local_smoke.coffea

Full local run over files under ``--rootdir``::

    coffea-dask/bin/python run_toptag_wp.py --env local --workers 4 \
        --out outputs/toptag_wp_2024.coffea

Full LPC Dask run over the 2024 MC manifest files via XRootD::

    python run_toptag_wp.py --env lpc \
        --out outputs/toptag_wp_2024_full.coffea

Data-only LPC run for the TopvsQCD data/MC shape comparison::

    python run_toptag_wp.py --env lpc --sample Data \
        --out outputs/toptag_score_data_2024.coffea

Coffea-casa smoke test over 1-2 files per dataset::

    python run_toptag_wp.py --env casa --test \
        --out outputs/toptag_wp_2024_casa_smoke.coffea

The local layout expected under ``--rootdir``
(default ``~/Projects/rootfiles/ttbar``) is::

    <rootdir>/<iov>/mc/<SAMPLE>/*.root

Samples whose directory name starts with ``TT`` are treated as signal
(gen-matched tops fill the ``"matched"`` jettype); everything else is treated as
background (QCD) and fills only ``"incl"``.
"""

import os
import sys
import glob
import time
import argparse
import json
import logging
import re

sys.path.append(os.path.join(os.getcwd(), 'python'))

from coffea import processor, util
from coffea.nanoevents import NanoAODSchema

from toptag_wp_processor import TopTagWPProcessor

# Per-sample cross sections [pb] for local directory scans. Manifest-backed
# runs use the xsec_pb already stored in data/nanoAOD/*.json.
XSEC_PB = {
    'TTto4Q': 350.6,          # 2024 inclusive TTbar (data/nanoAOD/TTbar.json)
    'QCD_PT600to800': 178.7,  # local directory name without manifest hyphen
    'QCD_PT-600to800': 178.7,
}

# Integrated luminosity [pb^-1] per IOV (preliminary; matches ttbarprocessor).
LUMI_PB = {
    '2023': 27000.0,
    '2024': 109950.0,  # golden-JSON certified 2024 lumi (109.95 fb^-1)
}

QCD_MIN_SUBSAMPLE_PT = 300.0
QCD_SUBSAMPLE_RE = re.compile(r'QCD_(?:Bin-)?PT-?(\d+(?:\.\d+)?)to')


def qcd_subsample_min_pt(name):
    match = QCD_SUBSAMPLE_RE.search(str(name))
    return float(match.group(1)) if match else None


def keep_qcd_subsample(sample, section, metadata=None, min_pt=QCD_MIN_SUBSAMPLE_PT):
    """Keep QCD generated-pT bins starting at ``min_pt`` GeV."""
    if sample != 'QCD':
        return True
    metadata = metadata or {}
    subsample = metadata.get('subsample', section)
    low = qcd_subsample_min_pt(subsample)
    return low is None or low >= min_pt


def quiet_dask_worker_logs():
    """Suppress Dask worker connection chatter that corrupts progress output."""
    try:
        import dask

        dask.config.set({'logging.distributed': 'error'})
    except Exception:
        pass

    for name in [
        'distributed',
        'distributed.scheduler',
        'distributed.core',
        'distributed.nanny',
        'distributed.worker',
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


def _limit_files(files, maxfiles=None):
    files = list(files)
    if maxfiles is not None:
        files = files[:maxfiles]
    return files


def _redirect(files, redirector):
    if not redirector:
        return list(files)
    return [
        f if '://' in f or not f.startswith('/store/') else redirector + f
        for f in files
    ]


def _dataset_name(sample, section):
    if sample == 'QCD':
        return section
    if sample == 'Data':
        return f'Data_{section}' if section else 'Data'
    if section and section != 'inclusive':
        return f'{sample}_{section}'
    return sample


def _load_manifest(path, iov, sample, redirector=None, maxfiles=None, is_mc=True):
    with open(path) as f:
        manifest = json.load(f)
    if iov not in manifest:
        raise KeyError(f"IOV {iov} not found in {path}")

    fileset = {}
    for section, entry in manifest[iov].items():
        if isinstance(entry, dict) and 'files' in entry:
            files = entry['files']
            metadata = dict(entry.get('metadata', {}))
        else:
            files = entry
            metadata = {}

        files = _limit_files(_redirect(files, redirector), maxfiles)
        if not files:
            continue

        metadata.setdefault('sample', sample)
        metadata.setdefault('subsample', section)
        metadata.setdefault('year', iov)
        metadata.setdefault('is_mc', is_mc)
        if not keep_qcd_subsample(sample, section, metadata):
            continue

        fileset[_dataset_name(sample, section)] = {
            'files': files,
            'metadata': metadata,
        }
    return fileset


def build_manifest_fileset(qcd_json, ttbar_json, iov, redirector=None, maxfiles=None,
                           samples=None, data_json='data/nanoAOD/data.json'):
    samples = set(samples or ['QCD', 'TTbar'])
    fileset = {}
    if 'QCD' in samples:
        fileset.update(_load_manifest(qcd_json, iov, 'QCD', redirector, maxfiles, is_mc=True))
    if 'TTbar' in samples:
        fileset.update(_load_manifest(ttbar_json, iov, 'TTbar', redirector, maxfiles, is_mc=True))
    if 'Data' in samples:
        fileset.update(_load_manifest(data_json, iov, 'Data', redirector, maxfiles, is_mc=False))
    return fileset


def build_local_fileset(rootdir, iov, maxfiles=None, samples=None):
    samples = set(samples or ['QCD', 'TTbar'])
    fileset = {}
    if samples & {'QCD', 'TTbar'}:
        pattern = os.path.join(rootdir, iov, 'mc', '*')
        for sampledir in sorted(glob.glob(pattern)):
            if not os.path.isdir(sampledir):
                continue
            sample = os.path.basename(sampledir)
            if not keep_qcd_subsample('QCD', sample, {'subsample': sample}):
                continue
            files = _limit_files(sorted(glob.glob(os.path.join(sampledir, '*.root'))), maxfiles)
            if not files:
                continue
            fileset[sample] = {
                'files': files,
                'metadata': {
                    'sample': sample,
                    'subsample': sample,
                    'year': iov,
                    'is_mc': True,
                    'xsec_pb': XSEC_PB.get(sample),
                },
            }
    if 'Data' in samples:
        pattern = os.path.join(rootdir, iov, 'data', '*')
        for sampledir in sorted(glob.glob(pattern)):
            if not os.path.isdir(sampledir):
                continue
            section = os.path.basename(sampledir)
            files = _limit_files(sorted(glob.glob(os.path.join(sampledir, '*.root'))), maxfiles)
            if not files:
                continue
            fileset[f'Data_{section}'] = {
                'files': files,
                'metadata': {
                    'sample': 'Data',
                    'subsample': section,
                    'year': iov,
                    'is_mc': False,
                },
            }
    return fileset


def default_redirector(env):
    if env == 'casa':
        return 'root://xcache/'
    if env == 'winterfell':
        return '/mnt/data/cms/'
    return 'root://cmsxrootd.fnal.gov/'


def start_dask_client(args, repo_root):
    from dask.distributed import Client, Security
    import dask.distributed

    if args.nocluster:
        cluster = dask.distributed.LocalCluster(
            n_workers=args.workers,
            threads_per_worker=1,
            scheduler_port=0,
            dashboard_address=':8787',
            protocol='tcp://',
            security=Security(),
        )
    elif args.env == 'lpc':
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            memory=args.dask_memory,
            transfer_input_files=['python'],
            scheduler_options={'dashboard_address': ':8787'},
        )
        cluster.adapt(minimum=1, maximum=args.maxworkers)
    elif args.env == 'casa':
        from coffea_casa import CoffeaCasaCluster

        cluster = CoffeaCasaCluster(memory=args.dask_memory)
        cluster.adapt(minimum=4, maximum=args.maxworkers)
    else:
        cluster = dask.distributed.LocalCluster(
            n_workers=args.workers,
            threads_per_worker=1,
            scheduler_port=0,
            dashboard_address=':8787',
            protocol='tcp://',
            security=Security(),
        )

    client = Client(cluster)

    if args.env == 'casa' and not args.nocluster:
        from distributed.diagnostics.plugin import UploadDirectory

        client.register_worker_plugin(
            UploadDirectory(os.path.join(repo_root, 'python'), restart=True, update_path=True),
            nanny=True,
        )

    return client, cluster


def close_dask(client, cluster):
    if client is not None:
        client.close()
    if cluster is not None:
        cluster.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iov', default='2024')
    ap.add_argument('--env', choices=['local', 'lpc', 'casa', 'winterfell'], default='local',
                    help='local uses --rootdir; lpc/casa/winterfell use manifests by default')
    ap.add_argument('--input-mode', choices=['auto', 'local', 'manifest'], default='auto',
                    help='auto = local for --env local, manifest otherwise')
    ap.add_argument('--rootdir', default=os.path.expanduser('~/Projects/rootfiles/ttbar'))
    ap.add_argument('--qcd-json', default='data/nanoAOD/QCD.json')
    ap.add_argument('--ttbar-json', default='data/nanoAOD/TTbar.json')
    ap.add_argument('--data-json', default='data/nanoAOD/data.json')
    ap.add_argument('--sample', choices=['QCD', 'TTbar', 'Data'], action='append', default=[],
                    help='sample group(s) to run; default is QCD and TTbar')
    ap.add_argument('-r', '--redirector', default=None,
                    help='redirector for manifest /store paths; env default if omitted')
    ap.add_argument('--out', default=None, help='output .coffea path')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--maxworkers', type=int, default=100)
    ap.add_argument('--dask', action='store_true', help='use DaskExecutor')
    ap.add_argument('-n', '--nocluster', action='store_true',
                    help='with --dask, start an explicit local Dask cluster')
    ap.add_argument('--dask-memory', default='5GB')
    ap.add_argument('--chunksize', type=int, default=50000)
    ap.add_argument('--chunksize-dask', type=int, default=100000)
    ap.add_argument('--test', action='store_true',
                    help='cap to 2 files per dataset and 1 chunk per dataset unless overridden')
    ap.add_argument('--maxfiles', type=int, default=None,
                    help='cap input files per dataset before running')
    ap.add_argument('--maxchunks', type=int, default=None,
                    help='cap chunks per dataset (use 1-2 for a smoke test)')
    args = ap.parse_args()

    if args.test:
        if args.maxfiles is None:
            args.maxfiles = 2
        if args.maxchunks is None:
            args.maxchunks = 1

    out = args.out or f'outputs/toptag_wp_{args.iov}.coffea'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    NanoAODSchema.warn_missing_crossrefs = False

    input_mode = args.input_mode
    if input_mode == 'auto':
        input_mode = 'local' if args.env == 'local' else 'manifest'

    if input_mode == 'local':
        fileset = build_local_fileset(
            args.rootdir,
            args.iov,
            maxfiles=args.maxfiles,
            samples=args.sample or ['QCD', 'TTbar'],
        )
        input_label = f'local:{args.rootdir}'
    else:
        redirector = args.redirector or default_redirector(args.env)
        fileset = build_manifest_fileset(
            qcd_json=args.qcd_json,
            ttbar_json=args.ttbar_json,
            iov=args.iov,
            redirector=redirector,
            maxfiles=args.maxfiles,
            samples=args.sample or ['QCD', 'TTbar'],
            data_json=args.data_json,
        )
        input_label = f'manifest:{redirector}'

    if not fileset:
        sys.exit('No MC files found for the selected input mode')

    print(f"IOV {args.iov} — env={args.env}, input={input_label}, datasets={len(fileset)}")
    for ds, spec in fileset.items():
        print(f"  {ds:22s} {len(spec['files']):4d} file(s)  "
              f"xsec={spec['metadata'].get('xsec_pb')}")

    proc = TopTagWPProcessor(iov=args.iov)  # match auto-detected per dataset

    tic = time.time()
    use_dask = args.dask or args.env in ('lpc', 'casa')
    client = cluster = None
    try:
        if use_dask:
            quiet_dask_worker_logs()
            client, cluster = start_dask_client(args, os.getcwd())
            runner = processor.Runner(
                metadata_cache={},
                executor=processor.DaskExecutor(client=client, retries=2, treereduction=20),
                schema=NanoAODSchema,
                chunksize=args.chunksize_dask,
                maxchunks=args.maxchunks,
                skipbadfiles=True,
                savemetrics=True,
            )
        else:
            runner = processor.Runner(
                executor=processor.FuturesExecutor(workers=args.workers),
                schema=NanoAODSchema,
                chunksize=args.chunksize,
                maxchunks=args.maxchunks,
                skipbadfiles=True,
                savemetrics=True,
            )
        output, metrics = runner(fileset, treename="Events", processor_instance=proc)
    finally:
        close_dask(client, cluster)

    dt = time.time() - tic

    # attach metadata + provenance for the derivation script
    output['datasets_metadata'] = {ds: spec['metadata'] for ds, spec in fileset.items()}
    output['run_info'] = {
        'iov': args.iov,
        'env': args.env,
        'input_mode': input_mode,
        'input': input_label,
        'lumi_pb': LUMI_PB.get(args.iov),
        'tagger_label': proc._cfg['label'],
        'chunksize': args.chunksize_dask if use_dask else args.chunksize,
        'maxchunks': args.maxchunks,
        'maxfiles': args.maxfiles,
        'executor': 'dask' if use_dask else 'futures',
    }

    util.save(output, out)
    print(f"\nsaved {out}")
    print(f"elapsed {dt:.1f}s, events/s = {metrics['entries']/dt:.0f}")
    for ds in fileset:
        raw = output.get('nevents_raw', {}).get(ds, output['nevents'][ds])
        rejected = output.get('qcd_genweight_rejected', {}).get(ds, 0)
        print(f"  {ds:20s} nevents={output['nevents'][ds]:>10d}/{raw:<10d}  "
              f"sumw={output['sumw'][ds]:.3e}  qcd_weight_reject={rejected}")


if __name__ == '__main__':
    main()
