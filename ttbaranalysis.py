# ttbaranalysis.py

from coffea import util
from coffea.nanoevents import NanoAODSchema
import coffea.processor as processor

import itertools
import argparse
import time
from datetime import date
import json
import os

from dask.distributed import Client

import warnings
warnings.filterwarnings("ignore")

default_datastets = ['data', 'TTbar', 'QCD']
default_signals = ['RSGluon', 'ZPrime10', 'ZPrime30', 'ZPrimeDM', 'ZPrime1']

from ttbarprocessor import TTbarResProcessor
from python.functions import printTime, makeSaveDirectories


def _build_sample_metadata(sample, subsection, iov, metadata):
    sample_metadata = {
        'sample': sample,
        'subsample': subsection or sample,
        'year': iov,
        'is_mc': not (('data' in sample.lower()) or ('singlemu' in sample.lower())),
    }
    sample_metadata.update(metadata)
    return sample_metadata


def _output_subsection(sample, subsection):
    if sample == 'QCD' and subsection and subsection.startswith('QCD_'):
        return subsection.removeprefix('QCD_')
    return subsection


def _parse_manifest_entry(sample, subsection, iov, entry):
    if isinstance(entry, dict) and 'files' in entry:
        files = entry['files']
        metadata = dict(entry.get('metadata', {}))
    else:
        files = entry
        metadata = {}

    return list(files), _build_sample_metadata(sample, subsection, iov, metadata)


def _collect_manifest_sections(sample, iov, manifest, subsections):
    iov_entry = manifest[iov]

    if isinstance(iov_entry, dict) and 'files' not in iov_entry:
        requested_sections = subsections if subsections else list(iov_entry.keys())
        entries = []
        for subsection in requested_sections:
            if subsection not in iov_entry:
                print(f'{subsection} not in {sample} {iov}')
                continue
            files, metadata = _parse_manifest_entry(sample, subsection, iov, iov_entry[subsection])
            entries.append((subsection, files, metadata))
        return entries

    files, metadata = _parse_manifest_entry(sample, '', iov, iov_entry)
    return [('', files, metadata)]


if __name__ == "__main__":

    tic = time.time()

    savedir = 'outputs/dy/'

    parser = argparse.ArgumentParser(
        prog='ttbaranalysis.py',
        description='Run ttbarprocessor',
        epilog='e.g. python ttbaranalysis.py --test --dataset TTbar --iov 2024')

    # datasets
    parser.add_argument('-d', '--dataset',
                        choices=['data', 'QCD', 'TTbar', 'ZPrime1', 'ZPrime10',
                                 'ZPrime30', 'ZPrimeDM', 'RSGluon', 'ZPrimeLocal'],
                        default=default_datastets, action='append')
    parser.add_argument('--iov', choices=['2022', '2023', '2024'], default='2024')
    parser.add_argument('--signals', action='store_true', help='run only signal samples')

    # subsections
    parser.add_argument('--era', choices=['A','B','C','D','E','F','G','H','I'],
                        action='append', default=[],
                        help='--era A --era B for multiple eras; runs all if omitted')
    parser.add_argument('-p', '--pt', choices=['700to1000', '1000toInf'],
                        action='append', default=[])
    parser.add_argument('-m', '--mass', action='append', default=[])
    parser.add_argument('--subsample', action='append', default=[],
                        help='run specific manifest subsection(s), e.g. --subsample QCD_PT-1000to1500')

    # analysis options
    parser.add_argument('--blind',    action='store_true', help='process 1/10th of the data')
    parser.add_argument('--bkgest',   choices=['2dalphabet', 'mistag'], default=None)
    parser.add_argument('--toptagger',choices=['deepak8', 'cmsv2', 'recomb'], default='deepak8',
                        help="'recomb' = learned per-pT GloParTv3 recombination top-tagger")
    parser.add_argument('--recomb-weights', default='data/recomb/recomb_deploy_2024.json',
                        help='deploy JSON for --toptagger recomb (build_recomb_deploy.py)')
    parser.add_argument('-r', '--redirector', default='root://cmsxrootd.fnal.gov/')
    parser.add_argument('--ttagWP',   choices=['loose', 'medium', 'tight'], default='medium')
    parser.add_argument('--btagger',  choices=['deepcsv', 'csvv2'], default='deepcsv')
    parser.add_argument('--ht',       choices=['1400', '950'], default='1400')
    parser.add_argument('--noSyst',   action='store_true', help='run without systematics')
    parser.add_argument('--ntuple',   action='store_true', help='collect flat ntuple in output')

    # run options
    parser.add_argument('--dask',      action='store_true')
    parser.add_argument('--env',       choices=['casa', 'lpc', 'winterfell', 'local', 'C', 'L', 'W'], default='lpc')
    parser.add_argument('--test',      action='store_true')
    parser.add_argument('-n', '--nocluster', action='store_true')

    args = parser.parse_args()

    if len(args.dataset) > len(default_datastets):
        args.dataset = args.dataset[len(default_datastets):]
    if args.signals:
        args.dataset = default_signals

    if args.dask and (args.env in ('lpc', 'L')):
        from lpcjobqueue import LPCCondorCluster

    ##### parameters #####
    samples           = args.dataset
    IOV               = args.iov
    useDeepAK8        = args.toptagger in ('deepak8', 'recomb')
    useDeepCSV        = args.btagger == 'deepcsv'
    htCut             = 1400.0 if args.ht == '1400' else 950.0
    dask_memory       = '5GB'
    chunksize_dask    = 100000
    chunksize_futures = 200000
    nworkers          = 1 if args.test else 4
    maxchunks         = 1 if args.test else None

    ##### systematics #####
    systematics = ['nominal', 'jes', 'jer', 'pileup', 'pdf', 'q2', 'ttag_pt1']
    if '2016' in IOV or '2017' in IOV:
        systematics.append('prefiring')
    if args.bkgest == '2dalphabet':
        systematics.append('transferFunction')

    ##### analysis categories #####
    ttagcats = ["at", "2t"]
    ycats    = ["cen", "fwd"]
    anacats  = [t + y for t, y in itertools.product(ttagcats, ycats)]
    label_map = {i: label for i, label in enumerate(anacats)}

    with open('out.log', 'w') as f:
        print('\n' + date.today().isoformat(), file=f)
        print('\n------args------', file=f)
        for k, v in vars(args).items(): print(k, '=', v, file=f)
        print('categories =', label_map, file=f)
        if not args.noSyst: print('systematics =', systematics, file=f)

    print('\n------args------')
    for k, v in vars(args).items(): print(k, '=', v)
    if not args.noSyst: print('systematics =', systematics)
    print('----------------\n')

    ##### redirector #####
    if args.env in ('casa', 'C'):
        redirector = 'root://xcache/'
    elif args.env in ('winterfell', 'W'):
        redirector = '/mnt/data/cms/'
    else:
        redirector = args.redirector

    jsonfiles = {
        "data":       'data/nanoAOD/data.json',
        "QCD":        'data/nanoAOD/QCD.json',
        "TTbar":      'data/nanoAOD/TTbar.json',
        "ZPrime1":    'data/nanoAOD/ZPrime1.json',
        "ZPrime10":   'data/nanoAOD/ZPrime10.json',
        "ZPrime30":   'data/nanoAOD/ZPrime30.json',
        "ZPrimeDM":   'data/nanoAOD/ZPrimeDM.json',
        "RSGluon":    'data/nanoAOD/RSGluon.json',
        "ZPrimeLocal":'data/nanoAOD/local.json',
    }

    upload_to_dask = ['data', 'python', 'ttbarprocessor.py']

    if not os.path.exists(savedir):
        os.makedirs(savedir)
        os.makedirs(savedir + 'logs/')
        os.makedirs(savedir + 'scale/')
        os.makedirs(savedir + 'twodalphabet/')
        os.popen('cp ttbarprocessor.py ' + savedir + 'logs/ttbarprocessor_'
                 + date.today().isoformat().replace('-', '') + '.py')
        os.popen('cat out.log >> ' + savedir + 'logs/ttbarprocessor_diff.txt')
    else:
        for f in os.listdir(savedir + 'logs/'):
            if 'ttbarprocessor' in f and f.endswith('.py'):
                os.popen('cat out.log >> ' + savedir + 'logs/ttbarprocessor_diff.txt')
                os.popen('diff ttbarprocessor.py ' + savedir + 'logs/' + f
                         + ' >> ' + savedir + 'logs/ttbarprocessor_diff.txt')
        if not os.path.exists(savedir + 'twodalphabet/'):
            os.makedirs(savedir + 'twodalphabet/')

    makeSaveDirectories(coffea_dir=savedir)

    output  = None
    metrics = None

    for sample in samples:
        skipbadfiles = False
        inputfile = jsonfiles[sample]

        with open(inputfile) as json_file:
            subsections = args.era + args.mass + args.pt + args.subsample
            manifest = json.load(json_file)
            sections = _collect_manifest_sections(
                sample=sample,
                iov=IOV,
                manifest=manifest,
                subsections=subsections,
            )

            for subsection, files, sample_metadata in sections:
                files = [redirector + f for f in files]
                if args.test:
                    files = [files[int(len(files) / 2)]]

                fileset = {
                    sample: {
                        'files': files,
                        'metadata': sample_metadata,
                    }
                }
                print(files[0])

                output_subsection = _output_subsection(sample, subsection)
                subString = f'_{output_subsection}' if output_subsection else ''
                if args.bkgest:
                    subString += '_bkgest'
                if (args.toptagger == 'cmsv2') and (args.btagger == 'csvv2'):
                    savedir = 'outputs/oldanalysis/'

                savefilename = f'{savedir}{sample}_{IOV}{subString}.coffea'
                if 'RSGluon' in sample:
                    subString = subString.replace(output_subsection, '')
                    savefilename = f'{savedir}{sample}{subsection}_{IOV}{subString}.coffea'
                elif 'ZPrime' in sample:
                    subString = subString.replace(output_subsection, '')
                    savefilename = f'{savedir}ZPrime{subsection}_{sample.replace("ZPrime","")}_{IOV}{subString}.coffea'

                print(f'running {IOV} {sample} {subsection}')

                if args.toptagger == 'cmsv2':
                    savefilename = savefilename.replace('.coffea', '_cmsv2.coffea')
                if args.toptagger == 'recomb':
                    savefilename = savefilename.replace('.coffea', '_recomb.coffea')
                if args.btagger == 'csvv2':
                    savefilename = savefilename.replace('.coffea', '_csvv2.coffea')
                if args.ht == '950':
                    savefilename = savefilename.replace('.coffea', '_ht950.coffea')
                if args.blind:
                    savefilename = savefilename.replace('.coffea', '_blind.coffea')
                if args.noSyst:
                    savefilename = savefilename.replace('.coffea', '_noSyst.coffea')
                if args.ntuple:
                    savefilename = savefilename.replace('.coffea', '_ntuple.coffea')
                if args.test:
                    savefilename = savefilename.replace('.coffea', '_test.coffea')

                processor_instance = TTbarResProcessor(
                    iov=IOV,
                    bkgEst=args.bkgest,
                    noSyst=args.noSyst,
                    deepAK8Cut=args.ttagWP,
                    useDeepAK8=useDeepAK8,
                    useDeepCSV=useDeepCSV,
                    topTagger=args.toptagger,
                    recomb_weights=(args.recomb_weights if args.toptagger == 'recomb' else None),
                    htCut=htCut,
                    anacats=anacats,
                    systematics=systematics,
                    blinding=args.blind,
                    produce_ntuple=args.ntuple,
                    sample_metadata=sample_metadata,
                )

                if not args.dask:
                    runner = processor.Runner(
                        executor=processor.FuturesExecutor(workers=nworkers),
                        schema=NanoAODSchema,
                        chunksize=chunksize_futures,
                        maxchunks=maxchunks,
                        skipbadfiles=skipbadfiles,
                        xrootdtimeout=500,
                        savemetrics=True,
                    )
                    output, metrics = runner(
                        fileset, treename="Events",
                        processor_instance=processor_instance,
                    )

                else:
                    if args.env in ('lpc', 'L'):
                        cluster = None if args.nocluster else LPCCondorCluster(
                            memory=dask_memory,
                            transfer_input_files=upload_to_dask,
                            scheduler_options={"dashboard_address": ":8787"},
                        )
                        if cluster:
                            cluster.adapt(minimum=1, maximum=100)
                    else:
                        cluster = None

                    with Client(cluster) as client:
                        run_instance = processor.Runner(
                            metadata_cache={},
                            executor=processor.DaskExecutor(client=client, retries=12),
                            schema=NanoAODSchema,
                            savemetrics=True,
                            skipbadfiles=skipbadfiles,
                            chunksize=chunksize_dask,
                            maxchunks=maxchunks,
                        )
                        worker_toc = time.time()
                        print("Waiting for at least one worker...")
                        client.wait_for_workers(1)
                        print(f'time to wait for worker = {int(time.time() - worker_toc)}s')

                        output, metrics = run_instance(
                            fileset, treename="Events",
                            processor_instance=processor_instance,
                        )
                        client.shutdown()
                        del cluster

                output['analysisCategories'] = label_map
                util.save(output, savefilename)
                print('saving', savefilename)

    elapsed = time.time() - tic
    printTime(elapsed)
    if metrics is not None:
        print(f"Events/s: {metrics['entries'] / elapsed:.0f}")
