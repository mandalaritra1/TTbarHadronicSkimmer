"""Shared run_analysis(args) used by both ttbaranalysis.py (CLI) and ttbaranalysis.ipynb."""
import itertools
import json
import logging
import os
import subprocess
import time
import traceback
from datetime import date

import dask
from coffea import util
from coffea.nanoevents import NanoAODSchema
import coffea.processor as processor

from ttbarprocessor import TTbarResProcessor
from python.functions import printTime, makeSaveDirectories
from python.ntuple_utils import (
    _ntuple_paths_for_coffea,
    _normalize_ntuple_base_dir,
    _default_ntuple_base_dir,
    _merge_ntuple_chunks,
    _write_lpc_ntuple_merge_instructions,
    _write_accumulated_ntuple,
)
from python.dask_resources import (
    _start_dask_resources,
    _close_dask_resources,
    _check_dask_ntuple_chunk_visibility,
)


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _build_sample_metadata(sample, subsection, iov, metadata):
    sample_metadata = {
        "sample": sample,
        "subsample": subsection or sample,
        "year": iov,
        "is_mc": not (("data" in sample.lower()) or ("singlemu" in sample.lower())),
    }
    sample_metadata.update(metadata)
    return sample_metadata


def _output_subsection(sample, subsection):
    if sample == "QCD" and subsection and subsection.startswith("QCD_"):
        return subsection.removeprefix("QCD_")
    return subsection


def _parse_manifest_entry(sample, subsection, iov, entry):
    if isinstance(entry, dict) and "files" in entry:
        files = entry["files"]
        metadata = dict(entry.get("metadata", {}))
    else:
        files = entry
        metadata = {}

    return list(files), _build_sample_metadata(sample, subsection, iov, metadata)


def _collect_manifest_sections(sample, iov, manifest, subsections):
    iov_entry = manifest[iov]

    if isinstance(iov_entry, dict) and "files" not in iov_entry:
        requested_sections = subsections if subsections else list(iov_entry.keys())
        entries = []
        for subsection in requested_sections:
            if subsection not in iov_entry:
                print(f"{subsection} not in {sample} {iov}")
                continue
            files, metadata = _parse_manifest_entry(
                sample, subsection, iov, iov_entry[subsection]
            )
            entries.append((subsection, files, metadata))
        return entries

    files, metadata = _parse_manifest_entry(sample, "", iov, iov_entry)
    return [("", files, metadata)]


def _format_section_label(iov, sample, subsection):
    return f"{iov} {sample} {subsection}".strip()


def _print_runner_block(lines, rule_char="-", width=66):
    print(rule_char * width)
    for line in lines:
        print(line)
    print(rule_char * width)


def _archive_existing_output(path, tag="old"):
    if not os.path.exists(path):
        return None

    root, ext = os.path.splitext(path)
    archive_path = f"{root}_{tag}{ext}"

    os.replace(path, archive_path)
    return archive_path


def _quiet_dask_logging():
    dask.config.set({"logging.distributed": "error"})
    for name in [
        "distributed",
        "distributed.scheduler",
        "distributed.core",
        "distributed.nanny",
        "distributed.worker",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def run_analysis(args):
    """Run the ttbar processor over the requested samples/sections.

    `args` may come from argparse (CLI) or types.SimpleNamespace (notebook).
    Returns a summary dict with elapsed time, metrics, last output, and lists
    of saved/skipped/failed sections.
    """
    _quiet_dask_logging()

    tic = time.time()

    savedir = "outputs/dy/"

    samples = args.dataset
    IOV = args.iov
    useDeepAK8 = args.toptagger == "deepak8"
    useDeepCSV = args.btagger == "deepcsv"
    htCut = 1400.0 if args.ht == "1400" else 950.0
    dask_memory = f"{int(args.daskMemory)}GB"
    chunksize_dask = 100 if args.test else 100000
    chunksize_futures = 100 if args.test else 200000
    maxchunks = 10 if args.test else None

    systematics = [
        "nominal",
        "jes",
        "jer",
        "pileup",
        "pdf",
        "q2",
        "ttag_pt1",
    ]

    if ("2016" in IOV) or ("2017" in IOV):
        systematics.append("prefiring")

    if args.bkgest == "2dalphabet":
        systematics.append("transferFunction")

    ttagcats = ["at", "2t"]
    ycats = ["cen", "fwd"]

    anacats = [t + y for t, y in itertools.product(ttagcats, ycats)]
    label_map = {i: label for i, label in enumerate(anacats)}

    with open("out.log", "w") as f:
        print("\n" + date.today().isoformat(), file=f)
        print("categories =", label_map, file=f)
        print("\n", file=f)
        if not args.noSyst:
            print("systematics =", systematics, file=f)

    print("\n------args------")
    for argname, value in vars(args).items():
        print(argname, "=", value)
    if not args.noSyst:
        print("systematics =", systematics)
    print("----------------\n")

    redirector = args.redirector

    jsonfiles = {
        "data": "data/nanoAOD/data.json",
        "QCD": "data/nanoAOD/QCD.json",
        "QCD_flat": "data/nanoAOD/QCD_flat.json",
        "TTbar": "data/nanoAOD/TTbar.json",
        "ZPrime1": "data/nanoAOD/ZPrime1.json",
        "ZPrime10": "data/nanoAOD/ZPrime10.json",
        "ZPrime30": "data/nanoAOD/ZPrime30.json",
        "ZPrimeDM": "data/nanoAOD/ZPrimeDM.json",
        "RSGluon": "data/nanoAOD/RSGluon.json",
        "ZPrimeLocal": "data/nanoAOD/local_xsec_test.json",
    }

    repo_root = os.path.abspath(os.getcwd())
    upload_to_dask = ["data", "python", "ttbarprocessor.py"]

    if not os.path.exists(savedir):
        os.makedirs(savedir)
        os.makedirs(savedir + "logs/")
        os.makedirs(savedir + "scale/")
        os.makedirs(savedir + "twodalphabet/")
        subprocess.run(
            [
                "cp",
                "ttbarprocessor.py",
                savedir
                + "logs/ttbarprocessor_"
                + date.today().isoformat().replace("-", "")
                + ".py",
            ],
            check=True,
        )
        subprocess.run(
            f"cat out.log >> {savedir}logs/ttbarprocessor_diff.txt",
            shell=True,
            check=True,
        )
    else:
        for f in os.listdir(savedir + "logs/"):
            if "ttbarprocessor" in f and "py" in f:
                subprocess.run(
                    f"cat out.log >> {savedir}logs/ttbarprocessor_diff.txt",
                    shell=True,
                    check=True,
                )
                diff_result = subprocess.run(
                    ["diff", "ttbarprocessor.py", savedir + "logs/" + f],
                    capture_output=True,
                    text=True,
                )
                with open(savedir + "logs/ttbarprocessor_diff.txt", "a") as df:
                    df.write(diff_result.stdout)

        if not os.path.exists(savedir + "scale/"):
            os.makedirs(savedir + "scale/")
        if not os.path.exists(savedir + "twodalphabet/"):
            os.makedirs(savedir + "twodalphabet/")

    makeSaveDirectories(coffea_dir=savedir)

    output = None
    metrics = None
    savefilenames = []
    skipped_outputs = []
    failures = []
    nworkers = 1 if args.test else 4

    # ── Dask cluster/client: created once and reused across all samples ────────
    client = None
    cluster = None
    client, cluster = _start_dask_resources(
        args=args,
        repo_root=repo_root,
        upload_to_dask=upload_to_dask,
        dask_memory=dask_memory,
        nworkers=nworkers,
    )

    for sample_index, sample in enumerate(samples):
        skipbadfiles = False
        inputfile = jsonfiles[sample]

        with open(inputfile) as json_file:
            subsections = (
                args.era + args.mass + args.pt + getattr(args, "subsample", [])
            )
            manifest = json.load(json_file)
            sections = _collect_manifest_sections(
                sample=sample,
                iov=IOV,
                manifest=manifest,
                subsections=subsections,
            )

            for section_index, (subsection, files, sample_metadata) in enumerate(
                sections
            ):
                files = [redirector + f for f in files]
                if args.test:
                    files = [files[int(len(files) / 2)]]
                    maxchunks = 1

                fileset = {
                    sample: {
                        "files": files,
                        "metadata": sample_metadata,
                    }
                }

                print(files[0])

                output_subsection = _output_subsection(sample, subsection)
                subString = f"_{output_subsection}" if output_subsection else ""
                if args.bkgest:
                    subString += "_bkgest"

                if (args.toptagger == "cmsv2") and (args.btagger == "csvv2"):
                    savedir = "outputs/oldanalysis/"

                savefilename = f"{savedir}{sample}_{IOV}{subString}.coffea"
                if "RSGluon" in sample:
                    subString = subString.replace(output_subsection, "")
                    savefilename = (
                        f"{savedir}{sample}{subsection}_{IOV}{subString}.coffea"
                    )
                elif "ZPrime" in sample:
                    subString = subString.replace(output_subsection, "")
                    savefilename = f'{savedir}ZPrime{subsection}_{sample.replace("ZPrime", "")}_{IOV}{subString}.coffea'
                print(f"running {IOV} {sample} {subsection}")

                if args.toptagger == "cmsv2":
                    savefilename = savefilename.replace(".coffea", "_cmsv2.coffea")
                if args.btagger == "csvv2":
                    savefilename = savefilename.replace(".coffea", "_csvv2.coffea")
                if args.ht == "950":
                    savefilename = savefilename.replace(".coffea", "_ht950.coffea")
                if args.blind:
                    savefilename = savefilename.replace(".coffea", "_blind.coffea")
                if args.noSyst:
                    savefilename = savefilename.replace(".coffea", "_noSyst.coffea")
                if args.test:
                    savefilename = savefilename.replace(".coffea", "_test.coffea")

                section_label = _format_section_label(IOV, sample, subsection)
                if section_index + 1 < len(sections):
                    next_label = _format_section_label(
                        IOV, sample, sections[section_index + 1][0]
                    )
                elif sample_index + 1 < len(samples):
                    next_label = f"next sample {samples[sample_index + 1]}"
                else:
                    next_label = "end of requested run"

                if os.path.exists(savefilename) and not args.overwrite:
                    _print_runner_block(
                        [
                            f"output already present: {savefilename}",
                            f"skipping {section_label}",
                        ]
                    )
                    skipped_outputs.append((savefilename, sample, subsection))
                    try:
                        output = util.load(savefilename)
                    except Exception as load_error:
                        print(
                            f"warning: could not load skipped output {savefilename}: {load_error}"
                        )
                    continue
                elif os.path.exists(savefilename) and args.overwrite:
                    archived_output = _archive_existing_output(savefilename)
                    _print_runner_block(
                        [
                            f"archived existing output: {archived_output}",
                            f"new output will use: {savefilename}",
                        ]
                    )

                try:
                    ntuple_mode = args.ntupleStorage if args.ntuple else "accumulator"
                    ntuple_base_dir = (
                        _normalize_ntuple_base_dir(args.ntupleBaseDir)
                        or _default_ntuple_base_dir(args, ntuple_mode)
                    )
                    ntuple_run_id = f"{int(time.time())}_{sample_index}_{section_index}"
                    _, ntuple_chunk_dir = _ntuple_paths_for_coffea(
                        savefilename, ntuple_run_id, ntuple_base_dir
                    )

                    if args.ntuple and args.dask and ntuple_mode == "chunks":
                        if args.env == "lpc":
                            print(f"LPC ntuple chunks will be written under: {ntuple_chunk_dir}")
                        _check_dask_ntuple_chunk_visibility(client, ntuple_chunk_dir)

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
                            fileset,
                            treename="Events",
                            processor_instance=TTbarResProcessor(
                                iov=IOV,
                                bkgEst=args.bkgest,
                                noSyst=args.noSyst,
                                deepAK8Cut=args.ttagWP,
                                useDeepAK8=useDeepAK8,
                                useDeepCSV=useDeepCSV,
                                htCut=htCut,
                                anacats=anacats,
                                systematics=systematics,
                                blinding=args.blind,
                                debug=True,
                                produce_ntuple=args.ntuple,
                                ntuple_mode=ntuple_mode,
                                ntuple_output_dir=ntuple_chunk_dir if ntuple_mode == "chunks" else None,
                                ntuple_tree_name=sample,
                                ntuple_columns=args.ntupleContent,
                                sample_metadata=sample_metadata,
                            ),
                        )
                    else:
                        run_instance = processor.Runner(
                            metadata_cache={},
                            executor=processor.DaskExecutor(client=client, retries=2, treereduction=20),
                            schema=NanoAODSchema,
                            savemetrics=True,
                            skipbadfiles=skipbadfiles,
                            chunksize=chunksize_dask,
                            maxchunks=maxchunks,
                        )

                        output, metrics = run_instance(
                            fileset,
                            treename="Events",
                            processor_instance=TTbarResProcessor(
                                iov=IOV,
                                bkgEst=args.bkgest,
                                noSyst=args.noSyst,
                                deepAK8Cut=args.ttagWP,
                                useDeepAK8=useDeepAK8,
                                useDeepCSV=useDeepCSV,
                                htCut=htCut,
                                anacats=anacats,
                                systematics=systematics,
                                blinding=args.blind,
                                produce_ntuple=args.ntuple,
                                ntuple_mode=ntuple_mode,
                                ntuple_output_dir=ntuple_chunk_dir if ntuple_mode == "chunks" else None,
                                ntuple_tree_name=sample,
                                ntuple_columns=args.ntupleContent,
                                sample_metadata=sample_metadata,
                            ),
                        )

                    output["analysisCategories"] = label_map
                    util.save(output, savefilename)
                    print("saving", savefilename)
                    if args.ntuple:
                        if ntuple_mode == "chunks":
                            if args.env == "lpc":
                                n_chunks = len(set(output.get("ntuple_chunks", [])))
                                instructions_file, merged_root_file = _write_lpc_ntuple_merge_instructions(
                                    savefilename, sample, ntuple_base_dir, n_chunks
                                )
                                print(
                                    f"wrote {n_chunks} ntuple chunks to EOS; "
                                    "merging this section before continuing"
                                )
                                print(f"merge instructions: {instructions_file}")
                                merged_root_file, n_chunks = _merge_ntuple_chunks(
                                    savefilename, sample, ntuple_base_dir
                                )
                                print(
                                    f"merged {n_chunks} ntuple chunks: "
                                    f"{merged_root_file}"
                                )
                            else:
                                merged_root_file, n_chunks = _merge_ntuple_chunks(
                                    savefilename, sample, ntuple_base_dir
                                )
                                print(
                                    f"merged {n_chunks} ntuple chunks: "
                                    f"{merged_root_file}"
                                )
                        else:
                            merged_root_file = _write_accumulated_ntuple(savefilename, sample)
                            print(f"wrote accumulated ntuple: {merged_root_file}")
                    savefilenames.append((savefilename, sample))
                except Exception as exc:
                    failures.append(
                        {
                            "sample": sample,
                            "subsection": subsection,
                            "savefilename": savefilename,
                            "error": repr(exc),
                        }
                    )
                    _print_runner_block(
                        [
                            f"crashed during {section_label}",
                            f"next queued section: {next_label}",
                            "",
                            traceback.format_exc().rstrip(),
                        ]
                    )
                    if args.dask:
                        _print_runner_block(
                            [
                                "restarting Dask client after section failure",
                                f"will retry scheduling from {next_label}",
                            ]
                        )
                        client, cluster = _close_dask_resources(client, cluster)
                        client, cluster = _start_dask_resources(
                            args=args,
                            repo_root=repo_root,
                            upload_to_dask=upload_to_dask,
                            dask_memory=dask_memory,
                            nworkers=nworkers,
                        )
                    continue

    elapsed = time.time() - tic
    printTime(elapsed)
    if metrics is not None:
        print(f"Events/s: {metrics['entries'] / elapsed:.0f}")

    print(
        "run summary:",
        f"saved={len(savefilenames)}",
        f"skipped={len(skipped_outputs)}",
        f"failed={len(failures)}",
    )

    client, cluster = _close_dask_resources(client, cluster)

    return {
        "elapsed": elapsed,
        "metrics": metrics,
        "output": output,
        "savefilenames": savefilenames,
        "skipped_outputs": skipped_outputs,
        "failures": failures,
    }
