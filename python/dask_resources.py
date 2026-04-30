"""Dask cluster lifecycle and worker-visibility helpers."""
import os
import subprocess
import tempfile

import dask
import dask.distributed
from dask.distributed import Client, Security

from python.ntuple_utils import (
    _is_xrootd_path,
    _xrootd_url_parts,
    _xrootd_exists,
    _copy_to_xrootd,
)


def _dask_write_visibility_probe(path):
    probe_path = os.path.join(path, f"worker_probe_{os.getpid()}.txt")
    if _is_xrootd_path(probe_path):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("worker wrote this file\n")
            local_probe_path = f.name
        try:
            _copy_to_xrootd(local_probe_path, probe_path)
        finally:
            os.remove(local_probe_path)
        return probe_path

    os.makedirs(path, exist_ok=True)
    with open(probe_path, "w") as f:
        f.write("worker wrote this file\n")
    return probe_path


def _check_dask_ntuple_chunk_visibility(client, chunk_dir):
    if client is None:
        return

    probe_dir = os.path.join(chunk_dir, "_visibility_probe")
    probe_path = client.submit(_dask_write_visibility_probe, probe_dir).result()
    probe_visible = _xrootd_exists(probe_path) if _is_xrootd_path(probe_path) else os.path.exists(probe_path)
    if not probe_visible:
        raise RuntimeError(
            "Dask worker chunk output is not visible from this notebook.\n"
            f"Worker reported writing: {probe_path}\n\n"
            "Set 'Ntuple dir' to a shared filesystem location visible to both "
            "Coffea-Casa workers and the notebook, then rerun. The default "
            "repository-local outputs directory is not shared in this session."
        )
    if _is_xrootd_path(probe_path):
        server, store_path = _xrootd_url_parts(probe_path)
        subprocess.run(["xrdfs", server, "rm", store_path], check=False)
    else:
        os.remove(probe_path)


def _close_dask_resources(client, cluster):
    if client is not None:
        client.close()
    if cluster is not None:
        cluster.close()
    return None, None


def _start_dask_resources(args, repo_root, upload_to_dask, dask_memory, nworkers):
    client = None
    cluster = None

    if not args.dask:
        return client, cluster

    if args.nocluster:
        cluster = dask.distributed.LocalCluster(
            n_workers=nworkers,
            threads_per_worker=1,
            scheduler_port=0,
            dashboard_address=":8787",
            protocol="tcp://",
            security=Security(),
        )
    elif args.env == "lpc":
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            memory=dask_memory,
            transfer_input_files=upload_to_dask,
            scheduler_options={"dashboard_address": ":8787"},
        )
        cluster.adapt(minimum=1, maximum=100)
    elif args.env == "casa":
        from coffea_casa import CoffeaCasaCluster

        cluster = CoffeaCasaCluster(memory=dask_memory)
        cluster.adapt(minimum=4, maximum=400)
    else:
        cluster = dask.distributed.LocalCluster(
            n_workers=nworkers,
            threads_per_worker=1,
            scheduler_port=0,
            dashboard_address=":8787",
            protocol="tcp://",
            security=Security(),
        )

    client = Client(cluster)

    if args.env == "casa" and not args.nocluster:
        from distributed.diagnostics.plugin import UploadDirectory

        client.register_worker_plugin(
            UploadDirectory(
                os.path.join(repo_root, "data"), restart=True, update_path=True
            ),
            nanny=True,
        )
        client.register_worker_plugin(
            UploadDirectory(
                os.path.join(repo_root, "python"), restart=True, update_path=True
            ),
            nanny=True,
        )
        client.upload_file(os.path.join(repo_root, "ttbarprocessor.py"))

    return client, cluster
