"""Size the condor Dask worker pool from a fileset's chunk count, per submit.

A fixed pool (cluster.scale(N)) on coffea-casa / LPC HTCondor fails two ways:
  1. workers that reach their condor walltime retire and are not resubmitted, so
     a long run drains to zero workers and the driver blocks without an error;
  2. a pool much larger than the task graph (400 workers, ~100 chunks) makes dask
     hoard the root tasks on one worker while the rest sit idle.
adapt(minimum=n, maximum=n) keeps a fixed footprint but resubmits retired workers.
n = min(cap, max(POOL_MIN_WORKERS, nchunks // POOL_CHUNKS_PER_WORKER)) is re-set
before every fileset (re-calling adapt on a live cluster works), because sample
bins differ by orders of magnitude in chunk count.

Ported from smp_jetmass_run2 notebook_utils (commit aafb4fa), with the header scan
run in threads and skipped when the file count alone already reaches the cap.
This does not fix the heavy_input hoarding stall; that is _BroadcastHeavyClient.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor

POOL_CHUNKS_PER_WORKER = 2
POOL_MIN_WORKERS = 4
SCAN_THREADS = 32


def _iter_files(fileset, treename="Events"):
    """(path, treename) for every file of a coffea fileset: {ds: [paths]},
    {ds: {path: treename}} or the {ds: {"files": ..., "metadata": ...}} form."""
    for entry in fileset.values():
        if isinstance(entry, dict) and "files" in entry:
            entry = entry["files"]
        if isinstance(entry, str):
            entry = [entry]
        if isinstance(entry, dict):
            for path, tree in entry.items():
                yield path, tree if isinstance(tree, str) else treename
        else:
            for path in entry:
                yield path, treename


def _file_nchunks(item, chunksize):
    import uproot

    path, treename = item
    try:
        with uproot.open(path) as handle:
            return math.ceil(handle[treename].num_entries / chunksize), False
    except Exception:
        return 1, True


def estimate_nchunks(fileset, chunksize, treename="Events", threads=SCAN_THREADS, log=print):
    """Chunk count of a fileset: sum over files of ceil(num_entries / chunksize).

    Reads only each file's header (num_entries, no branches), in parallel threads:
    a serial loop took ~4 min for 640 remote files over xcache. A file that cannot
    be opened counts as one chunk; the estimate only sets the pool size.
    Returns (nchunks, nfiles).
    """
    files = list(_iter_files(fileset, treename))
    with ThreadPoolExecutor(max_workers=max(1, min(threads, len(files)))) as pool:
        results = list(pool.map(lambda item: _file_nchunks(item, chunksize), files))
    nbad = sum(bad for _, bad in results)
    if nbad:
        log(f"[pool] {nbad} file(s) unreadable for num_entries; counted as 1 chunk each")
    return sum(n for n, _ in results), len(files)


def _pool_size(client):
    try:
        return len(client.nthreads())
    except Exception:
        return "?"


def size_pool_for_fileset(client, fileset, chunksize, cap=None, label="",
                          treename="Events", threads=SCAN_THREADS, log=print):
    """Re-target the cluster to adapt(n, n) for this fileset before it is submitted.

    Returns the requested worker count, or None when there is nothing to size:
    no client (futures/iterative executor), a cluster without .adapt, or a
    LocalCluster (local runs keep their fixed nworkers).
    """
    cluster = getattr(client, "cluster", None) if client is not None else None
    if cluster is None or not hasattr(cluster, "adapt"):
        return None
    from distributed import LocalCluster

    if isinstance(cluster, LocalCluster):
        return None

    nfiles = sum(1 for _ in _iter_files(fileset, treename))
    before = _pool_size(client)
    if cap and nfiles // POOL_CHUNKS_PER_WORKER >= cap:
        # every file is at least one chunk, so the scan cannot change the answer
        n_req = cap
        summary = f"{nfiles} files >= {POOL_CHUNKS_PER_WORKER} x cap {cap}, header scan skipped"
    else:
        tic = time.time()
        nchunks, nfiles = estimate_nchunks(fileset, chunksize, treename, threads=threads, log=log)
        n_req = max(POOL_MIN_WORKERS, nchunks // POOL_CHUNKS_PER_WORKER)
        if cap:
            n_req = min(cap, n_req)
        summary = (f"{nfiles} files, ~{nchunks} chunks @ chunksize={chunksize} "
                   f"(header scan {time.time() - tic:.1f}s, {min(threads, max(nfiles, 1))} threads)")
    cluster.adapt(minimum=n_req, maximum=n_req)
    log(f"[pool] {label}: {summary} -> adapt({n_req}, {n_req})  [pool was {before}]")
    return n_req
