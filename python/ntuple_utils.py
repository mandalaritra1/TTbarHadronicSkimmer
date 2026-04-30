"""Ntuple I/O and chunk-merge helpers shared by the notebook and CLI."""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import uproot

from coffea import util


LPC_EOS_XROOTD_PREFIX = "root://cmseos.fnal.gov//store/user/amandal2"
LPC_EOS_MOUNT_PREFIX = "/eos/uscms/store/user/amandal2"
LPC_DEFAULT_NTUPLE_BASE_DIR = f"{LPC_EOS_XROOTD_PREFIX}/TTbarHadronicSkimmer/ntuples"


def _is_xrootd_path(path):
    return str(path).startswith("root://")


def _xrootd_url_parts(path):
    prefix, remote_path = str(path).split("//", 1)
    host, store_path = remote_path.split("/", 1)
    return f"{prefix}//{host}", f"/{store_path.lstrip('/')}"


def _xrootd_parent(path):
    server, store_path = _xrootd_url_parts(path)
    return server, os.path.dirname(store_path)


def _xrootd_exists(path):
    server, store_path = _xrootd_url_parts(path)
    return subprocess.run(
        ["xrdfs", server, "stat", store_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _copy_to_xrootd(local_path, remote_path):
    server, remote_dir = _xrootd_parent(remote_path)
    subprocess.run(["xrdfs", server, "mkdir", "-p", remote_dir], check=True)
    subprocess.run(["xrdcp", "-f", local_path, remote_path], check=True)


def _stage_output_path(output_file):
    if not _is_xrootd_path(output_file):
        return output_file, None

    tmp_dir = tempfile.mkdtemp(prefix="ttbar_ntuple_merge_")
    return os.path.join(tmp_dir, os.path.basename(output_file)), tmp_dir


def _finish_output_path(local_output, final_output, tmp_dir):
    try:
        if _is_xrootd_path(final_output):
            _copy_to_xrootd(local_output, final_output)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def load_branches(coffea_file):
    output = util.load(coffea_file)
    ntuple = output["ntuple"]
    return {col: ntuple[col].value for col in ntuple.keys()}


def merge_branches(branch_list):
    keys = branch_list[0].keys()
    return {col: np.concatenate([b[col] for b in branch_list]) for col in keys}


def write_ntuple(coffea_files, root_file, tree_name="ttbar"):
    all_branches = []
    for f in coffea_files:
        print(f"Loading {f} ...")
        all_branches.append(load_branches(f))

    branches = merge_branches(all_branches) if len(all_branches) > 1 else all_branches[0]

    n_events = len(next(iter(branches.values())))
    print(f"Writing {n_events} events, {len(branches)} branches → {root_file}:{tree_name}")

    local_root_file, tmp_dir = _stage_output_path(root_file)
    with uproot.recreate(local_root_file) as f:
        f.mktree(tree_name, {col: arr.dtype for col, arr in branches.items()})
        f[tree_name].extend(branches)
    _finish_output_path(local_root_file, root_file, tmp_dir)

    print("Done.")


def merge_root_ntuples(root_files, output_file, tree_name="ttbar", weight_scale=1.0):
    root_files = list(root_files)
    if not root_files:
        print(f"No ntuple chunk files found for {output_file}; skipping merge.", flush=True)
        return 0

    n_events = 0
    tree_created = False
    local_output_file, tmp_dir = _stage_output_path(output_file)
    with uproot.recreate(local_output_file) as fout:
        for root_file in root_files:
            print(f"Merging {root_file} ...", flush=True)
            with uproot.open(root_file) as fin:
                tree = fin[tree_name]
                arrays = tree.arrays(library="np")

            if weight_scale != 1.0 and "weight" in arrays:
                arrays["weight"] = (arrays["weight"] * weight_scale).astype(arrays["weight"].dtype)

            if not tree_created:
                fout.mktree(tree_name, {col: arr.dtype for col, arr in arrays.items()})
                tree_created = True

            fout[tree_name].extend(arrays)
            n_events += len(next(iter(arrays.values()))) if arrays else 0

    _finish_output_path(local_output_file, output_file, tmp_dir)
    print(f"Wrote {n_events} events from {len(root_files)} chunks → {output_file}:{tree_name}", flush=True)
    return n_events


def _normalize_ntuple_base_dir(chunk_base_dir):
    chunk_base_dir = chunk_base_dir.strip()
    if not chunk_base_dir:
        return ""
    if chunk_base_dir.startswith(LPC_EOS_MOUNT_PREFIX):
        suffix = chunk_base_dir.removeprefix(LPC_EOS_MOUNT_PREFIX).lstrip("/")
        return os.path.join(LPC_EOS_XROOTD_PREFIX, suffix)
    return chunk_base_dir


def _default_ntuple_base_dir(args, ntuple_mode):
    if args.ntuple and ntuple_mode == "chunks" and args.env == "lpc":
        return LPC_DEFAULT_NTUPLE_BASE_DIR
    return ""


def _ntuple_paths_for_coffea(coffea_file, run_id, chunk_base_dir=""):
    chunk_base_dir = _normalize_ntuple_base_dir(chunk_base_dir)
    ntuple_dir = chunk_base_dir if chunk_base_dir else os.path.join(os.path.dirname(coffea_file), "ntuples")
    root_file = os.path.join(
        ntuple_dir, os.path.basename(coffea_file).replace(".coffea", "_ntuple.root")
    )
    chunk_parent = chunk_base_dir if _is_xrootd_path(chunk_base_dir) else os.path.abspath(chunk_base_dir) if chunk_base_dir else ntuple_dir
    chunk_dir = os.path.join(
        chunk_parent,
        "chunks",
        os.path.basename(coffea_file).replace(".coffea", f"_{run_id}"),
    )
    return root_file, chunk_dir if _is_xrootd_path(chunk_dir) else os.path.abspath(chunk_dir)


def _merge_ntuple_chunks(coffea_file, tree_name, ntuple_base_dir=""):
    output = util.load(coffea_file)
    chunk_files = sorted(set(output.get("ntuple_chunks", [])))
    root_file, _ = _ntuple_paths_for_coffea(coffea_file, "merged", ntuple_base_dir)
    if not _is_xrootd_path(root_file):
        os.makedirs(os.path.dirname(root_file), exist_ok=True)
    missing = [
        path for path in chunk_files
        if not (_xrootd_exists(path) if _is_xrootd_path(path) else os.path.exists(path))
    ]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} of {len(chunk_files)} ntuple chunk files are not visible "
            f"to this notebook process. First missing paths:\n{preview}\n\n"
            "This usually means Coffea-Casa workers wrote chunks on worker-local "
            "storage. Set 'Ntuple dir' to a filesystem path that is shared between "
            "the notebook and Dask workers, then rerun the section."
        )
    scale = float(output.get("normalization", {}).get("scale_factor", 1.0))
    merge_root_ntuples(chunk_files, root_file, tree_name=tree_name, weight_scale=scale)
    return root_file, len(chunk_files)


def _write_lpc_ntuple_merge_instructions(coffea_file, tree_name, ntuple_base_dir, n_chunks):
    root_file, _ = _ntuple_paths_for_coffea(coffea_file, "merged", ntuple_base_dir)
    instructions_file = coffea_file.replace(".coffea", "_merge_instructions.txt")
    lines = [
        "LPC ntuple chunk merge instructions",
        "====================================",
        "",
        "The chunk ROOT files were already written locally on workers and copied to EOS with xrdcp.",
        "Do not merge from /eos/uscms and do not write the merged ROOT file directly through /eos/uscms.",
        "",
        f"Coffea file with ntuple_chunks URLs: {coffea_file}",
        f"TTree name: {tree_name}",
        f"Number of chunk files: {n_chunks}",
        f"Final merged ROOT output: {root_file}",
        "",
        "From an LPC interactive node, open this notebook, run the import/helper cells, then run:",
        "",
        f"_merge_ntuple_chunks({coffea_file!r}, {tree_name!r}, {ntuple_base_dir!r})",
        "",
        "That helper reads the chunk URLs from the .coffea file, writes the merged ROOT file to local",
        "temporary storage first, and only then copies the completed file back to EOS with xrdcp.",
    ]
    with open(instructions_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    return instructions_file, root_file


def _write_accumulated_ntuple(coffea_file, tree_name):
    root_file, _ = _ntuple_paths_for_coffea(coffea_file, "accumulated")
    os.makedirs(os.path.dirname(root_file), exist_ok=True)
    write_ntuple([coffea_file], root_file, tree_name=tree_name)
    return root_file
