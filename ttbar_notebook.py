"""Interactive run backend for ttbaranalysis.ipynb.

Holds the widgets, config persistence, manifest/signal/ntuple helpers and
run_analysis, so the notebook only shows the interface. Use:
    from ttbar_notebook import *
    show_widgets()                # the widget UI
    args = build_args(); run_analysis(args)
"""

from coffea import util
from coffea.nanoevents import NanoAODSchema, BaseSchema
import coffea.processor as processor

import itertools
import time
from datetime import date
import json
import os
from types import SimpleNamespace

from dask.distributed import Client, performance_report, Security

import warnings

warnings.filterwarnings("ignore")
import logging
import dask

dask.config.set({"logging.distributed": "error"})

for name in [
    "distributed",
    "distributed.scheduler",
    "distributed.core",
    "distributed.nanny",
    "distributed.worker",
]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

default_datastets = ["data", "TTbar", "QCD"]
default_signals = ["RSGluon", "ZPrime10", "ZPrime30", "ZPrimeDM", "ZPrime1"]


class _BroadcastHeavyClient:
    """DaskExecutor ships the pickled processor (heavy_input) as a single future
    on ONE worker via client.submit, and every chunk task depends on it -- the
    scheduler then piles all chunks onto that worker (dependency locality) and
    work stealing won't move the big object. Seen as "all tasks on one worker"
    on a hot pool between consecutive runs (2026-07-23). Bare .submit is only
    used for heavy_input inside DaskExecutor.__call__ (chunks go through .map),
    so replicating every submitted future to the whole pool restores fan-out."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def submit(self, func, *args, **kwargs):
        future = self._inner.submit(func, *args, **kwargs)
        try:
            from distributed import wait as _dask_wait

            _dask_wait(future, timeout=300)
            self._inner.replicate(future)
        except Exception:
            pass
        return future

from ttbarprocessor import TTbarResProcessor
from python.functions import printTime, makeSaveDirectories, xs as _XS_TABLE

# ── Widgets for interactive configuration ─────────────────────────────────────
import ipywidgets as widgets
from IPython.display import clear_output, display
import json, os

CONFIG_FILE = ".last_config.json"

for _old_widget in list(globals().get("WIDGETS", {}).values()) + [
    globals().get("btn_reset"),
]:
    if _old_widget is not None:
        try:
            _old_widget.close()
        except Exception:
            pass
clear_output(wait=True)

# casa Dask pool size (fixed; see _start_dask_resources for why not adaptive)
CASA_WORKERS = 64

DEFAULTS = dict(
    dataset=["ZPrimeLocal"],
    signals=False,
    iov="2024",
    subsample=[],
    mass="",
    blind=False,
    bkgest=None,
    toptagger="topvsqcd",
    ttag_ptbinned=False,
    redirector="rootfiles/",
    ttagWP="tight",   # 0.5% QCD-mistag WP (2026-06-25 meeting decision)
    btagger="deepcsv",
    ht="1400",
    noSyst=False,
    ntuple=False,
    ntupleContent="slim",
    ntupleStorage="chunks",
    ntupleBaseDir="",
    overwrite=False,
    dask=False,
    daskMemory=5,
    env="lpc",
    test=False,
    nocluster=False,
    progress=False,
    outdir="",
    signal_batch=4,
)


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULTS, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULTS)


cfg = load_config()

style = {"description_width": "80px"}
layout = widgets.Layout(width="210px")
layout_wide = widgets.Layout(width="260px")

_dataset_opts = [
    "data",
    "QCD",
    "QCD_flat",
    "TTbar",
    "ZPrime1",
    "ZPrime10",
    "ZPrime30",
    "ZPrimeDM",
    "RSGluon",
    "ZPrimeLocal",
]
_redirector_opts = [
    ("Local (rootfiles/)", "rootfiles/"),
    ("FNAL XRootD (root://cmsxrootd.fnal.gov/)", "root://cmsxrootd.fnal.gov/"),
    ("CMS xcache (root://xcache/)", "root://xcache/"),
    ("Winterfell (/mnt/data/cms/)", "/mnt/data/cms/"),
]
_redirector_vals = [v for _, v in _redirector_opts]
_env_opts = ["casa", "lpc", "winterfell", "local"]
_iov_opts = ["2022", "2023", "2024", "2025"]
_bkgest_opts = [("None", None), "2dalphabet", "mistag"]
_toptagger_opts = ["topvsqcd", "cmsv2", "recomb"]
_ttagWP_opts = ["loose", "medium", "tight"]
_btagger_opts = ["deepcsv", "csvv2"]
_ht_opts = ["1400", "950"]
_manifest_files = {
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


def _option_values(options):
    return [option[1] if isinstance(option, tuple) else option for option in options]


def _valid_choice(value, options, default):
    values = _option_values(options)
    return value if value in values else default


def _valid_multi(values, options, default=()):
    allowed = set(_option_values(options))
    selected = [value for value in (values or []) if value in allowed]
    if selected:
        return tuple(selected)
    return tuple(value for value in default if value in allowed)


def _manifest_subsections(dataset, iov):
    path = _manifest_files.get(dataset)
    if not path or not os.path.exists(path):
        return []

    try:
        with open(path) as f:
            manifest = json.load(f)
    except Exception:
        return []

    entry = manifest.get(iov)
    if isinstance(entry, dict) and "files" not in entry:
        return list(entry.keys())
    return []


def _available_subsamples(datasets, iov):
    subsamples = []
    seen = set()
    for dataset in datasets:
        for subsection in _manifest_subsections(dataset, iov):
            if subsection not in seen:
                seen.add(subsection)
                subsamples.append(subsection)
    return subsamples


_initial_datasets = _valid_multi(cfg["dataset"], _dataset_opts, DEFAULTS["dataset"])
_initial_iov = _valid_choice(cfg["iov"], _iov_opts, DEFAULTS["iov"])
_initial_subsample_datasets = (
    list(default_signals) if cfg["signals"] else list(_initial_datasets)
)
_initial_subsample_options = _available_subsamples(
    _initial_subsample_datasets, _initial_iov
)
_initial_subsamples = _valid_multi(cfg["subsample"], _initial_subsample_options)


# ── Widget definitions ─────────────────────────────────────────────────────────
w_dataset = widgets.SelectMultiple(
    options=_dataset_opts,
    value=_initial_datasets,
    description="Dataset",
    style=style,
    layout=widgets.Layout(width="210px", height="150px"),
)
w_signals = widgets.Checkbox(
    value=cfg["signals"], description="Signals only", style=style, layout=layout
)
w_iov = widgets.Dropdown(
    options=_iov_opts,
    value=_initial_iov,
    description="IOV",
    style=style,
    layout=layout,
)
w_subsample = widgets.SelectMultiple(
    options=_initial_subsample_options,
    value=_initial_subsamples,
    description="Subsample",
    style=style,
    layout=widgets.Layout(width="260px", height="140px"),
)
w_mass = widgets.Text(
    value=cfg["mass"],
    placeholder="e.g. 1000,2000",
    description="Mass pts",
    style=style,
    layout=layout,
)
w_blind = widgets.Checkbox(
    value=cfg["blind"], description="Blind", style=style, layout=layout
)
w_bkgest = widgets.Dropdown(
    options=_bkgest_opts,
    value=_valid_choice(cfg["bkgest"], _bkgest_opts, DEFAULTS["bkgest"]),
    description="Bkg est",
    style=style,
    layout=layout,
)
w_toptagger = widgets.Dropdown(
    options=_toptagger_opts,
    value=_valid_choice(cfg["toptagger"], _toptagger_opts, DEFAULTS["toptagger"]),
    description="Top tagger",
    style=style,
    layout=layout,
)
w_redirector = widgets.Dropdown(
    options=_redirector_opts,
    value=cfg["redirector"] if cfg["redirector"] in _redirector_vals else "rootfiles/",
    description="Redirector",
    style=style,
    layout=layout_wide,
)
w_ttagWP = widgets.Dropdown(
    options=_ttagWP_opts,
    value=_valid_choice(cfg["ttagWP"], _ttagWP_opts, DEFAULTS["ttagWP"]),
    description="ttag WP",
    style=style,
    layout=layout,
)
w_btagger = widgets.Dropdown(
    options=_btagger_opts,
    value=_valid_choice(cfg["btagger"], _btagger_opts, DEFAULTS["btagger"]),
    description="B tagger",
    style=style,
    layout=layout,
)
w_ht = widgets.Dropdown(
    options=_ht_opts,
    value=_valid_choice(cfg["ht"], _ht_opts, DEFAULTS["ht"]),
    description="HT cut",
    style=style,
    layout=layout,
)
w_ttag_ptbinned = widgets.Checkbox(
    value=cfg.get("ttag_ptbinned", False), description="ptbinned baseline WP", style=style, layout=layout
)
w_noSyst = widgets.Checkbox(
    value=cfg["noSyst"], description="No syst", style=style, layout=layout
)
w_ntuple = widgets.Checkbox(
    value=cfg["ntuple"], description="Ntuple", style=style, layout=layout
)
w_ntupleContent = widgets.Dropdown(
    options=[("Slim 2DAlphabet", "slim"), ("Full diagnostics", "full")],
    value=cfg["ntupleContent"] if cfg["ntupleContent"] in {"slim", "full"} else "slim",
    description="Ntuple cols",
    style=style,
    layout=layout_wide,
)
w_ntupleStorage = widgets.Dropdown(
    options=[("Write chunks", "chunks"), ("Accumulate memory", "accumulator")],
    value=cfg["ntupleStorage"] if cfg["ntupleStorage"] in {"chunks", "accumulator"} else "chunks",
    description="Ntuple mode",
    style=style,
    layout=layout_wide,
)
w_ntupleBaseDir = widgets.Text(
    value=cfg["ntupleBaseDir"],
    placeholder="optional shared chunk directory",
    description="Ntuple dir",
    style=style,
    layout=widgets.Layout(width="430px"),
)
w_overwrite = widgets.Checkbox(
    value=cfg["overwrite"], description="Overwrite", style=style, layout=layout
)
w_dask = widgets.Checkbox(
    value=cfg["dask"], description="Dask", style=style, layout=layout
)
w_daskMemory = widgets.IntSlider(
    value=int(cfg.get("daskMemory", DEFAULTS["daskMemory"])),
    min=1,
    max=20,
    step=1,
    description="Dask GB",
    style=style,
    layout=layout_wide,
    continuous_update=False,
)
w_env = widgets.Dropdown(
    options=_env_opts,
    value=cfg["env"] if cfg["env"] in _env_opts else "lpc",
    description="Env",
    style=style,
    layout=layout,
)
w_test = widgets.Checkbox(
    value=cfg["test"], description="Test", style=style, layout=layout
)
w_nocluster = widgets.Checkbox(
    value=cfg["nocluster"], description="No cluster", style=style, layout=layout
)
w_progress = widgets.Checkbox(
    value=cfg["progress"], description="Progress", style=style, layout=layout
)
w_outdir = widgets.Text(
    value=cfg["outdir"],
    placeholder="outputs/<name>/  (blank = dy)",
    description="Out dir",
    style=style,
    layout=layout_wide,
)
w_signal_batch = widgets.BoundedIntText(
    value=int(cfg.get("signal_batch", DEFAULTS["signal_batch"])),
    min=1,
    max=100,
    step=1,
    description="Sig batch",
    style=style,
    layout=layout,
)

# ── Central widget registry ────────────────────────────────────────────────────
# To add a new config field: add it to DEFAULTS above and WIDGETS below.
# save_config, build_args, and reset_to_defaults all derive from this dict.
WIDGETS = {
    "dataset": w_dataset,
    "signals": w_signals,
    "iov": w_iov,
    "subsample": w_subsample,
    "mass": w_mass,
    "blind": w_blind,
    "bkgest": w_bkgest,
    "toptagger": w_toptagger,
    "ttag_ptbinned": w_ttag_ptbinned,
    "redirector": w_redirector,
    "ttagWP": w_ttagWP,
    "btagger": w_btagger,
    "ht": w_ht,
    "noSyst": w_noSyst,
    "ntuple": w_ntuple,
    "ntupleContent": w_ntupleContent,
    "ntupleStorage": w_ntupleStorage,
    "ntupleBaseDir": w_ntupleBaseDir,
    "overwrite": w_overwrite,
    "dask": w_dask,
    "daskMemory": w_daskMemory,
    "env": w_env,
    "test": w_test,
    "nocluster": w_nocluster,
    "progress": w_progress,
    "outdir": w_outdir,
    "signal_batch": w_signal_batch,
}

_MULTI = widgets.SelectMultiple


def _widget_value(w):
    return list(w.value) if isinstance(w, _MULTI) else w.value


def save_config(_=None):
    cfg = {k: _widget_value(w) for k, w in WIDGETS.items()}
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def reset_to_defaults(_):
    for key, w in WIDGETS.items():
        default = DEFAULTS[key]
        w.value = tuple(default) if isinstance(w, _MULTI) else default


def refresh_subsample_options(_=None):
    selected_datasets = (
        list(default_signals) if w_signals.value else list(w_dataset.value)
    )
    options = _available_subsamples(selected_datasets, w_iov.value)
    current = [v for v in w_subsample.value if v in options]
    w_subsample.options = options
    w_subsample.value = tuple(current)


for w in WIDGETS.values():
    w.observe(save_config, names="value")

for w in (w_dataset, w_iov, w_signals):
    w.observe(refresh_subsample_options, names="value")

refresh_subsample_options()

btn_reset = widgets.Button(
    description="↺ Reset to Defaults",
    button_style="warning",
    layout=widgets.Layout(width="160px", margin="8px 0 0 0"),
)
btn_reset.on_click(reset_to_defaults)



from types import SimpleNamespace


def build_args():
    selected_datasets = list(w_dataset.value)
    if w_signals.value:
        selected_datasets = list(default_signals)

    raw_mass = w_mass.value.strip()
    mass_list = []
    if raw_mass:
        parts = [m.strip() for m in raw_mass.split(",")]
        invalid = [p for p in parts if not p.isdigit()]
        if invalid:
            print(f"Warning: invalid mass entries ignored: {invalid}")
        mass_list = [p for p in parts if p.isdigit()]

    cfg = {k: _widget_value(w) for k, w in WIDGETS.items()}
    cfg["dataset"] = selected_datasets
    cfg["era"] = []
    cfg["pt"] = []
    cfg["mass"] = mass_list
    return SimpleNamespace(**cfg)


import subprocess
import traceback
from write_ntuple import merge_root_ntuples
from write_ntuple import write_ntuple


def _build_sample_metadata(sample, subsection, iov, metadata):
    sample_metadata = {
        "sample": sample,
        "subsample": subsection or sample,
        "year": iov,
        "is_mc": not (("data" in sample.lower()) or ("singlemu" in sample.lower())),
    }
    # `iov` (the run year being processed) is authoritative for normalization lumi,
    # so a manifest entry's own 'year' must NOT override it. This matters for the MC
    # stand-in: 2024 Summer24 files served for --iov 2025 carry year='2024' in their
    # metadata, but we want them normalized to the 2025 lumi (_LUMI_PB['2025']).
    sample_metadata.update({k: v for k, v in metadata.items() if k != "year"})
    return sample_metadata


def _output_subsection(sample, subsection):
    if sample == "QCD" and subsection and subsection.startswith("QCD_"):
        return subsection.removeprefix("QCD_")
    return subsection


def _signal_dataset_key(sample, subsection):
    if "RSGluon" in sample:
        return f"RSGluon{subsection}"
    if "ZPrime" in sample:
        return f'ZPrime{subsection}_{sample.replace("ZPrime", "")}'
    return subsection or sample


def _signal_xsec(sample, subsection):
    """1 pb reference xsec (pb) for a signal mass point, from functions.xs."""
    try:
        return _XS_TABLE.get(sample, {}).get(str(subsection))
    except Exception:
        return None


def _batch_tag(subsections):
    """Filename tag for a batch of signal masses: '_<lo>to<hi>' (or '_<mass>')."""
    nums = [int(s) for s in subsections if str(s).isdigit()]
    if nums:
        lo, hi = min(nums), max(nums)
        return f"_{lo}" if lo == hi else f"_{lo}to{hi}"
    return "_" + "-".join(str(s) for s in subsections)[:40]


def _parse_manifest_entry(sample, subsection, iov, entry):
    if isinstance(entry, dict) and "files" in entry:
        files = entry["files"]
        metadata = dict(entry.get("metadata", {}))
    else:
        files = entry
        metadata = {}

    return list(files), _build_sample_metadata(sample, subsection, iov, metadata)


def _collect_manifest_sections(sample, iov, manifest, subsections, source_iov=None):
    # source_iov: read file lists from this manifest key while still labelling the
    # sample with `iov` in its metadata (used to stand in 2024 Summer24 MC for an
    # IOV that has no v15 production of its own, e.g. 2025 TTbar/QCD/signal).
    iov_entry = manifest[source_iov or iov]

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


def _dask_write_visibility_probe(path):
    probe_path = os.path.join(path, f"worker_probe_{os.getpid()}.txt")
    if _is_xrootd_path(probe_path):
        import tempfile

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
        if not args.nocluster:
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
        # Fixed pool instead of adapt(4, 400): adaptive scaling on a free condor
        # pool mass-spawns then churns workers; the distributed scheduler's
        # handle_request_refresh_who_has KeyError race fires continuously under
        # that churn and can crash the scheduler mid-run ("Client lost the
        # connection to the scheduler", 2026-07-23). A fixed-size pool is stable.
        cluster.scale(CASA_WORKERS)
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
    try:
        print(f"[dask] dashboard : {client.dashboard_link}")
        print(f"[dask] scheduler : {client.scheduler.address}")
        print("[dask] to retire a stuck worker, from ANOTHER notebook/kernel:")
        print("       from dask_gateway import Gateway; g = Gateway()")
        print("       c = g.connect(g.list_clusters()[0].name).get_client()")
        print("       c.retire_workers(['tls://HOST:PORT'], close_workers=True)")
    except Exception as _e:
        print(f"[dask] could not report scheduler/dashboard: {_e}")

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


def run_analysis(args):
    tic = time.time()

    savedir = f"outputs/dy/"
    if args.outdir:
        savedir = "outputs/" + args.outdir.strip("/") + "/"

    samples = args.dataset
    IOV = args.iov
    useDeepAK8 = args.toptagger in ("topvsqcd", "recomb")
    useDeepCSV = args.btagger == "deepcsv"
    htCut = 1400.0 if args.ht == "1400" else 950.0
    dask_memory = f"{int(args.daskMemory)}GB"
    chunksize_dask = 100000
    chunksize_futures = 200000
    maxchunks = 10 if args.test else None

    systematics = [
        "nominal",
        "jes",
        "jer",
        "pileup",
        "pdf",
        "q2",
        "ttag_pt1",
        #'ttag_pt2',
        #'ttag_pt3'
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
            # MC fallback: an IOV with no v15 production of its own (e.g. 2025, and
            # signal for the 2022/2023 sub-eras) reads file lists from 2024 Summer24
            # while keeping the requested IOV in metadata. Data is never substituted.
            source_iov = None
            if sample != "data" and IOV not in manifest and "2024" in manifest:
                source_iov = "2024"
                print(f"[placeholder] {sample} {IOV}: no v15 MC for this IOV -> using 2024 Summer24 MC as stand-in")
            sections = _collect_manifest_sections(
                sample=sample,
                iov=IOV,
                manifest=manifest,
                subsections=subsections,
                source_iov=source_iov,
            )

            file_savedir = savedir
            if (args.toptagger == "cmsv2") and (args.btagger == "csvv2") and not args.outdir:
                file_savedir = "outputs/oldanalysis/"
            if args.test:
                maxchunks = 1

            # signals: ONE grouped job per width (masses on the dataset axis,
            # each scaled to 1 pb); everything else one job per subsection.
            is_signal = sample.startswith("ZPrime") or sample == "RSGluon"
            run_units = []  # (fileset, dataset_metadata, group_flag, subsection, savefilename)
            if is_signal:
                # batch masses into groups of args.signal_batch -> one file per batch
                # (middle ground: all-in-one OOMs the merge, per-mass is slow).
                batch_size = max(1, int(getattr(args, "signal_batch", 4)))
                built = []  # (subsection, ds_key, files, sample_metadata)
                for subsection, files, sample_metadata in sections:
                    files = [redirector + f for f in files]
                    if args.test:
                        files = [files[int(len(files) / 2)]]
                    ds_key = _signal_dataset_key(sample, subsection)
                    if sample_metadata.get("xsec_pb") is None:
                        _x = _signal_xsec(sample, subsection)
                        if _x is not None:
                            sample_metadata = {**sample_metadata, "xsec_pb": _x}
                    built.append((subsection, ds_key, files, sample_metadata))
                for bi in range(0, len(built), batch_size):
                    batch = built[bi:bi + batch_size]
                    gfs = {dk: {"files": f, "metadata": m} for (_, dk, f, m) in batch}
                    gm = {dk: m for (_, dk, f, m) in batch}
                    tag = _batch_tag([s for (s, _, _, _) in batch])
                    print(f"{sample} batch{tag}: {[dk for (_, dk, _, _) in batch]}")
                    run_units.append((gfs, gm, True, tag.lstrip('_'), f"{file_savedir}{sample}_{IOV}{tag}.coffea"))
            else:
                for subsection, files, sample_metadata in sections:
                    files = [redirector + f for f in files]
                    if args.test:
                        files = [files[int(len(files) / 2)]]
                    print(files[0])
                    output_subsection = _output_subsection(sample, subsection)
                    subString = f"_{output_subsection}" if output_subsection else ""
                    if args.bkgest:
                        subString += "_bkgest"
                    run_units.append(({sample: {"files": files, "metadata": sample_metadata}}, {sample: sample_metadata}, False, subsection, f"{file_savedir}{sample}_{IOV}{subString}.coffea"))

            for section_index, (fileset, dataset_metadata, group_flag, subsection, savefilename) in enumerate(run_units):
                print(f"running {IOV} {sample} {subsection} -> {savefilename}")

                if args.toptagger == "cmsv2":
                    savefilename = savefilename.replace(".coffea", "_cmsv2.coffea")
                if args.toptagger == "recomb":
                    savefilename = savefilename.replace(".coffea", "_recomb.coffea")
                if args.toptagger == "topvsqcd" and getattr(args, "ttag_ptbinned", False):
                    savefilename = savefilename.replace(".coffea", "_ptbin.coffea")
                if args.toptagger == "topvsqcd" and not getattr(args, "ttag_ptbinned", False):
                    savefilename = savefilename.replace(".coffea", "_topvsqcd.coffea")
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
                            executor=processor.FuturesExecutor(workers=nworkers, status=args.progress),
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
                                topTagger=args.toptagger,
                                recomb_weights=("data/recomb/recomb_deploy_2024.json" if args.toptagger == "recomb" else ("data/recomb/baseline_deploy_2024.json" if (args.toptagger == "topvsqcd" and getattr(args, "ttag_ptbinned", False)) else None)),
                                htCut=htCut,
                                anacats=anacats,
                                systematics=systematics,
                                blinding=args.blind,
                                debug=True,
                                cutflow_verbose=args.test,
                                produce_ntuple=args.ntuple,
                                ntuple_mode=ntuple_mode,
                                ntuple_output_dir=ntuple_chunk_dir if ntuple_mode == "chunks" else None,
                                ntuple_tree_name=sample,
                                ntuple_columns=args.ntupleContent,
                                sample_metadata=(next(iter(dataset_metadata.values())) if len(dataset_metadata) == 1 else {}),
                                dataset_metadata=dataset_metadata,
                                group_by_dataset=group_flag,
                            ),
                        )
                    else:
                        run_instance = processor.Runner(
                            metadata_cache={},
                            executor=processor.DaskExecutor(client=_BroadcastHeavyClient(client), retries=12, treereduction=6, status=args.progress),
                            schema=NanoAODSchema,
                            savemetrics=True,
                            skipbadfiles=skipbadfiles,
                            chunksize=chunksize_dask,
                            maxchunks=maxchunks,
                            xrootdtimeout=600,
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
                                topTagger=args.toptagger,
                                recomb_weights=("data/recomb/recomb_deploy_2024.json" if args.toptagger == "recomb" else ("data/recomb/baseline_deploy_2024.json" if (args.toptagger == "topvsqcd" and getattr(args, "ttag_ptbinned", False)) else None)),
                                htCut=htCut,
                                anacats=anacats,
                                systematics=systematics,
                                blinding=args.blind,
                                cutflow_verbose=args.test,
                                produce_ntuple=args.ntuple,
                                ntuple_mode=ntuple_mode,
                                ntuple_output_dir=ntuple_chunk_dir if ntuple_mode == "chunks" else None,
                                ntuple_tree_name=sample,
                                ntuple_columns=args.ntupleContent,
                                sample_metadata=(next(iter(dataset_metadata.values())) if len(dataset_metadata) == 1 else {}),
                                dataset_metadata=dataset_metadata,
                                group_by_dataset=group_flag,
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

def show_widgets():
    # ── Display ───────────────────────────────────────────────────────────────────
    _loaded = (
        "restored from last session" if os.path.exists(CONFIG_FILE) else "using defaults"
    )
    print(f"Config {_loaded}; changes save to {CONFIG_FILE}.")
    print("Datasets")
    display(w_dataset)
    display(w_signals)
    display(w_iov)
    print("Subsections")
    display(w_subsample)
    display(w_mass)
    print("Analysis options")
    for _widget in (
        w_blind,
        w_bkgest,
        w_toptagger,
        w_ttag_ptbinned,
        w_redirector,
        w_ttagWP,
        w_btagger,
        w_ht,
        w_noSyst,
        w_ntuple,
        w_ntupleContent,
        w_ntupleStorage,
        w_ntupleBaseDir,
        w_overwrite,
    ):
        display(_widget)
    print("Run options")
    for _widget in (w_dask, w_daskMemory, w_env, w_test, w_nocluster, w_progress, w_outdir, w_signal_batch, btn_reset):
        display(_widget)
    print("Adjust widgets above, then run the next cell to apply settings.")