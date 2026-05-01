"""Interactive ipywidgets-based configuration UI for ttbaranalysis.ipynb.

Usage in the notebook:

    from python.interactive_config import build_ui, build_args
    ui = build_ui()           # renders widgets
    args = build_args(ui)     # later, after the user has set values
    run_analysis(args)
"""
import json
import os
import shlex
from types import SimpleNamespace

import ipywidgets as widgets
from IPython.display import clear_output, display


CONFIG_FILE = ".last_config.json"

DEFAULT_DATASETS = ["data", "TTbar", "QCD"]
DEFAULT_SIGNALS = ["RSGluon", "ZPrime10", "ZPrime30", "ZPrimeDM", "ZPrime1"]

DEFAULTS = dict(
    dataset=["ZPrimeLocal"],
    signals=False,
    iov="2024",
    subsample=[],
    mass="",
    blind=False,
    bkgest=None,
    toptagger="deepak8",
    redirector="rootfiles/",
    ttagWP="medium",
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
    chunksize=0,
    env="lpc",
    test=False,
    nocluster=False,
    cliOnly=False,
)

_DATASET_OPTS = [
    "data", "QCD", "QCD_flat", "TTbar",
    "ZPrime1", "ZPrime10", "ZPrime30", "ZPrimeDM",
    "RSGluon", "ZPrimeLocal",
]
_REDIRECTOR_OPTS = [
    ("Local (rootfiles/)", "rootfiles/"),
    ("FNAL XRootD (root://cmsxrootd.fnal.gov/)", "root://cmsxrootd.fnal.gov/"),
    ("CMS xcache (root://xcache/)", "root://xcache/"),
    ("Winterfell (/mnt/data/cms/)", "/mnt/data/cms/"),
]
_REDIRECTOR_VALS = [v for _, v in _REDIRECTOR_OPTS]
_ENV_OPTS = ["casa", "lpc", "winterfell", "local"]
_IOV_OPTS = ["2022", "2023", "2024"]
_BKGEST_OPTS = [("None", None), "2dalphabet", "mistag"]
_TOPTAGGER_OPTS = ["deepak8", "cmsv2"]
_TTAGWP_OPTS = ["loose", "medium", "tight"]
_BTAGGER_OPTS = ["deepcsv", "csvv2"]
_HT_OPTS = ["1400", "950"]
_MANIFEST_FILES = {
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


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULTS, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULTS)


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
    path = _MANIFEST_FILES.get(dataset)
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


def _widget_value(w):
    return list(w.value) if isinstance(w, widgets.SelectMultiple) else w.value


def build_ui():
    """Construct widgets, render the UI, and return a dict {name: widget}."""
    cfg = load_config()

    style = {"description_width": "80px"}
    layout = widgets.Layout(width="210px")
    layout_wide = widgets.Layout(width="260px")

    initial_datasets = _valid_multi(cfg["dataset"], _DATASET_OPTS, DEFAULTS["dataset"])
    initial_iov = _valid_choice(cfg["iov"], _IOV_OPTS, DEFAULTS["iov"])
    initial_subsample_datasets = (
        list(DEFAULT_SIGNALS) if cfg["signals"] else list(initial_datasets)
    )
    initial_subsample_options = _available_subsamples(
        initial_subsample_datasets, initial_iov
    )
    initial_subsamples = _valid_multi(cfg["subsample"], initial_subsample_options)

    W = {}
    W["dataset"] = widgets.SelectMultiple(
        options=_DATASET_OPTS, value=initial_datasets,
        description="Dataset", style=style,
        layout=widgets.Layout(width="210px", height="150px"),
    )
    W["signals"] = widgets.Checkbox(value=cfg["signals"], description="Signals only", style=style, layout=layout)
    W["iov"] = widgets.Dropdown(options=_IOV_OPTS, value=initial_iov, description="IOV", style=style, layout=layout)
    W["subsample"] = widgets.SelectMultiple(
        options=initial_subsample_options, value=initial_subsamples,
        description="Subsample", style=style,
        layout=widgets.Layout(width="260px", height="140px"),
    )
    W["mass"] = widgets.Text(
        value=cfg["mass"], placeholder="e.g. 1000,2000",
        description="Mass pts", style=style, layout=layout,
    )
    W["blind"] = widgets.Checkbox(value=cfg["blind"], description="Blind", style=style, layout=layout)
    W["bkgest"] = widgets.Dropdown(
        options=_BKGEST_OPTS,
        value=_valid_choice(cfg["bkgest"], _BKGEST_OPTS, DEFAULTS["bkgest"]),
        description="Bkg est", style=style, layout=layout,
    )
    W["toptagger"] = widgets.Dropdown(
        options=_TOPTAGGER_OPTS,
        value=_valid_choice(cfg["toptagger"], _TOPTAGGER_OPTS, DEFAULTS["toptagger"]),
        description="Top tagger", style=style, layout=layout,
    )
    W["redirector"] = widgets.Dropdown(
        options=_REDIRECTOR_OPTS,
        value=cfg["redirector"] if cfg["redirector"] in _REDIRECTOR_VALS else "rootfiles/",
        description="Redirector", style=style, layout=layout_wide,
    )
    W["ttagWP"] = widgets.Dropdown(
        options=_TTAGWP_OPTS,
        value=_valid_choice(cfg["ttagWP"], _TTAGWP_OPTS, DEFAULTS["ttagWP"]),
        description="ttag WP", style=style, layout=layout,
    )
    W["btagger"] = widgets.Dropdown(
        options=_BTAGGER_OPTS,
        value=_valid_choice(cfg["btagger"], _BTAGGER_OPTS, DEFAULTS["btagger"]),
        description="B tagger", style=style, layout=layout,
    )
    W["ht"] = widgets.Dropdown(
        options=_HT_OPTS,
        value=_valid_choice(cfg["ht"], _HT_OPTS, DEFAULTS["ht"]),
        description="HT cut", style=style, layout=layout,
    )
    W["noSyst"] = widgets.Checkbox(value=cfg["noSyst"], description="No syst", style=style, layout=layout)
    W["ntuple"] = widgets.Checkbox(value=cfg["ntuple"], description="Ntuple", style=style, layout=layout)
    W["ntupleContent"] = widgets.Dropdown(
        options=[("Slim 2DAlphabet", "slim"), ("Full diagnostics", "full")],
        value=cfg["ntupleContent"] if cfg["ntupleContent"] in {"slim", "full"} else "slim",
        description="Ntuple cols", style=style, layout=layout_wide,
    )
    W["ntupleStorage"] = widgets.Dropdown(
        options=[("Write chunks", "chunks"), ("Accumulate memory", "accumulator")],
        value=cfg["ntupleStorage"] if cfg["ntupleStorage"] in {"chunks", "accumulator"} else "chunks",
        description="Ntuple mode", style=style, layout=layout_wide,
    )
    W["ntupleBaseDir"] = widgets.Text(
        value=cfg["ntupleBaseDir"], placeholder="optional shared chunk directory",
        description="Ntuple dir", style=style,
        layout=widgets.Layout(width="430px"),
    )
    W["overwrite"] = widgets.Checkbox(value=cfg["overwrite"], description="Overwrite", style=style, layout=layout)
    W["dask"] = widgets.Checkbox(value=cfg["dask"], description="Dask", style=style, layout=layout)
    W["daskMemory"] = widgets.IntSlider(
        value=int(cfg.get("daskMemory", DEFAULTS["daskMemory"])),
        min=1, max=20, step=1,
        description="Dask GB", style=style, layout=layout_wide,
        continuous_update=False,
    )
    W["chunksize"] = widgets.IntText(
        value=int(cfg.get("chunksize", DEFAULTS["chunksize"])),
        description="Chunksize", style=style, layout=layout_wide,
    )
    W["env"] = widgets.Dropdown(
        options=_ENV_OPTS,
        value=cfg["env"] if cfg["env"] in _ENV_OPTS else "lpc",
        description="Env", style=style, layout=layout,
    )
    W["test"] = widgets.Checkbox(value=cfg["test"], description="Test", style=style, layout=layout)
    W["nocluster"] = widgets.Checkbox(value=cfg["nocluster"], description="No cluster", style=style, layout=layout)
    W["cliOnly"] = widgets.Checkbox(
        value=cfg["cliOnly"], description="Print CLI only", style=style, layout=layout,
    )

    def save_config(_=None):
        snap = {k: _widget_value(w) for k, w in W.items()}
        with open(CONFIG_FILE, "w") as f:
            json.dump(snap, f, indent=2)

    def reset_to_defaults(_):
        for key, w in W.items():
            default = DEFAULTS[key]
            w.value = tuple(default) if isinstance(w, widgets.SelectMultiple) else default

    def refresh_subsample_options(_=None):
        selected_datasets = (
            list(DEFAULT_SIGNALS) if W["signals"].value else list(W["dataset"].value)
        )
        options = _available_subsamples(selected_datasets, W["iov"].value)
        current = [v for v in W["subsample"].value if v in options]
        W["subsample"].options = options
        W["subsample"].value = tuple(current)

    for w in W.values():
        w.observe(save_config, names="value")
    for w in (W["dataset"], W["iov"], W["signals"]):
        w.observe(refresh_subsample_options, names="value")
    refresh_subsample_options()

    btn_reset = widgets.Button(
        description="↺ Reset to Defaults", button_style="warning",
        layout=widgets.Layout(width="160px", margin="8px 0 0 0"),
    )
    btn_reset.on_click(reset_to_defaults)

    loaded = "restored from last session" if os.path.exists(CONFIG_FILE) else "using defaults"
    print(f"Config {loaded}; changes save to {CONFIG_FILE}.")
    print("Datasets")
    display(W["dataset"], W["signals"], W["iov"])
    print("Subsections")
    display(W["subsample"], W["mass"])
    print("Analysis options")
    for key in ("blind", "bkgest", "toptagger", "redirector", "ttagWP", "btagger",
                "ht", "noSyst", "ntuple", "ntupleContent", "ntupleStorage",
                "ntupleBaseDir", "overwrite"):
        display(W[key])
    print("Run options")
    for key in ("dask", "daskMemory", "chunksize", "env", "test", "nocluster", "cliOnly"):
        display(W[key])
    display(btn_reset)
    print("Adjust widgets above, then run the next cell to apply settings.")

    return W


def build_args(W):
    """Convert widget state into the args namespace expected by run_analysis()."""
    selected_datasets = list(W["dataset"].value)
    if W["signals"].value:
        selected_datasets = list(DEFAULT_SIGNALS)

    raw_mass = W["mass"].value.strip()
    mass_list = []
    if raw_mass:
        parts = [m.strip() for m in raw_mass.split(",")]
        invalid = [p for p in parts if not p.isdigit()]
        if invalid:
            print(f"Warning: invalid mass entries ignored: {invalid}")
        mass_list = [p for p in parts if p.isdigit()]

    cfg = {k: _widget_value(w) for k, w in W.items()}
    cfg["dataset"] = selected_datasets
    cfg["era"] = []
    cfg["pt"] = []
    cfg["mass"] = mass_list
    return SimpleNamespace(**cfg)


def build_cli_command(args, python_executable="python", script="ttbaranalysis.py"):
    """Build the CLI command equivalent to the notebook widget selection."""
    command = [python_executable, script]

    if getattr(args, "signals", False):
        command.append("--signals")
    else:
        for dataset in args.dataset:
            command.extend(["--dataset", dataset])

    command.extend(["--iov", args.iov])

    for era in getattr(args, "era", []):
        command.extend(["--era", era])
    for pt in getattr(args, "pt", []):
        command.extend(["--pt", pt])
    for mass in getattr(args, "mass", []):
        command.extend(["--mass", mass])
    for subsample in getattr(args, "subsample", []):
        command.extend(["--subsample", subsample])

    if args.blind:
        command.append("--blind")
    if args.bkgest:
        command.extend(["--bkgest", args.bkgest])
    command.extend(["--toptagger", args.toptagger])
    command.extend(["--redirector", args.redirector])
    command.extend(["--ttagWP", args.ttagWP])
    command.extend(["--btagger", args.btagger])
    command.extend(["--ht", args.ht])
    if args.noSyst:
        command.append("--noSyst")
    if args.ntuple:
        command.append("--ntuple")
        command.extend(["--ntupleContent", args.ntupleContent])
        command.extend(["--ntupleStorage", args.ntupleStorage])
        if args.ntupleBaseDir:
            command.extend(["--ntupleBaseDir", args.ntupleBaseDir])

    if args.dask:
        command.append("--dask")
    command.extend(["--env", args.env])
    if args.test:
        command.append("--test")
    if args.nocluster:
        command.append("--nocluster")
    if args.overwrite:
        command.append("--overwrite")
    command.extend(["--daskMemory", str(args.daskMemory)])
    if getattr(args, "chunksize", 0):
        command.extend(["--chunksize", str(args.chunksize)])

    return " ".join(shlex.quote(part) for part in command)
