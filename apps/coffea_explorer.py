from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import pickle
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import mplhep as hep
import numpy as np
import pandas as pd
import streamlit as st
from coffea import util
from hist import Hist


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOT = REPO_ROOT / "outputs"
SUPPORTED_SUFFIXES = {".coffea", ".pkl", ".pickle"}


@dataclass(frozen=True)
class LoadedFile:
    label: str
    path: Path
    size: int
    mtime_ns: int
    output: Any


@dataclass(frozen=True)
class Projection:
    axes: list[str]
    values: np.ndarray
    edges: list[np.ndarray]


def main() -> None:
    st.set_page_config(page_title="PKL/Coffea Explorer", layout="wide")
    plt.style.use(hep.style.CMS)
    apply_compact_plot_style()

    st.title("PKL/Coffea Explorer")
    st.caption("Local browser for Coffea outputs, pickle files, histograms, cutflows, and ntuples.")

    selected_paths = file_pickers()
    if not selected_paths:
        st.info("Choose a `.coffea`, `.pkl`, or `.pickle` file to start.")
        return

    loaded_files = load_selected_files(selected_paths)
    if not loaded_files:
        return

    active_file = st.sidebar.selectbox(
        "Summary file",
        loaded_files,
        format_func=lambda item: item.label,
    )
    st.sidebar.caption(file_identity(active_file))
    st.success("Loaded: " + ", ".join(item.label for item in loaded_files))

    tabs = st.tabs(["Summary", "Histograms", "Presets", "Ntuple", "Raw"])
    with tabs[0]:
        render_summary(active_file.output)
    with tabs[1]:
        render_histograms(loaded_files)
    with tabs[2]:
        render_presets(loaded_files)
    with tabs[3]:
        render_ntuple(active_file.output)
    with tabs[4]:
        render_raw(active_file.output)


def apply_compact_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.2, 3.8),
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "axes.linewidth": 0.9,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.major.size": 4,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.9,
            "xtick.minor.width": 0.7,
            "ytick.major.width": 0.9,
            "ytick.minor.width": 0.7,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "lines.linewidth": 1.15,
        }
    )


def file_pickers() -> list[Path]:
    st.sidebar.header("Input")
    discovered = discover_files(DEFAULT_SEARCH_ROOT)
    display = ["Manual path"] + [str(path.relative_to(REPO_ROOT)) for path in discovered]
    n_files = st.sidebar.slider("File slots", min_value=1, max_value=3, value=1)
    selected_paths = []

    for idx in range(n_files):
        choice = st.sidebar.selectbox(
            f"File {idx + 1}",
            display,
            index=1 if discovered else 0,
            key=f"file-choice-{idx}",
        )

        if choice == "Manual path":
            manual = st.sidebar.text_input("Path", value="", key=f"manual-path-{idx}")
            if not manual:
                continue
            path = Path(manual).expanduser()
            if not path.is_absolute():
                path = REPO_ROOT / path
        else:
            path = REPO_ROOT / choice

        if not path.exists():
            st.sidebar.error(f"`{path}` does not exist.")
            continue
        if path.suffix not in SUPPORTED_SUFFIXES:
            st.sidebar.warning(f"Unsupported suffix `{path.suffix}`.")
            continue
        selected_paths.append(path)
    return selected_paths


def load_selected_files(paths: list[Path]) -> list[LoadedFile]:
    loaded_files = []
    for idx, path in enumerate(paths, start=1):
        stat = path.stat()
        try:
            output = load_output(path, stat.st_size, stat.st_mtime_ns)
        except Exception as exc:
            st.error(f"Could not load `{path}`")
            st.exception(exc)
            continue
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        loaded_files.append(
            LoadedFile(
                label=f"{idx}: {rel}",
                path=path,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                output=output,
            )
        )
    return loaded_files


def discover_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for suffix in SUPPORTED_SUFFIXES:
        paths.extend(root.rglob(f"*{suffix}"))
    return sorted(paths)


def file_identity(item: LoadedFile) -> str:
    mtime = datetime.fromtimestamp(item.mtime_ns / 1_000_000_000).isoformat(timespec="seconds")
    return f"`{item.path.resolve()}`\n\n{item.size:,} bytes, modified {mtime}"


@st.cache_data(show_spinner="Loading file...")
def load_output(path: Path, size: int, mtime_ns: int) -> Any:
    del size, mtime_ns
    if path.suffix == ".coffea":
        return util.load(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def render_summary(output: Any) -> None:
    st.subheader("File Summary")
    rows = summarize_mapping(output)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write(type_name(output))

    histograms = find_histograms(output)
    if histograms:
        st.subheader("Histograms")
        hist_rows = [
            {
                "key": key,
                "dimensions": hist.ndim,
                "axes": ", ".join(axis.name or f"axis_{idx}" for idx, axis in enumerate(hist.axes)),
                "entries": safe_sum(hist),
            }
            for key, hist in histograms.items()
        ]
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    categories = get_mapping(output, "analysisCategories")
    if categories:
        st.subheader("Analysis Categories")
        st.dataframe(mapping_to_frame(categories), use_container_width=True, hide_index=True)

    cutflow = get_mapping(output, "cutflow")
    if cutflow:
        st.subheader("Cutflow")
        st.dataframe(mapping_to_frame(cutflow, key_name="cut", value_name="count"), use_container_width=True, hide_index=True)

    ntuple = get_mapping(output, "ntuple")
    if ntuple:
        st.subheader("Ntuple Branches")
        branch_rows = [
            {"branch": key, "type": type_name(value), "length": len(to_numpy(value))}
            for key, value in ntuple.items()
        ]
        st.dataframe(pd.DataFrame(branch_rows), use_container_width=True, hide_index=True)


def render_histograms(loaded_files: list[LoadedFile]) -> None:
    histograms_by_file = {item.label: find_histograms(item.output) for item in loaded_files}
    source_options = [item.label for item in loaded_files if histograms_by_file[item.label]]
    if not source_options:
        st.info("No `hist.Hist` objects found at the top level of the loaded file(s).")
        return

    top_cols = st.columns([1.2, 1.2, 1.8])
    with top_cols[0]:
        primary_source = st.selectbox("Source", source_options)
    histograms = histograms_by_file[primary_source]

    with top_cols[1]:
        hist_key = st.selectbox("Histogram", list(histograms))
    hist_obj = histograms[hist_key]

    axis_names = named_axes(hist_obj)
    default_plot_axes = default_numeric_axes(hist_obj)
    with top_cols[2]:
        plot_axes = st.multiselect(
            "Plot axes",
            axis_names,
            default=default_plot_axes[: min(2, len(default_plot_axes))],
            help="Choose one axis for a 1D plot or two axes for a 2D heatmap.",
        )

    if len(plot_axes) not in {1, 2}:
        st.warning("Choose exactly one or two plot axes.")
        return

    with st.expander("Axes", expanded=False):
        st.dataframe(axis_frame(hist_obj), use_container_width=True, hide_index=True)

    state_key = widget_key(primary_source, hist_key, ",".join(plot_axes))
    traces = trace_controls(histograms_by_file, source_options, primary_source, hist_key, plot_axes, state_key)
    selections = axis_selections(hist_obj, plot_axes, state_key)
    ranges = axis_ranges(hist_obj, state_key)
    opt_cols = st.columns([1, 1, 2.6])
    with opt_cols[0]:
        density = st.checkbox("Density", value=False)
    with opt_cols[1]:
        log_scale = st.checkbox("Log", value=False)
    with opt_cols[2]:
        plot_width = st.slider("Width", min_value=420, max_value=1000, value=720, step=20)

    if not traces:
        st.warning("No compatible traces selected for these plot axes.")
        return

    try:
        plotted_traces = [
            (
                trace_label,
                project_hist(trace_hist, plot_axes, selections, ranges),
            )
            for trace_label, trace_hist in traces
        ]
    except Exception as exc:
        st.error("Could not slice/project this histogram.")
        st.exception(exc)
        return

    if len(plot_axes) == 1:
        fig, ax = plt.subplots()
        for trace_label, plotted in plotted_traces:
            vals = plotted.values
            if density:
                area = np.sum(vals)
                if area > 0:
                    vals = vals / area
            hep.histplot(
                vals,
                bins=plotted.edges[0],
                ax=ax,
                histtype="step",
                linewidth=1.15,
                label=trace_label,
            )
        ax.set_ylabel("Density" if density else "Events")
        ax.set_xlabel(axis_label(hist_obj, plot_axes[0]))
        if log_scale:
            ax.set_yscale("log")
        apply_legend(ax, ranges)
    else:
        fig, axes = plt.subplots(1, len(plotted_traces), squeeze=False)
        for ax, (trace_label, plotted) in zip(axes[0], plotted_traces):
            mesh = ax.pcolormesh(
                plotted.edges[0],
                plotted.edges[1],
                plotted.values.T,
                norm=LogNorm() if log_scale else None,
            )
            fig.colorbar(mesh, ax=ax)
            ax.set_xlabel(axis_label(hist_obj, plot_axes[0]))
            ax.set_ylabel(axis_label(hist_obj, plot_axes[1]))
            ax.set_title(trace_label)
            if ranges:
                apply_legend(ax, ranges)
        if len(plotted_traces) == 1:
            ax = axes[0][0]
    if len(plot_axes) == 1:
        ax.set_title(", ".join(label for label, _ in plotted_traces))
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width=plot_width)


def render_presets(loaded_files: list[LoadedFile]) -> None:
    st.subheader("2DAlphabet Preset")
    candidates = [
        item
        for item in loaded_files
        if {"data", "ttbar"}.issubset(find_histograms(item.output))
    ]
    if not candidates:
        st.info("Load a file with top-level `data` and `ttbar` histograms to use this preset.")
        return

    cols = st.columns([1.4, 0.7, 0.7, 0.7, 0.7, 1.1])
    with cols[0]:
        source = st.selectbox("Source", candidates, format_func=lambda item: item.label, key="preset-source")
    with cols[1]:
        dy_region = st.radio("Preset", ["cen", "fwd"], horizontal=True)
    with cols[2]:
        low_min = st.number_input("mt low min", value=25.0, step=5.0)
    with cols[3]:
        sig_low = st.number_input("mt sig min", value=105.0, step=5.0)
    with cols[4]:
        sig_high = st.number_input("mt sig max", value=210.0, step=5.0)
    with cols[5]:
        plot_width = st.slider("Preset width", min_value=650, max_value=1300, value=1000, step=50)

    hists = find_histograms(source.output)
    data_hist = hists["data"]
    ttbar_hist = hists["ttbar"]
    if not compatible_2dalphabet_hist(data_hist) or not compatible_2dalphabet_hist(ttbar_hist):
        st.warning("The `data` and `ttbar` histograms need `systematic`, `anacat`, `jetmass`, and `ttbarmass` axes.")
        return

    mtt_edges = np.asarray(axis_by_name(data_hist, "ttbarmass").edges, dtype=float)
    sig_low = float(sig_low)
    sig_high = float(sig_high)
    low_min = float(low_min)
    high_max = st.number_input("mt high max", value=475.0, step=5.0)
    high_max = float(high_max)
    if not (low_min < sig_low < sig_high < high_max):
        st.warning("Require `mt low min < mt sig min < mt sig max < mt high max`.")
        return

    mtt_cols = st.columns([0.8, 0.8, 2.4])
    with mtt_cols[0]:
        mtt_low = st.number_input("mtt min", value=float(mtt_edges[0]), step=100.0)
    with mtt_cols[1]:
        mtt_high = st.number_input("mtt max", value=float(min(4000.0, mtt_edges[-1])), step=100.0)

    anacats = twodalphabet_anacats(dy_region)
    regions = [
        ("A", "Fail low", anacats["fail"], (low_min, sig_low)),
        ("C", "Fail signal", anacats["fail"], (sig_low, sig_high)),
        ("E", "Fail high", anacats["fail"], (sig_high, high_max)),
        ("B", "Pass low", anacats["pass"], (low_min, sig_low)),
        ("D", "Pass signal", anacats["pass"], (sig_low, sig_high)),
        ("F", "Pass high", anacats["pass"], (sig_high, high_max)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6), sharex=True)
    for ax, (letter, title, anacat, jet_range) in zip(axes.flat, regions):
        ranges = {"jetmass": jet_range, "ttbarmass": (float(mtt_low), float(mtt_high))}
        selections = {"systematic": "nominal", "anacat": anacat}
        for label, hist_obj in [("data", data_hist), ("ttbar", ttbar_hist)]:
            projected = project_hist(hist_obj, ["ttbarmass"], selections, ranges)
            hep.histplot(
                projected.values,
                bins=projected.edges[0],
                ax=ax,
                histtype="step",
                linewidth=1.0,
                label=f"{label} ({projected.values.sum():.0f})",
            )
        ax.set_title(f"{letter}: {title}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Events")
        ax.legend(fontsize=8, handlelength=1.4)
    for ax in axes[-1]:
        ax.set_xlabel(axis_label(data_hist, "ttbarmass"))
    fig.suptitle(f"2DAlphabet {dy_region}: data vs ttbar", fontsize=13)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width=plot_width)

    st.caption(
        f"{dy_region}: fail anacat={anacats['fail']}, pass anacat={anacats['pass']}; "
        f"ABCDEF mt windows={low_min:g}-{sig_low:g}, {sig_low:g}-{sig_high:g}, {sig_high:g}-{high_max:g}; "
        f"mtt={float(mtt_low):g}-{float(mtt_high):g}"
    )


def apply_legend(ax: Any, ranges: dict[str, tuple[float, float]]) -> None:
    handles, labels = ax.get_legend_handles_labels()
    range_label = range_legend_label(ranges)
    if range_label:
        handles = [Line2D([], [], linestyle="none", marker=None, color="none")] + handles
        labels = [range_label] + labels
    if not handles:
        return
    ax.legend(handles, labels, handlelength=1.8, handletextpad=0.8)


def range_legend_label(ranges: dict[str, tuple[float, float]]) -> str | None:
    if not ranges:
        return None
    return ", ".join(f"{axis}: {low:g}-{high:g}" for axis, (low, high) in ranges.items())


def compatible_2dalphabet_hist(hist_obj: Hist) -> bool:
    axis_names = set(named_axes(hist_obj))
    return {"systematic", "anacat", "jetmass", "ttbarmass"}.issubset(axis_names)


def axis_by_name(hist_obj: Hist, axis_name: str) -> Any:
    for idx, axis in enumerate(hist_obj.axes):
        name = axis.name or f"axis_{idx}"
        if name == axis_name:
            return axis
    raise KeyError(axis_name)


def twodalphabet_anacats(dy_region: str) -> dict[str, int]:
    # Current Run-3 category order from build_analysis_categories:
    # atcen=0, atfwd=1, 2tcen=2, 2tfwd=3.
    if dy_region == "cen":
        return {"fail": 0, "pass": 2}
    return {"fail": 1, "pass": 3}


def trace_controls(
    histograms_by_file: dict[str, dict[str, Hist]],
    source_options: list[str],
    primary_source: str,
    primary_hist_key: str,
    plot_axes: list[str],
    state_key: str,
) -> list[tuple[str, Hist]]:
    traces: list[tuple[str, Hist]] = []

    with st.expander("Traces", expanded=False):
        n_traces = st.slider("Count", min_value=1, max_value=3, value=1, key=f"trace-count-{state_key}")
        if len(source_options) == 1:
            st.caption("Only one source is loaded. Increase **File slots** in the sidebar to compare against another file.")

        for idx in range(n_traces):
            col_source, col_hist, col_label = st.columns([1.1, 1.1, 1.2])
            with col_source:
                source = st.selectbox(
                    f"Trace {idx + 1} source",
                    source_options,
                    index=source_options.index(primary_source) if primary_source in source_options else 0,
                    key=f"trace-source-{idx}-{state_key}",
                )

            hist_keys = [
                key for key, hist_obj in histograms_by_file[source].items()
                if all(axis in named_axes(hist_obj) for axis in plot_axes)
            ]
            if not hist_keys:
                st.warning(f"No trace histograms in `{source}` have axes: {', '.join(plot_axes)}")
                continue
            default_hist_index = hist_keys.index(primary_hist_key) if primary_hist_key in hist_keys else 0
            with col_hist:
                hist_key = st.selectbox(
                    f"Trace {idx + 1} histogram",
                    hist_keys,
                    index=default_hist_index,
                    key=f"trace-hist-{idx}-{state_key}-{source}",
                )

            default_label = hist_key if n_traces > 1 else primary_hist_key
            if len(source_options) > 1:
                default_label = f"{Path(source.split(': ', 1)[-1]).stem}:{hist_key}"
            with col_label:
                label = st.text_input(
                    f"Trace {idx + 1} label",
                    value=default_label,
                    key=f"trace-label-{idx}-{state_key}-{source}-{hist_key}",
                )

            traces.append((label, histograms_by_file[source][hist_key]))
    return traces


def render_ntuple(output: Any) -> None:
    ntuple = get_mapping(output, "ntuple")
    if not ntuple:
        st.info("This file does not contain an `ntuple` mapping.")
        return

    df = ntuple_to_frame(ntuple)
    if df.empty:
        st.warning("The ntuple is present, but no branches could be converted to arrays.")
        return

    st.write(f"{len(df):,} rows, {len(df.columns):,} branches")
    st.dataframe(df.head(1000), use_container_width=True)

    numeric_columns = list(df.select_dtypes(include=np.number).columns)
    if not numeric_columns:
        st.info("No numeric ntuple branches found for plotting.")
        return

    if "anacat" in df:
        category_values = sorted(df["anacat"].dropna().unique().tolist())
        selected_categories = st.multiselect("anacat filter", category_values, default=category_values)
        if selected_categories:
            df = df[df["anacat"].isin(selected_categories)]

    mode = st.radio("Plot mode", ["1D histogram", "2D heatmap"], horizontal=True)
    weight_col = "weight" if "weight" in numeric_columns else None
    use_weights = st.checkbox("Use `weight` branch", value=weight_col is not None, disabled=weight_col is None)

    if mode == "1D histogram":
        branch = st.selectbox("Branch", numeric_columns)
        bins = st.slider("Bins", min_value=10, max_value=200, value=50, step=5)
        log_y = st.checkbox("Log y", value=False)
        weights = df[weight_col].to_numpy() if use_weights and weight_col else None

        fig, ax = plt.subplots()
        ax.hist(df[branch].to_numpy(), bins=bins, weights=weights, histtype="step")
        ax.set_xlabel(branch)
        ax.set_ylabel("Weighted events" if weights is not None else "Events")
        if log_y:
            ax.set_yscale("log")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, width="content")
    else:
        x_col = st.selectbox("X branch", numeric_columns)
        y_options = [col for col in numeric_columns if col != x_col] or numeric_columns
        y_col = st.selectbox("Y branch", y_options)
        bins = st.slider("2D bins", min_value=10, max_value=150, value=50, step=5)
        log_z = st.checkbox("Log z", value=False)
        weights = df[weight_col].to_numpy() if use_weights and weight_col else None

        fig, ax = plt.subplots()
        norm = LogNorm() if log_z else None
        h = ax.hist2d(df[x_col].to_numpy(), df[y_col].to_numpy(), bins=bins, weights=weights, norm=norm)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, width="content")


def render_raw(output: Any) -> None:
    st.subheader("Raw Object")
    if isinstance(output, dict):
        key = st.selectbox("Top-level key", list(output))
        st.write(output[key])
    else:
        st.write(output)


def find_histograms(output: Any) -> dict[str, Hist]:
    if not isinstance(output, dict):
        return {}
    return {str(key): value for key, value in output.items() if isinstance(value, Hist)}


def summarize_mapping(output: Any) -> list[dict[str, Any]]:
    if not isinstance(output, dict):
        return []
    rows = []
    for key, value in output.items():
        rows.append(
            {
                "key": key,
                "type": type_name(value),
                "summary": object_summary(value),
            }
        )
    return rows


def get_mapping(output: Any, key: str) -> dict[Any, Any]:
    if not isinstance(output, dict):
        return {}
    value = output.get(key, {})
    return value if isinstance(value, dict) else {}


def mapping_to_frame(mapping: dict[Any, Any], key_name: str = "key", value_name: str = "value") -> pd.DataFrame:
    return pd.DataFrame([{key_name: key, value_name: value} for key, value in mapping.items()])


def axis_frame(hist_obj: Hist) -> pd.DataFrame:
    rows = []
    for idx, axis in enumerate(hist_obj.axes):
        rows.append(
            {
                "name": axis.name or f"axis_{idx}",
                "label": axis.label,
                "type": type_name(axis),
                "size": axis.size,
                "categories": ", ".join(map(str, axis)) if is_discrete_axis(axis) else "",
            }
        )
    return pd.DataFrame(rows)


def named_axes(hist_obj: Hist) -> list[str]:
    return [axis.name or f"axis_{idx}" for idx, axis in enumerate(hist_obj.axes)]


def default_numeric_axes(hist_obj: Hist) -> list[str]:
    numeric = []
    for idx, axis in enumerate(hist_obj.axes):
        if not is_discrete_axis(axis):
            numeric.append(axis.name or f"axis_{idx}")
    return numeric or named_axes(hist_obj)[:1]


def axis_selections(hist_obj: Hist, plot_axes: list[str], state_key: str = "") -> dict[str, Any]:
    selections: dict[str, Any] = {}
    discrete_axes = [
        (idx, axis)
        for idx, axis in enumerate(hist_obj.axes)
        if is_discrete_axis(axis) and (axis.name or f"axis_{idx}") not in plot_axes
    ]
    if not discrete_axes:
        return selections

    cols = st.columns(min(3, len(discrete_axes)))
    for col_idx, (idx, axis) in enumerate(discrete_axes):
        with cols[col_idx % len(cols)]:
            name = axis.name or f"axis_{idx}"
            values = list(axis)
            if not values:
                continue

            default = ["nominal"] if "nominal" in values else values[:1]
            chosen = st.multiselect(f"{name}", values, default=default, key=f"select-{name}-{state_key}")
            if not chosen:
                st.warning(f"No values selected for `{name}`; using all values.")
                continue
            selections[name] = chosen if len(chosen) > 1 else chosen[0]
    return selections


def axis_ranges(hist_obj: Hist, state_key: str = "") -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    numeric_axes = [
        (idx, axis)
        for idx, axis in enumerate(hist_obj.axes)
        if not is_discrete_axis(axis) and axis.size > 0
    ]
    if not numeric_axes:
        return ranges

    with st.expander("Ranges", expanded=False):
        st.caption("Applied before summing and projection.")
        for idx, axis in numeric_axes:
            name = axis.name or f"axis_{idx}"
            edges = axis.edges
            low_default = float(edges[0])
            high_default = float(edges[-1])
            enabled = st.checkbox(f"Restrict `{name}`", value=False, key=f"range-enable-{name}-{state_key}")
            if not enabled:
                continue

            col_low, col_high = st.columns(2)
            with col_low:
                low = st.number_input(
                    "Min",
                    value=low_default,
                    min_value=low_default,
                    max_value=high_default,
                    key=f"range-low-{name}-{state_key}",
                )
            with col_high:
                high = st.number_input(
                    "Max",
                    value=high_default,
                    min_value=low_default,
                    max_value=high_default,
                    key=f"range-high-{name}-{state_key}",
                )

            if low >= high:
                st.warning(f"`{name}` min must be smaller than max; ignoring this range.")
                continue
            ranges[name] = (float(low), float(high))
    return ranges


def project_hist(
    hist_obj: Hist,
    plot_axes: list[str],
    selections: dict[str, Any],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> Projection:
    sliced = hist_obj
    if selections:
        sliced = sliced[selections]

    values = np.asarray(sliced.values(flow=False))
    axis_info = [
        {
            "idx": idx,
            "name": axis.name or f"axis_{idx}",
            "axis": axis,
        }
        for idx, axis in enumerate(sliced.axes)
    ]
    ranges = ranges or {}

    for axis_index in reversed(range(len(axis_info))):
        info = axis_info[axis_index]
        name = info["name"]
        axis = info["axis"]
        if name in ranges and not is_discrete_axis(axis):
            low, high = ranges[name]
            centers = np.asarray(axis.centers)
            mask = (centers >= low) & (centers < high)
            indices = np.flatnonzero(mask)
            values = np.take(values, indices, axis=axis_index)
            info["range_indices"] = indices

        if name not in plot_axes:
            values = values.sum(axis=axis_index)
            axis_info.pop(axis_index)

    ordered_axes = []
    ordered_edges = []
    for plot_axis in plot_axes:
        info = next((item for item in axis_info if item["name"] == plot_axis), None)
        if info is None:
            available = ", ".join(item["name"] for item in axis_info)
            raise ValueError(f"Histogram does not have plot axis `{plot_axis}` after slicing. Available axes: {available}")
        axis = info["axis"]
        if is_discrete_axis(axis):
            raise ValueError(f"Plot axis `{plot_axis}` is categorical; choose numeric plot axes for now.")
        ordered_axes.append(plot_axis)
        ordered_edges.append(edges_for_axis(axis, info.get("range_indices")))

    current_order = [item["name"] for item in axis_info]
    transpose_order = [current_order.index(name) for name in plot_axes]
    if transpose_order != list(range(len(transpose_order))):
        values = np.transpose(values, transpose_order)
    return Projection(axes=ordered_axes, values=values, edges=ordered_edges)


def widget_key(*parts: Any) -> str:
    text = "::".join(str(part) for part in parts)
    return "".join(ch if ch.isalnum() else "_" for ch in text)[-120:]


def edges_for_axis(axis: Any, indices: np.ndarray | None) -> np.ndarray:
    edges = np.asarray(axis.edges, dtype=float)
    if indices is None:
        return edges
    if len(indices) == 0:
        return np.asarray([edges[0], edges[0]], dtype=float)
    return edges[indices[0] : indices[-1] + 2]


def axis_label(hist_obj: Hist, axis_name: str) -> str:
    for idx, axis in enumerate(hist_obj.axes):
        name = axis.name or f"axis_{idx}"
        if name == axis_name:
            return axis.label or name
    return axis_name


def is_discrete_axis(axis: Any) -> bool:
    return "StrCategory" in type_name(axis) or "IntCategory" in type_name(axis) or "Boolean" in type_name(axis)


def ntuple_to_frame(ntuple: dict[Any, Any]) -> pd.DataFrame:
    arrays = {}
    target_len = None
    for key, value in ntuple.items():
        arr = to_numpy(value)
        if arr.ndim != 1:
            continue
        if target_len is None:
            target_len = len(arr)
        if len(arr) == target_len:
            arrays[str(key)] = arr
    return pd.DataFrame(arrays)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "value") and not callable(value.value):
        return np.asarray(value.value)
    return np.asarray(value)


def safe_sum(hist_obj: Hist) -> float:
    try:
        return float(np.nansum(hist_obj.values()))
    except Exception:
        return float("nan")


def object_summary(value: Any) -> str:
    if isinstance(value, Hist):
        return f"{value.ndim}D histogram, axes: {', '.join(named_axes(value))}"
    if isinstance(value, dict):
        return f"{len(value)} keys"
    try:
        return f"len={len(value)}"
    except Exception:
        return ""


def type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


if __name__ == "__main__":
    main()
