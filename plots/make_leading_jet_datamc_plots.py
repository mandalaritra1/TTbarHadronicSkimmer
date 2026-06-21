#!/usr/bin/env python3
"""Make leading-jet kinematics Data/MC plots from coffea outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from coffea import util
from hist import sum as hist_sum


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COFFEA_DIR = REPO_ROOT / "outputs" / "dy"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "plots" / "leading_jet_datamc"

BACKGROUND_STYLES = {
    "TTbar": {"label": "SM TTbar", "color": "#b00000"},
    "QCD": {"label": "SM QCD", "color": "#f6c85f"},
}

CATEGORY_GROUPS = {
    "all": [0, 1, 2, 3],
    "fail": [0, 1],
    "pass": [2, 3],
    "central": [0, 2],
    "forward": [1, 3],
    "fail-central": [0],
    "fail-forward": [1],
    "pass-central": [2],
    "pass-forward": [3],
}


@dataclass(frozen=True)
class VariableConfig:
    hist_name: str
    axis_name: str
    label: str
    rebin: int = 1
    xlim: tuple[float, float] | None = None


VARIABLES = {
    "pt": VariableConfig(
        hist_name="jet0_pt",
        axis_name="jetpt",
        label=r"Leading jet $p_T$ [GeV]",
        rebin=2,
        xlim=(400.0, 2000.0),
    ),
    "phi": VariableConfig(
        hist_name="jet0_phi",
        axis_name="jetphi",
        label=r"Leading jet $\phi$",
        rebin=1,
        xlim=(-3.14159, 3.14159),
    ),
    "msd": VariableConfig(
        hist_name="jetmsd",
        axis_name="jetmsd",
        label=r"Leading jet $m_{SD}$ [GeV]",
        rebin=2,
        xlim=(0.0, 500.0),
    ),
    "y": VariableConfig(
        hist_name="jet0_rapidity",
        axis_name="jety",
        label=r"Leading jet $y$",
        rebin=1,
        xlim=(-2.5, 2.5),
    ),
}


@dataclass
class Hist1D:
    values: np.ndarray
    variances: np.ndarray
    edges: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot leading-jet kinematic Data/MC comparisons from coffea files."
    )
    parser.add_argument("--year", default="2024", help="Year label for the CMS lumi text.")
    parser.add_argument(
        "--lumi",
        default="109.95",
        help="Luminosity text in fb^{-1}; set empty to omit.",
    )
    parser.add_argument(
        "--coffea-dir",
        type=Path,
        default=DEFAULT_COFFEA_DIR,
        help="Directory containing data, TTbar, and QCD coffea outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--data-pattern",
        default="data_2024_*.coffea",
        help="Glob pattern for data coffea files inside --coffea-dir.",
    )
    parser.add_argument(
        "--ttbar-pattern",
        default="TTbar_2024*.coffea",
        help="Glob pattern for TTbar coffea files inside --coffea-dir.",
    )
    parser.add_argument(
        "--qcd-pattern",
        default="QCD_2024*.coffea",
        help="Glob pattern for QCD coffea files inside --coffea-dir.",
    )
    parser.add_argument(
        "--category-group",
        choices=sorted(CATEGORY_GROUPS),
        default="all",
        help="Analysis categories to sum.",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        choices=sorted(VARIABLES),
        default=["pt", "phi", "msd", "y"],
        help="Variables to draw in the 2x2 summary.",
    )
    parser.add_argument(
        "--scale-qcd-to-data-minus-ttbar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale QCD independently in each panel to max(Data - TTbar, 0).",
    )
    parser.add_argument(
        "--single-figsize",
        nargs=2,
        type=float,
        default=(12.0, 10.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches for each individual plot.",
    )
    return parser.parse_args()


def _load_many(paths: list[Path]) -> list[dict]:
    return [util.load(path) for path in paths]


def _paths_from_pattern(directory: Path, pattern: str, sample_name: str) -> list[Path]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {sample_name} files matched {directory / pattern}")
    return paths


def _project_one(output: dict, config: VariableConfig, category_ids: list[int]) -> Hist1D:
    if config.hist_name not in output:
        raise KeyError(f"{config.hist_name!r} not found in coffea output")

    histo = output[config.hist_name]
    selected = histo[{"systematic": "nominal", "anacat": category_ids}]
    projected = selected[{"anacat": hist_sum}]
    view = projected.view(flow=False)
    return Hist1D(
        values=np.asarray(view.value, dtype=float),
        variances=np.asarray(view.variance, dtype=float),
        edges=np.asarray(projected.axes[config.axis_name].edges, dtype=float),
    )


def _sum_outputs(outputs: list[dict], config: VariableConfig, category_ids: list[int]) -> Hist1D:
    total: Hist1D | None = None
    for output in outputs:
        current = _project_one(output, config, category_ids)
        if total is None:
            total = Hist1D(current.values.copy(), current.variances.copy(), current.edges.copy())
            continue
        if not np.allclose(total.edges, current.edges):
            raise ValueError(f"Inconsistent binning for {config.hist_name}")
        total.values += current.values
        total.variances += current.variances
    if total is None:
        raise ValueError("No outputs to sum")
    return total


def _rebin(histogram: Hist1D, factor: int) -> Hist1D:
    if factor <= 1:
        return histogram
    n_bins = len(histogram.values)
    n_full_groups = n_bins // factor
    kept_bins = n_full_groups * factor
    values = histogram.values[:kept_bins].reshape(n_full_groups, factor).sum(axis=1)
    variances = histogram.variances[:kept_bins].reshape(n_full_groups, factor).sum(axis=1)
    edges = histogram.edges[: kept_bins + 1 : factor]
    if edges[-1] != histogram.edges[kept_bins]:
        edges = np.append(edges, histogram.edges[kept_bins])
    if kept_bins < n_bins:
        values = np.append(values, histogram.values[kept_bins:].sum())
        variances = np.append(variances, histogram.variances[kept_bins:].sum())
        edges = np.append(edges, histogram.edges[-1])
    return Hist1D(values=values, variances=variances, edges=edges)


def _restrict_range(histogram: Hist1D, xlim: tuple[float, float] | None) -> Hist1D:
    if xlim is None:
        return histogram
    low, high = xlim
    mask = (histogram.edges[:-1] >= low) & (histogram.edges[1:] <= high)
    if not np.any(mask):
        return histogram
    indices = np.where(mask)[0]
    start, stop = indices[0], indices[-1] + 1
    return Hist1D(
        values=histogram.values[start:stop],
        variances=histogram.variances[start:stop],
        edges=histogram.edges[start : stop + 1],
    )


def _prepare_hist(histogram: Hist1D, config: VariableConfig) -> Hist1D:
    return _restrict_range(_rebin(histogram, config.rebin), config.xlim)


def _qcd_scale(data: Hist1D, ttbar: Hist1D, qcd: Hist1D) -> float:
    qcd_integral = float(np.sum(qcd.values))
    if qcd_integral <= 0.0:
        return 0.0
    target = float(np.sum(data.values) - np.sum(ttbar.values))
    return max(target / qcd_integral, 0.0)


def _apply_scale(histogram: Hist1D, scale: float) -> Hist1D:
    return Hist1D(
        values=histogram.values * scale,
        variances=histogram.variances * scale * scale,
        edges=histogram.edges,
    )


def _poisson_errors(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(values, 0.0))


def _draw_stack(ax: plt.Axes, edges: np.ndarray, ttbar: np.ndarray, qcd: np.ndarray) -> None:
    ax.stairs(
        ttbar,
        edges,
        baseline=np.zeros_like(ttbar),
        fill=True,
        color=BACKGROUND_STYLES["TTbar"]["color"],
        edgecolor=BACKGROUND_STYLES["TTbar"]["color"],
        linewidth=0.7,
        label=BACKGROUND_STYLES["TTbar"]["label"],
    )
    ax.stairs(
        ttbar + qcd,
        edges,
        baseline=ttbar,
        fill=True,
        color=BACKGROUND_STYLES["QCD"]["color"],
        edgecolor=BACKGROUND_STYLES["QCD"]["color"],
        linewidth=0.7,
        label=BACKGROUND_STYLES["QCD"]["label"],
    )


def _draw_uncertainty_band(ax: plt.Axes, edges: np.ndarray, center: np.ndarray, err: np.ndarray) -> None:
    low = np.maximum(center - err, 1e-9)
    high = center + err
    ax.stairs(
        high,
        edges,
        baseline=low,
        fill=True,
        color="0.75",
        alpha=0.45,
        hatch="///",
        edgecolor="0.55",
        linewidth=0.0,
        label="Total bkg unc.",
    )


def _draw_data(ax: plt.Axes, histo: Hist1D, label: str = "Data") -> None:
    centers = 0.5 * (histo.edges[:-1] + histo.edges[1:])
    widths = 0.5 * np.diff(histo.edges)
    ax.errorbar(
        centers,
        histo.values,
        yerr=_poisson_errors(histo.values),
        xerr=widths,
        fmt="o",
        color="black",
        markersize=2.5,
        linewidth=0.8,
        capsize=0,
        label=label,
    )


def _draw_ratio(rax: plt.Axes, data: Hist1D, mc_total: np.ndarray, mc_err: np.ndarray) -> None:
    centers = 0.5 * (data.edges[:-1] + data.edges[1:])
    widths = 0.5 * np.diff(data.edges)
    ratio = np.full_like(data.values, np.nan, dtype=float)
    ratio_err = np.full_like(data.values, np.nan, dtype=float)
    mask = mc_total > 0.0
    ratio[mask] = data.values[mask] / mc_total[mask]
    ratio_err[mask] = _poisson_errors(data.values[mask]) / mc_total[mask]

    rel_unc = np.zeros_like(mc_total, dtype=float)
    rel_mask = mc_total > 0.0
    rel_unc[rel_mask] = mc_err[rel_mask] / mc_total[rel_mask]
    _draw_uncertainty_band(rax, data.edges, np.ones_like(mc_total), rel_unc)
    rax.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        xerr=widths,
        fmt="o",
        color="black",
        markersize=2.0,
        linewidth=0.7,
        capsize=0,
    )
    rax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
    rax.set_ylim(0.0, 2.0)
    rax.set_ylabel("Data/MC", fontsize=24, labelpad=18)
    rax.yaxis.set_label_coords(-0.075, 0.5)
    rax.grid(axis="y", color="0.85", linewidth=0.7)


def _plot_panel(
    fig: plt.Figure,
    outer_spec,
    config: VariableConfig,
    data: Hist1D,
    ttbar: Hist1D,
    qcd: Hist1D,
    year: str,
    lumi: str,
) -> None:
    gs = outer_spec.subgridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)

    _draw_stack(ax, data.edges, ttbar.values, qcd.values)
    mc_total = ttbar.values + qcd.values
    mc_var = ttbar.variances + qcd.variances
    mc_err = np.sqrt(np.maximum(mc_var, 0.0))
    _draw_uncertainty_band(ax, data.edges, mc_total, mc_err)
    _draw_data(ax, data)
    _draw_ratio(rax, data, mc_total, mc_err)

    ax.set_yscale("log")
    positive = np.concatenate([data.values[data.values > 0], mc_total[mc_total > 0]])
    ymin = 0.1 if positive.size == 0 else max(min(positive) * 0.3, 0.08)
    ymax = 10.0 if positive.size == 0 else max(positive.max() * 35.0, 10.0)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Events/Bin")
    rax.set_xlabel(config.label)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper right", frameon=False)
    hep.cms.label("Work in Progress", data=True, lumi=(lumi or None), year=year, ax=ax, fontsize=20)


def main() -> None:
    args = _parse_args()
    plt.style.use(hep.style.CMS)

    data_paths = _paths_from_pattern(args.coffea_dir, args.data_pattern, "data")
    ttbar_paths = _paths_from_pattern(args.coffea_dir, args.ttbar_pattern, "TTbar")
    qcd_paths = _paths_from_pattern(args.coffea_dir, args.qcd_pattern, "QCD")
    category_ids = CATEGORY_GROUPS[args.category_group]

    data_outputs = _load_many(data_paths)
    ttbar_outputs = _load_many(ttbar_paths)
    qcd_outputs = _load_many(qcd_paths)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    group_suffix = args.category_group.replace("-", "_")
    for variable_name in args.variables:
        config = VARIABLES[variable_name]
        data = _prepare_hist(_sum_outputs(data_outputs, config, category_ids), config)
        ttbar = _prepare_hist(_sum_outputs(ttbar_outputs, config, category_ids), config)
        qcd = _prepare_hist(_sum_outputs(qcd_outputs, config, category_ids), config)
        if not np.allclose(data.edges, ttbar.edges) or not np.allclose(data.edges, qcd.edges):
            raise ValueError(f"Inconsistent binning after preparation for {variable_name}")

        scale = _qcd_scale(data, ttbar, qcd) if args.scale_qcd_to_data_minus_ttbar else 1.0
        qcd = _apply_scale(qcd, scale)
        fig = plt.figure(figsize=tuple(args.single_figsize), constrained_layout=False)
        outer = fig.add_gridspec(1, 1)
        _plot_panel(
            fig,
            outer[0, 0],
            config,
            data,
            ttbar,
            qcd,
            args.year,
            args.lumi,
        )

        basename = f"leading_jet_{variable_name}_datamc_{args.year}_{group_suffix}"
        png_path = args.out_dir / f"{basename}.png"
        pdf_path = args.out_dir / f"{basename}.pdf"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(png_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
