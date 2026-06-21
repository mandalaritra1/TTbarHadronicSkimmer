#!/usr/bin/env python3
"""Make background-estimate Data/MC plots from 2DAlphabet ROOT inputs.

The plots project the 2D ``mtt_vs_mt`` inputs onto ``mttbar`` in the
low-mass sideband, signal-region mass window, and high-mass sideband for
central/forward pass or fail regions.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from coffea import util


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_DIR = REPO_ROOT / "outputs" / "twodalphabet"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "plots" / "bgest_datamc"

REGION_LABELS = {
    "Cen": "central",
    "Fwd": "forward",
}

MASS_WINDOWS = {
    "low": ("Low-mass sideband", 25.0, 105.0),
    "signal": ("Signal region", 105.0, 210.0),
    "high": ("High-mass sideband", 210.0, 475.0),
}

BACKGROUND_STYLES = {
    "QCD": {"label": "QCD", "color": "#f6c85f"},
    "TTbar": {"label": "SM TTbar", "color": "#b00000"},
}


@dataclass
class Projection:
    values: np.ndarray
    variances: np.ndarray
    edges: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 2DAlphabet ROOT inputs into Data/MC mttbar plots."
    )
    parser.add_argument("--year", default="2024", help="Year label, e.g. 2024.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Directory containing TTbarAllHad*_*.root files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for PNG/PDF outputs.",
    )
    parser.add_argument(
        "--region",
        choices=["Pass", "Fail"],
        default="Fail",
        help="2DAlphabet tag region to plot.",
    )
    parser.add_argument(
        "--mass-windows",
        nargs="*",
        choices=list(MASS_WINDOWS),
        default=list(MASS_WINDOWS),
        help="Mass windows to plot.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=list(REGION_LABELS),
        default=list(REGION_LABELS),
        help="Rapidity categories to plot.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        help="Override data ROOT file. Default is TTbarAllHadYY_Data.root.",
    )
    parser.add_argument(
        "--ttbar-file",
        type=Path,
        help="Override TTbar ROOT file. Default is TTbarAllHadYY_TTbar.root.",
    )
    parser.add_argument(
        "--qcd-file",
        type=Path,
        help="Optional QCD ROOT file. Coffea QCD inputs are preferred if provided.",
    )
    parser.add_argument(
        "--coffea-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "dy",
        help="Directory containing QCD coffea files with the mtt_vs_mt histogram.",
    )
    parser.add_argument(
        "--qcd-coffea-pattern",
        default="QCD*.coffea",
        help="Glob pattern for QCD coffea files inside --coffea-dir.",
    )
    parser.add_argument(
        "--hist",
        default="mtt_vs_mt",
        help="Coffea histogram name to use for QCD projections.",
    )
    parser.add_argument(
        "--mtt-rebin",
        type=int,
        default=1,
        help="Merge this many adjacent mttbar bins after projecting. Use 3 for 300 GeV bins.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=7000.0,
        help="Upper x-axis limit in GeV. Use a non-positive value for the full histogram range.",
    )
    parser.add_argument(
        "--allow-missing-qcd",
        action="store_true",
        help="Make TTbar-only plots if no QCD ROOT/coffea file is found.",
    )
    parser.add_argument(
        "--lumi",
        default="109.95",
        help="Luminosity text in fb^{-1}. Use an empty string to omit.",
    )
    parser.add_argument(
        "--summary-name",
        default=None,
        help="Output basename for the combined grid plot.",
    )
    parser.add_argument(
        "--single-figsize",
        nargs=2,
        type=float,
        default=(12.0, 10.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches for each individual panel.",
    )
    parser.add_argument(
        "--summary-panel-figsize",
        nargs=2,
        type=float,
        default=(12.0, 9.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Approximate per-panel size in inches for the six-panel summary.",
    )
    return parser.parse_args()


def _year_key(year: str) -> str:
    return year.replace("20", "").replace("all", "")


def _default_file(root_dir: Path, year: str, sample: str) -> Path:
    return root_dir / f"TTbarAllHad{_year_key(year)}_{sample}.root"


def _hist_key(category: str, year: str, region: str) -> str:
    return f"MttvsMt{category}{_year_key(year)}{region}"


def _read_projection(root_file: Path, hist_key: str, mt_min: float, mt_max: float) -> Projection:
    with uproot.open(root_file) as root:
        if hist_key not in root:
            available = ", ".join(key.split(";")[0] for key in root.keys()[:8])
            raise KeyError(f"{hist_key!r} not found in {root_file}; first keys: {available}")
        histo = root[hist_key]
        values, mt_edges, mtt_edges = histo.to_numpy(flow=False)
        variances = histo.variances(flow=False)
        if variances is None:
            variances = np.abs(values)

    mt_mask = (mt_edges[:-1] >= mt_min) & (mt_edges[1:] <= mt_max)
    if not np.any(mt_mask):
        raise ValueError(f"No jet-mass bins found inside [{mt_min}, {mt_max}] GeV")

    return Projection(
        values=np.sum(values[mt_mask, :], axis=0),
        variances=np.sum(variances[mt_mask, :], axis=0),
        edges=mtt_edges,
    )


def _rebin_projection(projection: Projection, factor: int) -> Projection:
    if factor <= 1:
        return projection
    if factor < 1:
        raise ValueError("--mtt-rebin must be a positive integer")

    n_bins = len(projection.values)
    n_full_groups = n_bins // factor
    if n_full_groups == 0:
        raise ValueError(f"Cannot rebin {n_bins} bins by factor {factor}")

    kept_bins = n_full_groups * factor
    rebinned_values = projection.values[:kept_bins].reshape(n_full_groups, factor).sum(axis=1)
    rebinned_variances = projection.variances[:kept_bins].reshape(n_full_groups, factor).sum(axis=1)
    rebinned_edges = projection.edges[: kept_bins + 1 : factor]
    if rebinned_edges[-1] != projection.edges[kept_bins]:
        rebinned_edges = np.append(rebinned_edges, projection.edges[kept_bins])

    if kept_bins < n_bins:
        rebinned_values = np.append(rebinned_values, np.sum(projection.values[kept_bins:]))
        rebinned_variances = np.append(rebinned_variances, np.sum(projection.variances[kept_bins:]))
        rebinned_edges = np.append(rebinned_edges, projection.edges[-1])

    return Projection(rebinned_values, rebinned_variances, rebinned_edges)


def _category_fragment(category: str) -> str:
    return {
        "Cen": "cen",
        "Fwd": "fwd",
    }[category]


def _region_fragment(region: str) -> str:
    return {
        "Pass": "2t",
        "Fail": "at",
    }[region]


def _category_ids(label_map: dict[int, str], category: str, region: str) -> list[int]:
    cat_fragment = _category_fragment(category)
    region_fragment = _region_fragment(region)
    ids = [
        idx
        for idx, label in label_map.items()
        if region_fragment in label and cat_fragment in label
    ]
    if not ids:
        raise KeyError(f"No coffea analysis categories matched {category=} and {region=}: {label_map}")
    return ids


def _coffea_projection(
    outputs: list[dict],
    hist_name: str,
    category: str,
    region: str,
    mt_min: float,
    mt_max: float,
    syst: str = "nominal",
) -> Projection:
    pieces = []
    for output in outputs:
        if hist_name not in output:
            raise KeyError(f"Histogram {hist_name!r} missing from one QCD coffea output")
        if "analysisCategories" not in output:
            raise KeyError("QCD coffea output is missing analysisCategories")

        anacat_ids = _category_ids(output["analysisCategories"], category, region)
        histo = output[hist_name][{"systematic": syst, "anacat": anacat_ids}][{"anacat": sum}]

        mt_axis = histo.axes["jetmass"]
        mtt_axis = histo.axes["ttbarmass"]
        mt_edges = mt_axis.edges
        mt_mask = (mt_edges[:-1] >= mt_min) & (mt_edges[1:] <= mt_max)
        if not np.any(mt_mask):
            raise ValueError(f"No jet-mass bins found inside [{mt_min}, {mt_max}] GeV")

        values = histo.values(flow=False)
        variances = histo.variances(flow=False)
        if variances is None:
            variances = np.abs(values)

        pieces.append(
            Projection(
                values=np.sum(values[mt_mask, :], axis=0),
                variances=np.sum(variances[mt_mask, :], axis=0),
                edges=np.asarray(mtt_axis.edges),
            )
        )

    total_values = np.zeros_like(pieces[0].values)
    total_variances = np.zeros_like(pieces[0].variances)
    for piece in pieces:
        total_values += piece.values
        total_variances += piece.variances
    return Projection(total_values, total_variances, pieces[0].edges)


def _load_qcd_coffea(coffea_dir: Path, pattern: str) -> tuple[list[Path], list[dict]]:
    paths = sorted(coffea_dir.expanduser().glob(pattern))
    outputs = [util.load(path) for path in paths]
    return paths, outputs


def _scale_projection(projection: Projection, scale: float) -> Projection:
    return Projection(
        values=projection.values * scale,
        variances=projection.variances * scale * scale,
        edges=projection.edges,
    )


def _syst_keys(root_file: Path, nominal_key: str) -> list[str]:
    if not root_file.exists():
        return []
    pattern = re.compile(rf"^{re.escape(nominal_key)}(.+);")
    keys = []
    with uproot.open(root_file) as root:
        for key in root.keys():
            match = pattern.match(key)
            if match:
                keys.append(key.split(";")[0])
    return keys


def _root_background_projection(
    components: dict[str, Path],
    nominal_key: str,
    mt_min: float,
    mt_max: float,
) -> tuple[dict[str, Projection], Projection, np.ndarray]:
    pieces = {
        name: _read_projection(path, nominal_key, mt_min, mt_max)
        for name, path in components.items()
    }

    first = next(iter(pieces.values()))
    total_values = np.zeros_like(first.values)
    total_variances = np.zeros_like(first.variances)
    syst_up2 = np.zeros_like(first.values)
    syst_down2 = np.zeros_like(first.values)

    for name, projection in pieces.items():
        total_values += projection.values
        total_variances += projection.variances

        for syst_key in _syst_keys(components[name], nominal_key):
            shifted = _read_projection(components[name], syst_key, mt_min, mt_max)
            delta = shifted.values - projection.values
            syst_up2 += np.square(np.clip(delta, 0.0, None))
            syst_down2 += np.square(np.clip(delta, None, 0.0))

    total_unc = np.sqrt(total_variances + syst_up2 + syst_down2)
    total = Projection(total_values, total_variances, first.edges)
    return pieces, total, total_unc


def _make_background(
    ttbar_file: Path,
    qcd_file: Path | None,
    qcd_outputs: list[dict],
    data: Projection,
    nominal_key: str,
    category: str,
    region: str,
    hist_name: str,
    mt_min: float,
    mt_max: float,
) -> tuple[dict[str, Projection], Projection, np.ndarray, float]:
    ttbar_pieces, ttbar_total, ttbar_unc = _root_background_projection(
        {"TTbar": ttbar_file},
        nominal_key,
        mt_min,
        mt_max,
    )

    if qcd_outputs:
        qcd_nominal = _coffea_projection(qcd_outputs, hist_name, category, region, mt_min, mt_max)
    elif qcd_file is not None:
        qcd_nominal = _read_projection(qcd_file, nominal_key, mt_min, mt_max)
    else:
        total = Projection(ttbar_total.values.copy(), ttbar_total.variances.copy(), ttbar_total.edges)
        return ttbar_pieces, total, ttbar_unc, float("nan")

    target_qcd = np.sum(data.values - ttbar_total.values)
    qcd_integral = np.sum(qcd_nominal.values)
    if qcd_integral <= 0:
        raise ValueError(f"QCD integral is non-positive for {category} {region} {mt_min:g}-{mt_max:g} GeV")
    if target_qcd < 0:
        print(
            f"warning: Data - TTbar is negative for {category} {region} "
            f"{mt_min:g}-{mt_max:g} GeV; setting QCD scale to 0."
        )
        qcd_scale = 0.0
    else:
        qcd_scale = target_qcd / qcd_integral

    qcd = _scale_projection(qcd_nominal, qcd_scale)
    backgrounds = {"QCD": qcd, **ttbar_pieces}
    total_values = ttbar_total.values + qcd.values
    total_variances = ttbar_total.variances + qcd.variances
    total = Projection(total_values, total_variances, ttbar_total.edges)
    total_unc = np.sqrt(np.square(ttbar_unc) + qcd.variances)
    return backgrounds, total, total_unc, qcd_scale


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _positive_ylim(values: list[np.ndarray]) -> tuple[float, float]:
    combined = np.concatenate([array[array > 0] for array in values if np.any(array > 0)])
    if combined.size == 0:
        return 0.5, 2.0
    ymin = max(0.05, np.min(combined) * 0.3)
    ymax = np.max(combined) * 30.0
    return ymin, ymax


def _draw_datamc(
    ax_main,
    ax_ratio,
    data: Projection,
    backgrounds: dict[str, Projection],
    total_bkg: Projection,
    total_unc: np.ndarray,
    title: str,
    subtitle: str,
    lumi: str,
    year: str,
    x_max: float,
) -> None:
    edges = total_bkg.edges
    centers = _bin_centers(edges)
    width = np.diff(edges)
    bottom = np.zeros_like(total_bkg.values)
    x_min = edges[0]
    x_max = x_max if x_max and x_max > 0 else edges[-1]

    for name in ("TTbar", "QCD"):
        if name not in backgrounds:
            continue
        style = BACKGROUND_STYLES[name]
        vals = backgrounds[name].values
        ax_main.stairs(bottom + vals, edges, baseline=bottom, fill=True, color=style["color"], label=style["label"])
        bottom = bottom + vals

    data_err = np.sqrt(np.clip(data.variances, 0.0, None))
    ax_main.errorbar(
        centers,
        data.values,
        yerr=data_err,
        xerr=0.5 * width,
        fmt="o",
        color="black",
        markersize=2.5,
        linewidth=0.8,
        label="Data",
    )
    ax_main.fill_between(
        centers,
        np.clip(total_bkg.values - total_unc, 0.0, None),
        total_bkg.values + total_unc,
        step="mid",
        color="0.55",
        alpha=0.35,
        hatch="///",
        linewidth=0,
        label="Total bkg unc.",
    )

    hep.cms.label("Work in Progress", data=True, lumi=(lumi or None), year=year, ax=ax_main, fontsize=20)
    ax_main.text(
        0.04,
        0.90,
        subtitle,
        transform=ax_main.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    ax_main.text(0.98, 0.08, title, transform=ax_main.transAxes, ha="right", weight="bold")
    ax_main.set_yscale("log")
    ax_main.set_ylim(*_positive_ylim([data.values, total_bkg.values]))
    bin_widths = np.diff(edges)
    width_label = f"{bin_widths[0]:g} GeV" if np.allclose(bin_widths, bin_widths[0]) else "bin"
    ax_main.set_ylabel(f"Events / {width_label}")
    ax_main.set_xlim(x_min, x_max)
    ax_main.legend(loc="upper right", frameon=False)

    ratio = np.divide(data.values, total_bkg.values, out=np.full_like(data.values, np.nan), where=total_bkg.values > 0)
    ratio_err = np.divide(data_err, total_bkg.values, out=np.zeros_like(data.values), where=total_bkg.values > 0)
    ratio_unc = np.divide(total_unc, total_bkg.values, out=np.zeros_like(total_unc), where=total_bkg.values > 0)

    ax_ratio.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        xerr=0.5 * width,
        fmt="o",
        color="black",
        markersize=2.0,
        linewidth=0.7,
    )
    ax_ratio.fill_between(
        centers,
        1.0 - ratio_unc,
        1.0 + ratio_unc,
        step="mid",
        color="0.55",
        alpha=0.35,
        hatch="///",
        linewidth=0,
    )
    ax_ratio.axhline(1.0, color="black", linewidth=0.8)
    ax_ratio.set_ylim(0.0, 2.0)
    ax_ratio.set_xlim(x_min, x_max)
    ax_ratio.set_ylabel("Data/Bkg")
    ax_ratio.set_xlabel(r"$m_{t\bar{t}}$ [GeV]")


def _plot_one(
    output_base: Path,
    data: Projection,
    backgrounds: dict[str, Projection],
    total_bkg: Projection,
    total_unc: np.ndarray,
    title: str,
    subtitle: str,
    lumi: str,
    year: str,
    x_max: float,
    figsize: tuple[float, float],
) -> None:
    fig, (ax_main, ax_ratio) = plt.subplots(
        nrows=2,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
    )
    _draw_datamc(ax_main, ax_ratio, data, backgrounds, total_bkg, total_unc, title, subtitle, lumi, year, x_max)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def _plot_summary(
    output_base: Path,
    plot_inputs: dict[tuple[str, str], tuple[Projection, dict[str, Projection], Projection, np.ndarray, float]],
    args: argparse.Namespace,
) -> None:
    nrows = len(args.categories)
    ncols = len(args.mass_windows)
    panel_width, panel_height = args.summary_panel_figsize
    fig = plt.figure(figsize=(panel_width * ncols + 1.3, panel_height * nrows + 1.3))
    outer = fig.add_gridspec(nrows, ncols, left=0.12, right=0.98, top=0.86, bottom=0.08, wspace=0.24, hspace=0.30)

    fig.suptitle(f"Background Estimate: {args.year} {args.region}", x=0.03, y=0.98, ha="left", fontsize=22)

    for col, window_name in enumerate(args.mass_windows):
        window_title = MASS_WINDOWS[window_name][0]
        fig.text(0.12 + (col + 0.5) * 0.86 / ncols, 0.9, window_title, ha="center", fontsize=18)

    for row, category in enumerate(args.categories):
        fig.text(0.025, 0.73 - row * (0.78 / max(nrows, 1)), REGION_LABELS[category], ha="left", fontsize=16)
        for col, window_name in enumerate(args.mass_windows):
            inner = outer[row, col].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.05)
            ax_main = fig.add_subplot(inner[0])
            ax_ratio = fig.add_subplot(inner[1], sharex=ax_main)
            data, backgrounds, total_bkg, total_unc, _ = plot_inputs[(category, window_name)]
            _, mt_min, mt_max = MASS_WINDOWS[window_name]
            subtitle = f"{mt_min:g} < $m_j$ [GeV] < {mt_max:g}"
            _draw_datamc(
                ax_main,
                ax_ratio,
                data,
                backgrounds,
                total_bkg,
                total_unc,
                REGION_LABELS[category],
                subtitle,
                args.lumi,
                args.year,
                args.x_max,
            )
            plt.setp(ax_main.get_xticklabels(), visible=False)

    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    hep.style.use("CMS")

    data_file = args.data_file or _default_file(args.root_dir, args.year, "Data")
    ttbar_file = args.ttbar_file or _default_file(args.root_dir, args.year, "TTbar")
    qcd_file = args.qcd_file or None

    for required in (data_file, ttbar_file):
        if not required.exists():
            raise FileNotFoundError(required)

    qcd_paths, qcd_outputs = _load_qcd_coffea(args.coffea_dir, args.qcd_coffea_pattern)
    if qcd_paths:
        print("QCD coffea inputs:")
        for path in qcd_paths:
            print(f"  {path}")
    else:
        default_qcd_file = _default_file(args.root_dir, args.year, "QCD")
        qcd_file = qcd_file or (default_qcd_file if default_qcd_file.exists() else None)
        if qcd_file is None and not args.allow_missing_qcd:
            raise FileNotFoundError(
                f"No QCD coffea files matched {args.coffea_dir / args.qcd_coffea_pattern} "
                f"and no QCD ROOT file was provided. Pass --qcd-coffea-pattern/--qcd-file, "
                "or use --allow-missing-qcd for TTbar-only plots."
            )
        if qcd_file is not None:
            print(f"QCD ROOT input: {qcd_file}")
        else:
            print("warning: no QCD input found; plotting TTbar-only MC.")

    output_dir = args.out_dir / args.year / args.region.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_inputs = {}
    for category in args.categories:
        nominal_key = _hist_key(category, args.year, args.region)
        for window_name in args.mass_windows:
            window_title, mt_min, mt_max = MASS_WINDOWS[window_name]
            data_native = _read_projection(data_file, nominal_key, mt_min, mt_max)
            bkg_pieces, total_bkg, total_unc, qcd_scale = _make_background(
                ttbar_file,
                qcd_file,
                qcd_outputs,
                data_native,
                nominal_key,
                category,
                args.region,
                args.hist,
                mt_min,
                mt_max,
            )
            total_unc_projection = Projection(
                values=total_unc,
                variances=np.square(total_unc),
                edges=total_bkg.edges,
            )
            bkg_pieces = {
                name: _rebin_projection(projection, args.mtt_rebin)
                for name, projection in bkg_pieces.items()
            }
            data = _rebin_projection(data_native, args.mtt_rebin)
            total_bkg = _rebin_projection(total_bkg, args.mtt_rebin)
            total_unc = np.sqrt(_rebin_projection(total_unc_projection, args.mtt_rebin).variances)
            plot_inputs[(category, window_name)] = (data, bkg_pieces, total_bkg, total_unc, qcd_scale)
            if np.isfinite(qcd_scale):
                print(
                    f"QCD scale {category} {args.region} {window_name}: "
                    f"{qcd_scale:.6g}"
                )

            category_label = REGION_LABELS[category]
            basename = f"{args.region.lower()}_{category_label}_{window_name}"
            subtitle = f"{mt_min:g} < $m_j$ [GeV] < {mt_max:g}"
            _plot_one(
                output_dir / basename,
                data,
                bkg_pieces,
                total_bkg,
                total_unc,
                category_label,
                subtitle,
                args.lumi,
                args.year,
                args.x_max,
                tuple(args.single_figsize),
            )

    summary_name = args.summary_name or f"background_estimate_{args.year}_{args.region.lower()}"
    _plot_summary(output_dir / summary_name, plot_inputs, args)
    print(f"saved plots under {output_dir}")


if __name__ == "__main__":
    main()
