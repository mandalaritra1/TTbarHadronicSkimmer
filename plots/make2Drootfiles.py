#!/usr/bin/env python3
"""Build 2DAlphabet ROOT inputs from coffea ``mtt_vs_mt`` histograms.

The default mode is tuned for the current Run-3 2024 workflow:

  * data is read from ``outputs/dy`` and summed across era files matching
    ``data_2024*_noSyst.coffea``
  * QCD is read from ``outputs/dy`` and summed across pT-bin files matching
    ``QCD_2024*_PT-*to*_noSyst.coffea``
  * output histograms are written for central/forward pass/fail regions
    under ``outputs/twodalphabet``

Examples:
    python plots/make2Drootfiles.py
    python plots/make2Drootfiles.py --year 2024 --include-systs
    python plots/make2Drootfiles.py --qcd-pattern "QCD_2024*_PT-*to*.coffea"
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.append(str(REPO_ROOT / "python"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write summed data and QCD mtt_vs_mt TH2 inputs for 2DAlphabet."
    )
    parser.add_argument("--year", default="2024", help="Year label used in input/output names.")
    parser.add_argument(
        "--coffea-dir",
        default=str(REPO_ROOT / "outputs" / "dy"),
        help="Directory containing coffea outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: outputs/twodalphabet.",
    )
    parser.add_argument(
        "--data-pattern",
        default=None,
        help="Glob for data coffea files. Default: data_<year>*_noSyst.coffea.",
    )
    parser.add_argument(
        "--qcd-pattern",
        default=None,
        help="Glob for QCD pT-bin coffea files. Default: QCD_<year>*_PT-*to*_noSyst.coffea.",
    )
    parser.add_argument(
        "--skip-qcd",
        action="store_true",
        help="Do not load QCD or write the QCD ROOT file (data-only output).",
    )
    parser.add_argument(
        "--ttbar-pattern",
        default=None,
        help="Glob for TTbar coffea files. Default: TTbar_<year>*.coffea. Empty string disables.",
    )
    parser.add_argument(
        "--signal-pattern",
        default=None,
        help=(
            "Glob for one legacy combined signal coffea group. "
            "Default: ZPrime*_<year>*.coffea. Empty string disables."
        ),
    )
    parser.add_argument(
        "--signal-label",
        default="signal",
        help="Label used with --signal-pattern in the signal output filename.",
    )
    parser.add_argument(
        "--signal-points",
        nargs="+",
        default=None,
        help=(
            "Exact ZPrime points to export separately as MASS or MASS:WIDTH, "
            "e.g. 900 1000 4000 4000:10 4000:30. Missing width defaults to 1."
        ),
    )
    parser.add_argument(
        "--signal-masses",
        nargs="+",
        default=None,
        help=(
            "ZPrime mass labels to export separately, e.g. 900 1000 4000. "
            "Uses input pattern ZPrime<MASS>_<WIDTH>_<year>*.coffea."
        ),
    )
    parser.add_argument(
        "--signal-widths",
        nargs="+",
        default=["1"],
        help=(
            "ZPrime width labels to export for each requested mass. "
            "Default: 1. The 1%% width omits the width from the output label."
        ),
    )
    parser.add_argument(
        "--hist",
        default="mtt_vs_mt",
        help="Histogram key to export.",
    )
    parser.add_argument(
        "--include-systs",
        action="store_true",
        help="Write all systematic labels available in the inputs. Default writes nominal only.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for output ROOT filenames, e.g. _test.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cen", "fwd"],
        help="Category substrings to export from analysisCategories.",
    )
    return parser.parse_args()


def _qcd_pt_sort_key(path: Path) -> tuple[float, float, str]:
    match = re.search(r"(?:QCD_)?PT-(\d+)to(\d+|Inf)", path.name)
    if match is None:
        return (float("inf"), float("inf"), path.name)
    low = int(match.group(1))
    high = float("inf") if match.group(2) == "Inf" else int(match.group(2))
    return (low, high, path.name)


def _discover_files(coffea_dir: Path, pattern: str, label: str, sort_qcd: bool = False) -> list[Path]:
    paths = list(coffea_dir.glob(pattern))
    paths = sorted(paths, key=_qcd_pt_sort_key if sort_qcd else lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No {label} files found in {coffea_dir} matching {pattern!r}")
    return paths


def _normalize_zprime_field(value: str, field_name: str) -> str:
    label = str(value).strip()
    if label.endswith("%"):
        label = label[:-1]
    if not label:
        raise ValueError(f"Empty ZPrime {field_name} label")
    if any(char in label for char in "/\\_"):
        raise ValueError(
            f"ZPrime {field_name} label {value!r} cannot contain path or field separators"
        )
    return label


def _zprime_signal_specs(
    masses: list[str],
    widths: list[str],
    year: str,
) -> list[tuple[str, str]]:
    """Return ``(output_label, input_glob)`` pairs for requested ZPrime signals."""

    specs = []
    for mass_value in masses:
        mass = _normalize_zprime_field(mass_value, "mass")
        for width_value in widths:
            width = _normalize_zprime_field(width_value, "width")
            output_label = f"signalZPrime{mass}"
            if width != "1":
                output_label += f"_{width}"
            input_pattern = f"ZPrime{mass}_{width}_{year}*.coffea"
            specs.append((output_label, input_pattern))
    return specs


def _zprime_signal_point_specs(points: list[str], year: str) -> list[tuple[str, str]]:
    specs = []
    for point in points:
        fields = str(point).split(":", maxsplit=1)
        mass = fields[0]
        width = fields[1] if len(fields) == 2 else "1"
        specs.extend(_zprime_signal_specs([mass], [width], year))
    return specs


def _check_unique_signal_specs(specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen_patterns: dict[str, str] = {}
    for output_label, input_pattern in specs:
        if output_label in seen_patterns:
            raise ValueError(
                f"Duplicate signal output label {output_label!r} from patterns "
                f"{seen_patterns[output_label]!r} and {input_pattern!r}"
            )
        seen_patterns[output_label] = input_pattern
    return specs


def _load_outputs(paths: list[Path], label: str, util_module) -> list[dict]:
    print(f"Loading {label}:")
    outputs = []
    for path in paths:
        print(f"  {path}")
        outputs.append(util_module.load(path))
    return outputs


def _first_output_with_categories(outputs: list[dict]) -> dict:
    for output in outputs:
        if "analysisCategories" in output:
            return output
    raise KeyError("Could not find analysisCategories in any input coffea file")


def _category_ids(label_to_int: dict[str, int], region: str, cat: str) -> list[int]:
    ids = [
        idx
        for label, idx in label_to_int.items()
        if region in label and (cat == "" or cat in label)
    ]
    if not ids:
        raise KeyError(f"No analysis categories matched region={region!r}, category={cat!r}")
    return ids


def _available_systematics(outputs: list[dict], hist_name: str) -> list[str]:
    for output in outputs:
        if hist_name not in output:
            continue
        for axis in output[hist_name].axes:
            if axis.name == "systematic":
                return [str(value) for value in axis]
    raise KeyError(f"Could not find histogram {hist_name!r} with a systematic axis")


def _sum_hists(outputs: list[dict], hist_name: str, anacat_ids: list[int], syst: str):
    pieces = []
    for output in outputs:
        if hist_name not in output:
            raise KeyError(f"Histogram {hist_name!r} missing from one input")
        pieces.append(
            output[hist_name][{"anacat": anacat_ids, "systematic": syst}][{"anacat": sum}]
        )

    total = pieces[0]
    for histo in pieces[1:]:
        total = total + histo
    return total


def _syst_suffix(syst: str) -> str:
    if syst == "nominal":
        return ""
    if syst.endswith("Up"):
        return syst[:-2].upper() + "up"
    if syst.endswith("Down"):
        return syst[:-4].upper() + "down"
    return syst


def _year_label(year: str) -> str:
    return year.replace("20", "").replace("all", "")


def _root_category_label(cat: str) -> str:
    """Return the 2DAlphabet ROOT-key category label for an analysis category."""
    labels = {
        "cen": "Cen",
        "fwd": "Fwd",
    }
    return labels.get(cat, cat)


def _print_time(seconds: float) -> None:
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 1:
        print(f"time: {int(hours)}h {int(minutes)}m {sec:.1f}s")
    elif minutes >= 1:
        print(f"time: {int(minutes)}m {sec:.1f}s")
    else:
        print(f"time: {sec:.1f}s")


def main() -> None:
    tic = time.time()
    args = _parse_args()

    import uproot
    from coffea import util

    coffea_dir = Path(args.coffea_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else REPO_ROOT / "outputs" / "twodalphabet"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_pattern = args.data_pattern or f"data_{args.year}*_noSyst.coffea"
    qcd_pattern = args.qcd_pattern or f"QCD_{args.year}*_PT-*to*_noSyst.coffea"
    ttbar_pattern = args.ttbar_pattern if args.ttbar_pattern is not None else f"TTbar_{args.year}*.coffea"
    signal_pattern = args.signal_pattern if args.signal_pattern is not None else f"ZPrime*_{args.year}*.coffea"

    samples: list[tuple[str, list[dict]]] = []

    data_paths = _discover_files(coffea_dir, data_pattern, "data")
    data_outputs = _load_outputs(data_paths, "data", util)
    samples.append(("Data", data_outputs))

    if not args.skip_qcd:
        qcd_paths = _discover_files(coffea_dir, qcd_pattern, "QCD pT-bin", sort_qcd=True)
        qcd_outputs = _load_outputs(qcd_paths, "QCD pT bins", util)
        samples.append(("QCD", qcd_outputs))

    if ttbar_pattern:
        ttbar_paths = _discover_files(coffea_dir, ttbar_pattern, "TTbar")
        ttbar_outputs = _load_outputs(ttbar_paths, "TTbar", util)
        samples.append(("TTbar", ttbar_outputs))

    if args.signal_points or args.signal_masses:
        if args.signal_pattern is not None:
            raise ValueError("--signal-pattern cannot be combined with requested ZPrime points")
        if args.signal_label != "signal":
            raise ValueError("--signal-label cannot be combined with requested ZPrime points")
        if args.signal_points and args.signal_masses:
            raise ValueError("--signal-points cannot be combined with --signal-masses")
        signal_specs = _check_unique_signal_specs(
            _zprime_signal_point_specs(args.signal_points, args.year)
            if args.signal_points
            else _zprime_signal_specs(args.signal_masses, args.signal_widths, args.year)
        )
        for signal_label, signal_pattern_for_point in signal_specs:
            signal_paths = _discover_files(
                coffea_dir,
                signal_pattern_for_point,
                signal_label,
            )
            signal_outputs = _load_outputs(signal_paths, signal_label, util)
            samples.append((signal_label, signal_outputs))
    elif signal_pattern:
        signal_paths = _discover_files(coffea_dir, signal_pattern, "signal")
        signal_outputs = _load_outputs(signal_paths, "signal", util)
        samples.append((args.signal_label, signal_outputs))

    all_outputs = [out for _, outs in samples for out in outs]
    ref_output = _first_output_with_categories(all_outputs)
    label_map = ref_output["analysisCategories"]
    label_to_int = {label: idx for idx, label in label_map.items()}
    print("Analysis categories:", label_map)

    syst_labels = _available_systematics(all_outputs, args.hist)
    if not args.include_systs:
        syst_labels = ["nominal"]
    print("Systematics:", ", ".join(syst_labels))

    year_label = _year_label(args.year)
    file_prefix = out_dir / f"TTbarAllHad{year_label}_"

    cat_ids = {
        cat: (_category_ids(label_to_int, "2t", cat), _category_ids(label_to_int, "at", cat))
        for cat in args.categories
    }
    for cat, (pass_ids, fail_ids) in cat_ids.items():
        print(f"{cat}: pass={pass_ids}, fail={fail_ids}")

    written: list[Path] = []
    for label, outputs in samples:
        out_path = file_prefix.with_name(file_prefix.name + f"{label}{args.tag}.root")
        sample_systs = ["nominal"] if label == "Data" else syst_labels
        with uproot.recreate(out_path) as fout:
            for cat, (pass_ids, fail_ids) in cat_ids.items():
                root_cat = _root_category_label(cat)
                for syst in sample_systs:
                    suffix = _syst_suffix(syst)
                    pass_name = f"MttvsMt{root_cat}{year_label}Pass{suffix}"
                    fail_name = f"MttvsMt{root_cat}{year_label}Fail{suffix}"
                    fout[pass_name] = _sum_hists(outputs, args.hist, pass_ids, syst)
                    fout[fail_name] = _sum_hists(outputs, args.hist, fail_ids, syst)
        written.append(out_path)
        print(f"Saved {out_path}")

    _print_time(time.time() - tic)


if __name__ == "__main__":
    main()
