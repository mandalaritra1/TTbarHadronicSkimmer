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

    data_paths = _discover_files(coffea_dir, data_pattern, "data")
    qcd_paths = _discover_files(coffea_dir, qcd_pattern, "QCD pT-bin", sort_qcd=True)

    data_outputs = _load_outputs(data_paths, "data", util)
    qcd_outputs = _load_outputs(qcd_paths, "QCD pT bins", util)

    ref_output = _first_output_with_categories(data_outputs + qcd_outputs)
    label_map = ref_output["analysisCategories"]
    label_to_int = {label: idx for idx, label in label_map.items()}
    print("Analysis categories:", label_map)

    syst_labels = _available_systematics(data_outputs + qcd_outputs, args.hist)
    if not args.include_systs:
        syst_labels = ["nominal"]
    print("Systematics:", ", ".join(syst_labels))

    year_label = _year_label(args.year)
    file_prefix = out_dir / f"TTbarAllHad{year_label}_"
    data_out = file_prefix.with_name(file_prefix.name + f"Data{args.tag}.root")
    qcd_out = file_prefix.with_name(file_prefix.name + f"QCD{args.tag}.root")

    with uproot.recreate(data_out) as fdata, uproot.recreate(qcd_out) as fqcd:
        for cat in args.categories:
            pass_ids = _category_ids(label_to_int, "2t", cat)
            fail_ids = _category_ids(label_to_int, "at", cat)
            print(f"{cat}: pass={pass_ids}, fail={fail_ids}")

            for syst in syst_labels:
                suffix = _syst_suffix(syst)
                pass_name = f"MttvsMt{cat}{year_label}Pass{suffix}"
                fail_name = f"MttvsMt{cat}{year_label}Fail{suffix}"

                fdata[pass_name] = _sum_hists(data_outputs, args.hist, pass_ids, syst)
                fdata[fail_name] = _sum_hists(data_outputs, args.hist, fail_ids, syst)
                fqcd[pass_name] = _sum_hists(qcd_outputs, args.hist, pass_ids, syst)
                fqcd[fail_name] = _sum_hists(qcd_outputs, args.hist, fail_ids, syst)

    print(f"Saved {data_out}")
    print(f"Saved {qcd_out}")
    _print_time(time.time() - tic)


if __name__ == "__main__":
    main()
