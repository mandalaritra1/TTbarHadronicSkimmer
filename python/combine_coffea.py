from __future__ import annotations

import argparse
import copy
import glob
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from coffea import processor, util
from hist import Hist


def _as_paths(paths: Iterable[str | Path]) -> list[Path]:
    selected = [Path(path).expanduser() for path in paths]
    missing = [str(path) for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing coffea files:\n" + "\n".join(missing))
    if not selected:
        raise ValueError("No input coffea files were provided.")
    return selected


_METADATA_KEYS = {
    "analysisCategories",
    "cutflow_table_steps",
    "sample_metadata",
    "normalization",
    "provenance",
}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, (dict, processor.defaultdict_accumulator, processor.dict_accumulator))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, processor.list_accumulator))


def _combine_sequence(left: Any, right: Any) -> Any:
    if isinstance(left, processor.list_accumulator) or isinstance(right, processor.list_accumulator):
        return processor.list_accumulator(list(left) + list(right))
    if isinstance(left, tuple):
        return tuple(left) + tuple(right)
    return list(left) + list(right)


def _axis_signature(axis: Any) -> tuple[Any, ...]:
    if hasattr(axis, "edges"):
        return (
            type(axis).__name__,
            axis.name,
            tuple(np.asarray(axis.edges, dtype=float).tolist()),
        )
    return (
        type(axis).__name__,
        axis.name,
        tuple(axis),
    )


def _hist_axis_summary(histogram: Hist) -> list[tuple[Any, ...]]:
    return [_axis_signature(axis) for axis in histogram.axes]


def _hist_axes_match(left: Hist, right: Hist) -> bool:
    return _hist_axis_summary(left) == _hist_axis_summary(right)


def _add_hist_values(left: Hist, right: Hist) -> Hist:
    if not _hist_axes_match(left, right):
        raise ValueError(
            "histogram axes differ\n"
            f"left axes: {_hist_axis_summary(left)}\n"
            f"right axes: {_hist_axis_summary(right)}"
        )

    result = left.copy(deep=True)
    result_view = result.view(flow=True)
    right_view = right.view(flow=True)

    if hasattr(result_view, "value") and hasattr(right_view, "value"):
        result_view.value[...] = result_view.value + right_view.value
        if hasattr(result_view, "variance") and hasattr(right_view, "variance"):
            result_view.variance[...] = result_view.variance + right_view.variance
    else:
        result_view[...] = result_view + right_view

    return result


def _add_mapping_values(left: dict[Any, Any], right: dict[Any, Any], source: Path) -> dict[Any, Any]:
    result = copy.deepcopy(left)
    for key, value in right.items():
        if key in result:
            result[key] = _combine_value(str(key), result[key], value, source)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _combine_value(key: str, left: Any, right: Any, source: Path) -> Any:
    if key in _METADATA_KEYS:
        return left

    if isinstance(left, Hist) and isinstance(right, Hist):
        try:
            return left + right
        except Exception as exc:
            try:
                return _add_hist_values(left, right)
            except Exception as fallback_exc:
                raise ValueError(
                    f"Histogram {key!r} is not compatible in {source}.\n"
                    f"Original add error: {exc}\n"
                    f"Fallback add error: {fallback_exc}"
                ) from fallback_exc

    if _is_mapping(left) and _is_mapping(right):
        return _add_mapping_values(left, right, source)

    if _is_sequence(left) and _is_sequence(right):
        return _combine_sequence(left, right)

    if isinstance(left, processor.column_accumulator) and isinstance(
        right, processor.column_accumulator
    ):
        return processor.column_accumulator(np.concatenate([left.value, right.value]))

    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.concatenate([left, right])

    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return left + right

    return left


def combine_coffea_outputs(
    input_files: Iterable[str | Path],
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    """Combine top-level histograms and accumulator counters from coffea outputs.

    The first file is used as the template. Compatible ``hist.Hist`` objects are
    added bin-by-bin, cutflow/weight-style accumulators are added by key, and
    metadata-like objects such as ``analysisCategories`` are kept from the first
    file. If ``output_file`` is provided, the combined object is written with
    ``coffea.util.save``.
    """

    paths = _as_paths(input_files)
    combined = copy.deepcopy(util.load(paths[0]))
    if not isinstance(combined, dict):
        raise TypeError(f"{paths[0]} did not load to a dictionary-like coffea output.")
    combined_metadata = [
        {
            "file": str(paths[0]),
            "sample_metadata": copy.deepcopy(combined.get("sample_metadata", {})),
            "normalization": copy.deepcopy(combined.get("normalization", {})),
        }
    ]

    for source in paths[1:]:
        current = util.load(source)
        if not isinstance(current, dict):
            raise TypeError(f"{source} did not load to a dictionary-like coffea output.")
        combined_metadata.append(
            {
                "file": str(source),
                "sample_metadata": copy.deepcopy(current.get("sample_metadata", {})),
                "normalization": copy.deepcopy(current.get("normalization", {})),
            }
        )

        for key, value in current.items():
            if key not in combined:
                combined[key] = copy.deepcopy(value)
                continue
            combined[key] = _combine_value(key, combined[key], value, source)

    combined["combined_inputs"] = [str(path) for path in paths]
    combined["combined_metadata"] = combined_metadata

    if output_file is not None:
        output_path = Path(output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        util.save(combined, output_path)

    return combined


def _expand_input_patterns(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?[]"):
            matches = sorted(Path(path) for path in glob.glob(pattern))
            files.extend(matches)
        else:
            files.append(Path(pattern))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine QCD coffea outputs by summing their histograms and counters."
    )
    parser.add_argument("inputs", nargs="+", help="Input .coffea files or glob patterns")
    parser.add_argument("-o", "--output", required=True, help="Output .coffea file")
    args = parser.parse_args()

    input_files = _expand_input_patterns(args.inputs)
    combined = combine_coffea_outputs(input_files, args.output)
    hist_count = sum(isinstance(value, Hist) for value in combined.values())
    print(f"Wrote {args.output}")
    print(f"Combined {len(input_files)} files and {hist_count} histograms")


if __name__ == "__main__":
    main()
