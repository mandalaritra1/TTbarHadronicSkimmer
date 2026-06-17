#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from cutflow import (  # noqa: E402
    build_cutflow_matrix,
    build_cutflow_rows,
    cutflow_normalization_text,
    format_latex_matrix,
    format_latex_table,
    format_markdown_matrix,
    format_markdown_table,
)


def _load_coffea(path):
    try:
        from coffea import util
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "coffea is required to read .coffea files. "
            "Run this inside the same analysis environment used for processing."
        ) from exc
    return util.load(path)


def _default_title(path, title):
    if title:
        return title
    return Path(path).name


def _render_one(path, args, output_format):
    output = _load_coffea(path)
    rows = build_cutflow_rows(
        output,
        weight_mode=args.weight_mode,
        include_zero=args.include_zero,
    )
    normalization = None
    if not args.no_normalization_note and args.weight_mode == "scaled":
        normalization = cutflow_normalization_text(output)
    title = _default_title(path, args.title)

    if output_format == "markdown":
        return format_markdown_table(rows, title=title, normalization=normalization)

    caption = args.caption or f"Cutflow table for {Path(path).name}"
    return format_latex_table(
        rows,
        caption=caption,
        label=args.label,
        normalization=normalization,
    )


def _render_all(paths, args, output_format):
    rendered = [_render_one(path, args, output_format) for path in paths]
    separator = "\n\n" if output_format == "markdown" else "\n\n"
    return separator.join(rendered)


def _derive_column_label(path):
    """Best-effort column label from an output filename.

    ZPrime outputs are named ``ZPrime{mass}_{width}_{IOV}.coffea`` -> "{mass} GeV".
    Anything else falls back to the file stem.
    """

    stem = Path(path).stem
    match = re.search(r"ZPrime(\d+)_", stem)
    if match:
        return f"{match.group(1)} GeV"
    return stem


def _render_matrix(paths, args, output_format):
    outputs = [_load_coffea(path) for path in paths]
    if args.labels:
        labels = [label.strip() for label in args.labels.split(",")]
        if len(labels) != len(paths):
            raise ValueError(
                f"--labels has {len(labels)} entries but {len(paths)} input file(s) given."
            )
    else:
        labels = [_derive_column_label(path) for path in paths]

    matrix = build_cutflow_matrix(
        outputs,
        labels=labels,
        value=args.matrix_value,
        include_zero=args.include_zero,
        add_total=args.total,
    )

    if output_format == "markdown":
        return format_markdown_matrix(matrix, title=args.title)

    caption = args.caption or "Cutflow table"
    return format_latex_matrix(matrix, caption=caption, label=args.label)


def _write_or_print(text, output_path):
    if output_path is None:
        print(text)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n")
    print(f"Wrote {output_path}")


def _both_output_paths(output):
    if output is None:
        return None, None
    output = Path(output)
    if output.suffix in {".md", ".markdown"}:
        return output, output.with_suffix(".tex")
    if output.suffix == ".tex":
        return output.with_suffix(".md"), output
    return output.with_suffix(".md"), output.with_suffix(".tex")


def main():
    parser = argparse.ArgumentParser(
        description="Render TTbarHadronicSkimmer cutflow tables from .coffea outputs."
    )
    parser.add_argument("inputs", nargs="+", help="Input .coffea file(s)")
    parser.add_argument(
        "--format",
        choices=["markdown", "latex", "both"],
        default="markdown",
        help="Output table format.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["scaled", "raw", "none"],
        default="scaled",
        help=(
            "Weighted column to show. 'scaled' uses postprocess normalization when available, "
            "'raw' uses raw generator/LHE sumw, and 'none' hides the weighted column."
        ),
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Keep zero-yield rows in the table.",
    )
    parser.add_argument(
        "--no-normalization-note",
        action="store_true",
        help="Do not print the normalization note above/below the table.",
    )
    parser.add_argument("--title", help="Markdown section title. Defaults to each input filename.")
    parser.add_argument("--caption", help="LaTeX table caption.")
    parser.add_argument("--label", help="LaTeX table label.")
    parser.add_argument("-o", "--output", help="Output file. With --format both, this is used as a stem.")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help=(
            "Lay all inputs side by side as one multi-column table (column per file) "
            "instead of one table per file. Use for per-year (data) or per-mass (signal) cutflows."
        ),
    )
    parser.add_argument(
        "--matrix-value",
        choices=["events", "yield"],
        default="events",
        help="Matrix cell value: 'events' (unweighted counts) or 'yield' (scaled weighted yields).",
    )
    parser.add_argument(
        "--labels",
        help="Comma-separated column labels for --matrix (one per input). Defaults to mass/filename.",
    )
    parser.add_argument(
        "--total",
        action="store_true",
        help="In --matrix mode, append a Total column summing across columns (use for raw counts).",
    )
    args = parser.parse_args()

    input_paths = [Path(path).expanduser() for path in args.inputs]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input .coffea file(s):\n" + "\n".join(missing))

    render = _render_matrix if args.matrix else _render_all

    if args.format == "both":
        markdown_path, latex_path = _both_output_paths(args.output)
        _write_or_print(render(input_paths, args, "markdown"), markdown_path)
        _write_or_print(render(input_paths, args, "latex"), latex_path)
        return

    output_path = Path(args.output).expanduser() if args.output else None
    _write_or_print(render(input_paths, args, args.format), output_path)


if __name__ == "__main__":
    main()
