#!/usr/bin/env python3
"""Merge single-mass signal outputs into one batch output (the layout ttbaranalysis.py writes).

Single-mass runs (`-m 4500`) write `<W>_<IOV>_4500_<tagger>.coffea`; a batch run writes
`<W>_<IOV>_4500to7000_<tagger>.coffea` with the masses on the dataset axis. The datasets of the
inputs are disjoint, so histograms and counters add, per-dataset dicts are joined, and the
static entries (cutflow step definitions, categories, provenance) must agree and are kept once.

    python scripts/merge_signal_masses.py -o out/ZPrime30_2023preBPix_4500to7000_topvsqcd.coffea \
        in/ZPrime30_2023preBPix_{4500,5000,6000,7000}_topvsqcd.coffea
"""
import argparse
import sys

from coffea.processor import defaultdict_accumulator
from coffea.util import load, save

PER_DATASET = ("theory_norm", "sample_metadata", "normalization")
STATIC = ("cutflow_table_steps", "analysisCategories")
SUMMED_DICTS = ("cutflow_scaled", "cutflow_weighted_scaled", "cutflow_weighted2_scaled")


def merge(outputs):
    first = outputs[0]
    shas = {o["provenance"].get("git_sha") for o in outputs}
    if len(shas) != 1:
        raise SystemExit(f"inputs come from different commits: {shas}")
    datasets = [set(o["sample_metadata"]) for o in outputs]
    if sum(len(d) for d in datasets) != len(set().union(*datasets)):
        raise SystemExit("inputs share datasets: refusing to double count")
    merged = {}
    for key, val in first.items():
        rest = [o[key] for o in outputs[1:]]
        if key in PER_DATASET:
            merged[key] = {k: v for d in [val] + rest for k, v in d.items()}
        elif key in STATIC:
            if any(r != val for r in rest):
                raise SystemExit(f"'{key}' differs between inputs")
            merged[key] = val
        elif key == "provenance":
            merged[key] = dict(val, merged_from=len(outputs))
        elif key in SUMMED_DICTS:
            merged[key] = {k: sum(d.get(k, 0) for d in [val] + rest) for k in val}
        elif hasattr(val, "axes") or isinstance(val, defaultdict_accumulator):
            acc = val.copy() if hasattr(val, "copy") else val
            for r in rest:
                acc = acc + r
            merged[key] = acc
        else:
            raise SystemExit(f"don't know how to merge '{key}' ({type(val).__name__})")
    missing = set().union(*(set(o) for o in outputs)) - set(merged)
    if missing:
        raise SystemExit(f"keys missing from the first input: {sorted(missing)}")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    merged = merge([load(f) for f in args.inputs])
    save(merged, args.output)
    print(f"wrote {args.output}: datasets {sorted(merged['sample_metadata'])}")


if __name__ == "__main__":
    sys.exit(main())
