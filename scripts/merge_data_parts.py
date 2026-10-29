#!/usr/bin/env python3
"""Sum data coffea parts into one file, loading one part at a time (memory: two outputs).

Data outputs are not normalised, so the parts simply add: every hist.Hist is summed, every numeric
entry of a dict of numbers (cutflows) is summed; other entries (provenance, normalization,
metadata) are taken from the first part, with the number of merged parts and their commits added.

    python merge_data_parts.py -o data_2024_C_topvsqcd.coffea parts/C_p*/data_2024_C_topvsqcd.coffea
"""
import argparse
import numbers

import hist
from coffea import util


def add(acc, part):
    for k, v in part.items():
        if k not in acc:
            acc[k] = v
        elif isinstance(v, hist.Hist):
            acc[k] = acc[k] + v
        elif isinstance(v, dict) and v and all(isinstance(x, numbers.Number) for x in v.values()):
            acc[k] = {kk: acc[k].get(kk, 0) + vv for kk, vv in v.items()} | {kk: vv for kk, vv in acc[k].items() if kk not in v}
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    acc, shas = None, set()
    for f in sorted(a.inputs):
        part = util.load(f)
        shas.add(str(part.get("provenance", {}).get("git_sha")))
        acc = part if acc is None else add(acc, part)
        del part
    if isinstance(acc.get("provenance"), dict):
        acc["provenance"] = dict(acc["provenance"], merged_from=len(a.inputs), merged_commits=sorted(shas))
    util.save(acc, a.output)
    print(f"wrote {a.output}: {len(a.inputs)} parts, commits {sorted(shas)}")


if __name__ == "__main__":
    main()
