#!/usr/bin/env python
"""Convert the ntuple section of one or more coffea output files to a ROOT TTree.

Usage (single file):
    python scripts/write_ntuple.py <input.coffea> <output.root> [tree_name]

Usage (merge multiple eras into one file):
    python scripts/write_ntuple.py <input1.coffea> <input2.coffea> ... --out <output.root> [--tree <tree_name>]

Usage (merge chunk ROOT files, optionally scaling the weight branch):
    python scripts/write_ntuple.py --merge-root <chunk1.root> <chunk2.root> ... --out <output.root> [--weight-scale <scale>]

No ROOT installation required — uproot writes the file in pure Python.
"""
import argparse
import os
import sys

# allow running from anywhere; locate repo root via `python/` sibling
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from python.ntuple_utils import write_ntuple, merge_root_ntuples  # noqa: E402


if __name__ == "__main__":
    if "--merge-root" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--merge-root", nargs="+", metavar="chunk.root", help="Chunk ROOT files to merge")
        parser.add_argument("--out", required=True, metavar="output.root", help="Output ROOT file")
        parser.add_argument("--tree", default="ttbar", metavar="tree_name", help="TTree name (default: ttbar)")
        parser.add_argument("--weight-scale", type=float, default=1.0, help="Scale factor applied to the weight branch")
        args = parser.parse_args()
        merge_root_ntuples(args.merge_root, args.out, args.tree, args.weight_scale)
    elif "--out" not in sys.argv:
        # Legacy single-file usage: write_ntuple.py input.coffea output.root [tree_name]
        if len(sys.argv) < 3:
            print(__doc__)
            sys.exit(1)
        tree_name = sys.argv[3] if len(sys.argv) > 3 else "ttbar"
        write_ntuple([sys.argv[1]], sys.argv[2], tree_name)
    else:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("inputs", nargs="+", metavar="input.coffea", help="One or more coffea files to merge")
        parser.add_argument("--out", required=True, metavar="output.root", help="Output ROOT file")
        parser.add_argument("--tree", default="ttbar", metavar="tree_name", help="TTree name (default: ttbar)")
        args = parser.parse_args()
        write_ntuple(args.inputs, args.out, args.tree)
