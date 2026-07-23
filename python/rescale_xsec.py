"""rescale_xsec.py -- correct the cross-section normalization of an already-produced
MC .coffea output without re-running the processor.

The processor applies one global factor `lumi * xsec_pb / sumw` to every histogram
(`hist * scale_factor`), so revising only the cross section is an exact linear
rescale by `xsec_new / xsec_old`, bin-by-bin and across every systematic slice --
identical in spirit to python/scale_iov.py, but the ratio is a cross-section ratio
at fixed IOV rather than a luminosity ratio across IOVs.

Use case: the 2024 TTbar all-hadronic xsec was corrected from 350.6 pb
(XSDB NLO inclusive x BR) to 419.9 pb (924 pb theoretical NNLO x 0.4544 all-had BR).
Existing TTbar outputs made with the old value are stale by 419.9/350.6 = 1.198.

The old xsec is read from the file's own `normalization["xsec_pb"]` and asserted
against `--old-xsec` so re-running this is a no-op-detecting guard (it refuses if the
file's recorded xsec already differs from the stated old value -- prevents
double-applying). Data outputs are rejected (no xsec normalization).

CLI:
    python python/rescale_xsec.py outputs/dy/TTbar_2024_inclusive.coffea \
        --old-xsec 350.6 --new-xsec 419.9
"""

import os
import sys

import numpy as np
import hist
from coffea import processor
from coffea.util import load, save

_RTOL = 1e-3


def _relabel(n, ratio, new_xsec):
    n = dict(n)
    old = float(n.get("xsec_pb")) if n.get("xsec_pb") is not None else None
    n["xsec_pb"] = new_xsec
    if n.get("scale_factor") is not None:
        n["scale_factor"] = float(n["scale_factor"]) * ratio
    n["rescaled_xsec_from"] = {"xsec_pb": old, "ratio": ratio}
    return n


def _check(norm, old_xsec):
    if not isinstance(norm, dict):
        raise ValueError("output has no 'normalization' metadata; cannot rescale")
    if not norm.get("is_mc", False):
        raise ValueError("input is a DATA output -- data carries no xsec normalization")
    rec = norm.get("xsec_pb")
    if rec is None:
        raise ValueError("normalization has no 'xsec_pb'; cannot verify the old value")
    if abs(float(rec) - old_xsec) > _RTOL * old_xsec:
        raise ValueError(
            f"file records xsec_pb={float(rec)} but --old-xsec={old_xsec}. "
            f"Refusing to rescale (already corrected? wrong old value?).")
    return float(rec)


def rescale_xsec(input_path, old_xsec, new_xsec, output_path=None):
    """Load an MC .coffea and write a copy renormalized to `new_xsec`. In place by
    default (overwrites input_path). Returns (output_path, ratio)."""
    ratio = new_xsec / old_xsec

    out = load(input_path)
    norm = out.get("normalization")
    grouped = (isinstance(norm, dict) and norm
               and all(isinstance(v, dict) for v in norm.values()))

    if grouped:
        for v in norm.values():
            _check(v, old_xsec)
    else:
        _check(norm, old_xsec)

    # 1) every histogram: value * ratio, variance * ratio^2 (Weight storage)
    for key, value in list(out.items()):
        if isinstance(value, hist.Hist):
            out[key] = value * ratio

    # 2) ntuple weights
    if "ntuple" in out and isinstance(out["ntuple"], dict) and "weight" in out["ntuple"]:
        w = (out["ntuple"]["weight"].value * ratio).astype(np.float32)
        out["ntuple"]["weight"] = processor.column_accumulator(w)

    # 3) scaled cutflow bookkeeping (weighted^1 * r, weighted^2 * r^2)
    for ck, power in (("cutflow_scaled", 1),
                      ("cutflow_weighted_scaled", 1),
                      ("cutflow_weighted2_scaled", 2)):
        if ck in out and isinstance(out[ck], dict):
            out[ck] = {k: v * (ratio ** power) for k, v in out[ck].items()}

    # 4) update normalization xsec + scale_factor + provenance
    if grouped:
        out["normalization"] = {ds: _relabel(v, ratio, new_xsec) for ds, v in norm.items()}
    else:
        out["normalization"] = _relabel(norm, ratio, new_xsec)

    if output_path is None:
        output_path = input_path  # in place
    save(out, output_path)
    return output_path, ratio


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Rescale an MC .coffea to a corrected cross section without "
                    "re-running (MC only; the old xsec is verified against the file).")
    ap.add_argument("input", help="path to the source MC .coffea")
    ap.add_argument("--old-xsec", type=float, required=True,
                    help="cross section [pb] the file was produced with (verified)")
    ap.add_argument("--new-xsec", type=float, required=True,
                    help="corrected cross section [pb]")
    ap.add_argument("-o", "--output", default=None,
                    help="output path (default: overwrite input in place)")
    args = ap.parse_args()

    outp, r = rescale_xsec(args.input, args.old_xsec, args.new_xsec, args.output)
    print(f"[rescale_xsec] {args.input}")
    print(f"[rescale_xsec] -> {outp}")
    print(f"[rescale_xsec] xsec {args.old_xsec} -> {args.new_xsec}  (x {r:.6f})")
