"""scale_iov.py -- derive an MC output for a "stand-in" IOV from an already-produced
one by pure luminosity rescaling, without re-running the processor.

This is valid ONLY when the target IOV's MC processing is identical to the source's
up to the integrated-luminosity normalization: same files, same JEC/JER, pileup,
top-tag SF and tagger WPs. That holds for the current Summer24 stand-in -- 2025 MC IS
the 2024 Summer24 MC (see corrections._JSONPOG_JME_DIR, corrections.GetPUSF,
weights.py top-tag SF, ttbarprocessor._TAGGER_WPS), so

    output_2025 == output_2024 * (L_2025 / L_2024)

bin-by-bin and across every systematic-axis slice (the processor applies one global
lumi*xsec/sumw factor: `hist * scale_factor`). If 2025 ever gets its own MC or its
own pileup/JEC/SF, this shortcut is no longer valid -- re-run instead.

Data is never rescaled: real per-year data must be processed for real.

CLI:
    python python/scale_iov.py outputs/dy/TTbar_2024_..._topvsqcd.coffea --to-iov 2025
"""

import os
import sys

import numpy as np
import hist
from coffea import processor
from coffea.util import load, save

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttbarprocessor import _LUMI_PB

# target IOV -> source IOV whose MC processing is byte-identical to it (lumi aside).
# Keep this in lockstep with the corrections/weights/WP wiring: an entry here asserts
# "running the source-IOV MC and the target-IOV MC gives the same histograms up to
# the lumi factor". Only add an IOV once that is actually true.
_MC_LUMI_EQUIVALENT = {"2025": "2024"}


def _scale_mapping(m, factor, power=1):
    return {k: (v * (factor ** power)) for k, v in m.items()}


def _check_source(norm, dst_iov, src_iov):
    if not isinstance(norm, dict):
        raise ValueError("output has no 'normalization' metadata; cannot rescale safely")
    if not norm.get("is_mc", False):
        raise ValueError(
            "input is a DATA output -- data must be processed for real per year, "
            "not derived by lumi rescaling")
    yr = str(norm.get("year"))
    if yr != src_iov:
        raise ValueError(
            f"input normalization year={yr!r}, but a {dst_iov} rescale expects a "
            f"{src_iov} MC output (the lumi-equivalent source)")
    return float(norm["lumi_pb"])


def scale_mc_output(input_path, dst_iov, output_path=None):
    """Load an MC .coffea produced for the source IOV and write a copy normalized to
    `dst_iov` by the luminosity ratio. Returns (output_path, ratio)."""
    if dst_iov not in _MC_LUMI_EQUIVALENT:
        raise ValueError(
            f"No lumi-equivalent base IOV registered for {dst_iov!r} "
            f"(known: {sorted(_MC_LUMI_EQUIVALENT)}). If {dst_iov} has its own MC or "
            f"corrections, re-run the processor instead of rescaling.")
    src_iov = _MC_LUMI_EQUIVALENT[dst_iov]

    out = load(input_path)
    norm = out.get("normalization")
    grouped = (isinstance(norm, dict) and norm
               and all(isinstance(v, dict) for v in norm.values()))

    # validate, and read the lumi actually used (more robust than re-looking it up)
    if grouped:
        src_lumis = {_check_source(v, dst_iov, src_iov) for v in norm.values()}
        src_lumi = next(iter(src_lumis))
    else:
        src_lumi = _check_source(norm, dst_iov, src_iov)

    ratio = _LUMI_PB[dst_iov] / src_lumi

    # 1) every histogram: hist * ratio scales value by r and variance by r^2 (Weight
    #    storage), matching the processor's own `hist * scale_factor`.
    for key, value in list(out.items()):
        if isinstance(value, hist.Hist):
            out[key] = value * ratio

    # 2) ntuple weights (if present)
    if "ntuple" in out and isinstance(out["ntuple"], dict) and "weight" in out["ntuple"]:
        w = (out["ntuple"]["weight"].value * ratio).astype(np.float32)
        out["ntuple"]["weight"] = processor.column_accumulator(w)

    # 3) scaled cutflow bookkeeping (mirror processor: weighted^1 * r, weighted^2 * r^2)
    for ck, power in (("cutflow_scaled", 1),
                      ("cutflow_weighted_scaled", 1),
                      ("cutflow_weighted2_scaled", 2)):
        if ck in out and isinstance(out[ck], dict):
            out[ck] = _scale_mapping(out[ck], ratio, power=power)

    # 4) relabel normalization + sample_metadata to the target IOV
    def _relabel(n):
        n = dict(n)
        n["year"] = dst_iov
        n["lumi_pb"] = _LUMI_PB[dst_iov]
        if n.get("scale_factor") is not None:
            n["scale_factor"] = float(n["scale_factor"]) * ratio
        n["rescaled_from"] = {"iov": src_iov, "lumi_pb": src_lumi, "ratio": ratio}
        return n

    if grouped:
        out["normalization"] = {ds: _relabel(v) for ds, v in norm.items()}
    else:
        out["normalization"] = _relabel(norm)

    sm = out.get("sample_metadata")
    if isinstance(sm, dict) and sm:
        if all(isinstance(v, dict) for v in sm.values()):
            for v in sm.values():
                v["year"] = dst_iov
        else:
            sm["year"] = dst_iov

    if output_path is None:
        if f"_{src_iov}_" in input_path:
            output_path = input_path.replace(f"_{src_iov}_", f"_{dst_iov}_")
        else:
            output_path = input_path.replace(".coffea", f"_{dst_iov}.coffea")
    save(out, output_path)
    return output_path, ratio


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Rescale a Summer24-MC .coffea to a lumi-equivalent IOV (e.g. 2025) "
                    "without re-running. MC only; data must be run for real.")
    ap.add_argument("input", help="path to the source MC .coffea (e.g. a 2024 output)")
    ap.add_argument("--to-iov", required=True, choices=sorted(_MC_LUMI_EQUIVALENT),
                    help="target IOV to normalize to")
    ap.add_argument("-o", "--output", default=None,
                    help="output path (default: source path with the IOV swapped in)")
    args = ap.parse_args()

    outp, r = scale_mc_output(args.input, args.to_iov, args.output)
    print(f"[scale_iov] {args.input}")
    print(f"[scale_iov] -> {outp}")
    print(f"[scale_iov] {args.to_iov} = source x {r:.6f}  (L_{args.to_iov}={_LUMI_PB[args.to_iov]})")
