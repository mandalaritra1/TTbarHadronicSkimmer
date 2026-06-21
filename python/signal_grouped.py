"""Helpers for reading signal outputs grouped one-file-per-width.

Signal runs now write a single ``.coffea`` per width (e.g. ``ZPrime1_2024.coffea``)
containing every mass point on a ``dataset`` histogram axis, with ``normalization``
and ``sample_metadata`` stored as dicts keyed by the per-mass dataset label
(e.g. ``ZPrime4000_1``, ``RSGluon3000``).

``select_mass`` projects one mass back out into the *legacy* single-sample layout
(histograms without the dataset axis, flat ``normalization``/``sample_metadata``),
so existing per-mass plotting code keeps working with a one-line change at the
load site instead of a rewrite.
"""
from __future__ import annotations

import glob as _glob
import os
import re

import hist
from coffea import util


def dataset_labels(output):
    """All per-mass dataset labels present in a grouped output (sorted)."""
    for value in output.values():
        if isinstance(value, hist.Hist) and "dataset" in value.axes.name:
            return sorted(value.axes["dataset"])
    # not a grouped file (no dataset axis) -> nothing to enumerate
    return []


def is_grouped(output):
    """True if this output carries a dataset axis (a grouped width-file)."""
    return bool(dataset_labels(output))


def select_mass(output, mass_label):
    """Return a per-mass view of a grouped output in the legacy single-sample layout.

    Histograms are sliced to ``mass_label`` on the dataset axis (which drops the
    axis); ``normalization``/``sample_metadata`` dicts are reduced to that mass.
    Non-dataset entries (cutflow tables, analysisCategories, ...) pass through.
    """
    available = dataset_labels(output)
    if available and mass_label not in available:
        raise KeyError(f"{mass_label!r} not in grouped output; have {available}")

    result = {}
    for key, value in output.items():
        if isinstance(value, hist.Hist) and "dataset" in value.axes.name:
            result[key] = value[{"dataset": mass_label}]
        elif key in ("normalization", "sample_metadata") and isinstance(value, dict) and mass_label in value:
            result[key] = value[mass_label]
        else:
            result[key] = value
    return result


def load_mass(path, mass_label):
    """Load a grouped width-file from ``path`` and return the per-mass view."""
    return select_mass(util.load(path), mass_label)


def find_signal_mass(coffea_dir, mass, width="1", year="2024"):
    """Per-mass output for one ZPrime point, from either a legacy per-mass file
    (``ZPrime<mass>_<width>_<year>*.coffea``) or a grouped width-file
    (``ZPrime<width>_<year>*.coffea``) selected on the dataset axis. Raises
    FileNotFoundError if neither is present."""
    coffea_dir = str(coffea_dir)
    per_mass = sorted(_glob.glob(os.path.join(coffea_dir, f"ZPrime{mass}_{width}_{year}*.coffea")))
    if per_mass:
        return util.load(per_mass[0])
    ds_label = f"ZPrime{mass}_{width}"
    for path in sorted(_glob.glob(os.path.join(coffea_dir, f"ZPrime{width}_{year}*.coffea"))):
        out = util.load(path)
        if is_grouped(out) and ds_label in dataset_labels(out):
            return select_mass(out, ds_label)
    raise FileNotFoundError(
        f"No coffea for ZPrime mass {mass} width {width} year {year} in {coffea_dir}"
    )


def discover_signal_masses(coffea_dir, width="1", year="2024"):
    """Sorted int ZPrime masses available for a width, scanning both legacy
    per-mass files and a grouped width-file's dataset axis."""
    coffea_dir = str(coffea_dir)
    masses = set()
    for path in _glob.glob(os.path.join(coffea_dir, f"ZPrime*_{width}_{year}*.coffea")):
        m = re.match(rf"ZPrime(\d+)_{width}_{year}", os.path.basename(path))
        if m:
            masses.add(int(m.group(1)))
    for path in _glob.glob(os.path.join(coffea_dir, f"ZPrime{width}_{year}*.coffea")):
        for lbl in dataset_labels(util.load(path)):
            mm = re.match(rf"ZPrime(\d+)_{width}$", lbl)
            if mm:
                masses.add(int(mm.group(1)))
    return sorted(masses)


def stack_masses(mapping):
    """Build a grouped output from ``{mass_label: per_mass_output}``.

    Each histogram gains a leading ``dataset`` axis whose categories are the
    mass labels; ``normalization``/``sample_metadata`` become dicts keyed by mass.
    Inverse of :func:`select_mass`. Useful as a migration tool to fold existing
    per-mass ``.coffea`` files into one grouped width-file.
    """
    masses = list(mapping)
    if not masses:
        raise ValueError("stack_masses: empty mapping")
    first = mapping[masses[0]]
    grouped = {}
    for key, val in first.items():
        if isinstance(val, hist.Hist):
            ds_axis = hist.axis.StrCategory(masses, name="dataset", label="dataset")
            g = hist.Hist(ds_axis, *val.axes, storage="weight")
            for i, m in enumerate(masses):
                g.view(flow=True)[i] = mapping[m][key].view(flow=True)
            grouped[key] = g
        elif key in ("normalization", "sample_metadata"):
            grouped[key] = {m: mapping[m].get(key) for m in masses}
        else:
            grouped[key] = val  # passthrough (cutflow, analysisCategories, ...)
    return grouped
