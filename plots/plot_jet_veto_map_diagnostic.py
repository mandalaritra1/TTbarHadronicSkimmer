#!/usr/bin/env python3
"""Plot the official Run-3 jet-veto map and the jets that trigger it.

This is a validation utility, not an analysis selection.  It reproduces the
PFHT1050 + HT + MET-filter stage used immediately before the jet-veto map in
the processor and overlays eligible AK4 PUPPI jets that land in vetoed bins.
"""

from __future__ import annotations

import argparse
from datetime import date
import subprocess
import sys
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from corrections import _load_jet_veto_correction, _tight_lepton_veto_mask  # noqa: E402


MET_FILTERS_2024 = (
    "goodVertices",
    "globalSuperTightHalo2016Filter",
    "EcalDeadCellTriggerPrimitiveFilter",
    "BadPFMuonFilter",
    "BadPFMuonDzFilter",
    "hfNoisyHitsFilter",
    "eeBadScFilter",
    "ecalBadCalibFilter",
)

JET_FIELDS = (
    "pt", "eta", "phi", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF",
    "chMultiplicity", "neMultiplicity",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="NanoAOD ROOT files")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG")
    parser.add_argument("--label", default="2024 simulation", help="Short CMS right label")
    parser.add_argument("--ht-cut", type=float, default=1400.0, help="AK4 HT threshold [GeV]")
    return parser.parse_args()


def _branches() -> list[str]:
    branches = [f"Jet_{field}" for field in JET_FIELDS]
    branches += ["HLT_PFHT1050"]
    branches += [f"Flag_{name}" for name in MET_FILTERS_2024]
    return branches


def _selection_before_veto(arrays, jets, ht_cut: float) -> np.ndarray:
    ht = ak.sum(jets.pt[(jets.pt > 30.0) & (abs(jets.eta) < 3.0)], axis=1)
    met_filter = np.logical_and.reduce([
        ak.to_numpy(arrays[f"Flag_{name}"]) for name in MET_FILTERS_2024
    ])
    return (
        ak.to_numpy(arrays["HLT_PFHT1050"])
        & (ak.to_numpy(ht) > ht_cut)
        & met_filter
    )


def _vetoed_jets(jets, event_mask: np.ndarray):
    tight_lepton_veto = _tight_lepton_veto_mask(jets, "2024")
    eligible = (
        (jets.pt > 15.0)
        & tight_lepton_veto
        & ((jets.chEmEF + jets.neEmEF) < 0.9)
    )
    correction = _load_jet_veto_correction("2024")
    counts = ak.to_numpy(ak.num(jets, axis=1))
    map_values = ak.unflatten(
        correction.evaluate(
            "jetvetomap",
            ak.to_numpy(ak.flatten(jets.eta, axis=1)),
            ak.to_numpy(ak.flatten(jets.phi, axis=1)),
        ),
        counts,
    )
    selected_vetoed = jets[event_mask][(eligible & (map_values != 0))[event_mask]]
    event_is_vetoed = ak.to_numpy(ak.any(eligible & (map_values != 0), axis=1))
    return selected_vetoed, event_is_vetoed


def _diagnose(paths: list[Path], ht_cut: float, eta_edges, phi_edges):
    n_before = 0
    n_vetoed = 0
    hit_counts = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1))
    tree_paths = [f"{path}:Events" for path in paths]
    for arrays in uproot.iterate(
        tree_paths,
        expressions=_branches(),
        step_size="50 MB",
        library="ak",
    ):
        jets = ak.zip({field: arrays[f"Jet_{field}"] for field in JET_FIELDS})
        before_veto = _selection_before_veto(arrays, jets, ht_cut)
        vetoed_jets, event_is_vetoed = _vetoed_jets(jets, before_veto)
        hit_eta = ak.to_numpy(ak.flatten(vetoed_jets.eta, axis=1))
        hit_phi = ak.to_numpy(ak.flatten(vetoed_jets.phi, axis=1))
        chunk_counts, _, _ = np.histogram2d(
            hit_eta, hit_phi, bins=(eta_edges, phi_edges)
        )
        hit_counts += chunk_counts
        n_before += int(np.count_nonzero(before_veto))
        n_vetoed += int(np.count_nonzero(before_veto & event_is_vetoed))
    return n_before, n_vetoed, hit_counts


def main() -> None:
    args = _parse_args()
    eta_edges = np.linspace(-5.2, 5.2, 261)
    phi_edges = np.linspace(-np.pi, np.pi, 181)
    n_before, n_vetoed, hit_counts = _diagnose(
        args.inputs, args.ht_cut, eta_edges, phi_edges
    )
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    eta_grid, phi_grid = np.meshgrid(eta_centers, phi_centers, indexing="ij")
    correction = _load_jet_veto_correction("2024")
    map_grid = correction.evaluate(
        "jetvetomap", eta_grid.ravel(), phi_grid.ravel()
    ).reshape(eta_grid.shape)

    positive_hit_counts = np.ma.masked_where(hit_counts == 0, hit_counts)

    hep.style.use(hep.style.CMS)
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(21.2, 10.6))
    map_image = axes[0].pcolormesh(
        eta_edges, phi_edges, (map_grid != 0).T,
        cmap=colors.ListedColormap(["white", "#f89c20"]),
        vmin=0, vmax=1, shading="flat",
    )
    hit_image = axes[1].pcolormesh(
        eta_edges, phi_edges, positive_hit_counts.T,
        cmap="viridis", norm=colors.LogNorm(vmin=1), shading="flat",
    )

    for ax in axes:
        ax.set_xlabel(r"AK4 jet $\eta$")
        ax.set_ylabel(r"AK4 jet $\phi$")
        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-np.pi, np.pi)
        hep.cms.label(data=False, loc=2, ax=ax, rlabel=args.label)

    axes[0].text(
        0.03, 0.84, "Official final jet-veto map",
        transform=axes[0].transAxes, ha="left", va="top", fontsize=20,
    )
    axes[1].text(
        0.03, 0.84, "Eligible jets in vetoed bins",
        transform=axes[1].transAxes, ha="left", va="top", fontsize=20,
    )
    fraction = n_vetoed / n_before if n_before else float("nan")
    axes[1].text(
        0.03, 0.74,
        f"PFHT1050 + HT + MET filters: {n_before:,} events\n"
        f"Vetoed: {n_vetoed:,} ({fraction:.1%})",
        transform=axes[1].transAxes, ha="left", va="top", fontsize=17,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    fig.colorbar(map_image, ax=axes[0], ticks=[0, 1], label="Map decision")
    fig.colorbar(hit_image, ax=axes[1], label="Eligible jets / bin")
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.97))
    git_version = subprocess.run(
        ["git", "describe", "--tags", "--always", "--dirty"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "unknown"
    fig.text(
        0.99, 0.005,
        f"{date.today().isoformat()} | TTbarHadronicSkimmer {git_version} | "
        "inputs: local 2024 ZPrime4000 W10",
        ha="right", va="bottom", fontsize=8, color="0.45", family="monospace",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"wrote {args.output}")
    print(f"before veto: {n_before}; vetoed: {n_vetoed}; fraction: {fraction:.6f}")


if __name__ == "__main__":
    main()
