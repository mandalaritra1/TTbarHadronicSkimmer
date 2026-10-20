import json
from pathlib import Path

import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights

from corrections import GetPDFWeights, GetPSWeights, GetPUSF, GetQ2weights, GetTopPtWeight
from truthstudy import MERGE_FULL, top_merge_category

# Versioned top-tag SF table (data/toptag/ttag_sf_<version>.json); v1.2 holds the 2024
# SFs measured on the v1.2 objects.
TTAG_SF_FILE = Path(__file__).resolve().parent.parent / "data" / "toptag" / "ttag_sf_v1.2.json"

# SF_flep nuisances per block of the table's 'flep' entry
FLEP_NUISANCES = {"tag": "ttag_flep_tag", "antitag": "ttag_flep_band"}

# year-specific top-tag drift nuisances (table entry 'drift'): one per tight pT bin and one for
# the band; the table maps each pT bin of a block to one of these names
DRIFT_NUISANCES = ("ttag_drift_pt1", "ttag_drift_pt2", "ttag_drift_pt3", "ttag_drift_band")


def load_ttag_sf(path=TTAG_SF_FILE):
    with open(path) as f:
        table = json.load(f)
    for alias, target in table["aliases"].items():
        table["iovs"][alias] = table["iovs"][target]
    return table


def _mc_campaign(events):
    """'Summer24' when the chunk comes from a Summer24 file (e.g. the 2024 Z' standing in for
    2022/2023), else None. Read from the NanoEvents metadata (input file name)."""
    try:
        fname = str(events.metadata.get("filename", ""))
    except Exception:
        return None
    return "Summer24" if "Summer24" in fname else None


class Run3WeightManager:
    """Handle event-weight construction and systematic variations."""

    def __init__(self, iov, systematics, no_syst, deepak8_cut):
        self.iov = iov
        self.systematics = systematics
        self.no_syst = no_syst
        self.deepak8_cut = deepak8_cut
        self.ttag_sf = load_ttag_sf()

    def build_weights(self, dataset, events, evtweights, is_data, jet0, jet1, ttag2, antitag, flep_sf=True):
        """flep_sf=False leaves out SF_flep (ttag_flep_*), which belongs to jets that pass the
        f_lep cut: use it for histograms filled before that cut."""
        weights = Weights(len(evtweights))
        weights.add("genWeight", evtweights)

        if "TTbar" in dataset:
            # top-pT reweighting on the gen last-copy tops (Run-3 TOP practice, TOP-23-008);
            # uncertainty = with vs without, symmetrised (down = 2w - 1)
            ttbar_wgt = GetTopPtWeight(events.GenPart)
            if self.no_syst or "toppt" not in self.systematics:
                weights.add("toppt", ttbar_wgt)
            else:
                weights.add("toppt", weight=ttbar_wgt, weightUp=np.ones_like(ttbar_wgt),
                            weightDown=2.0 * ttbar_wgt - 1.0)

        if self.no_syst or is_data:
            return weights

        if "pileup" in self.systematics:
            puNom, puUp, puDown = GetPUSF(events, self.iov, mc_campaign=_mc_campaign(events))
            weights.add("pileup", weight=puNom, weightUp=puUp, weightDown=puDown)

       
        if "pdf" in self.systematics:
            pdfNom, pdfUp, pdfDown = GetPDFWeights(events)
            weights.add("pdf", weight=pdfNom, weightUp=pdfUp, weightDown=pdfDown)

        if "q2" in self.systematics:
            q2Nom, q2Up, q2Down = GetQ2weights(events)
            weights.add("q2", weight=q2Nom, weightUp=q2Up, weightDown=q2Down)

        for ps in ("isr", "fsr"):
            if ps in self.systematics:
                psNom, psUp, psDown = GetPSWeights(events, ps)
                weights.add(ps, weight=psNom, weightUp=psUp, weightDown=psDown)

        flep_on = flep_sf and any(s in self.systematics for s in FLEP_NUISANCES.values())
        if flep_on or any(s in self.systematics for s in ("ttag_pt1", "ttag_nonmerged") + DRIFT_NUISANCES):
            merged = None
            if self.ttag_sf["iovs"][self.iov].get("applies_to", "all") == "fully_merged":
                merged = tuple(
                    top_merge_category(events.GenPart, ak.to_numpy(j.p4.eta), ak.to_numpy(j.p4.phi)) == MERGE_FULL
                    for j in (jet0, jet1)
                )
            if "ttag_pt1" in self.systematics:
                self._add_ttag_pt_weights(weights, jet0, jet1, ttag2, antitag, merged)
            if "ttag_nonmerged" in self.systematics:
                self._add_ttag_nonmerged_weights(weights, jet0, jet1, ttag2, antitag, merged)
            if flep_on:
                self._add_ttag_flep_weights(weights, jet0, jet1, ttag2, antitag, merged)
            if any(s in self.systematics for s in DRIFT_NUISANCES):
                self._add_ttag_drift_weights(weights, jet0, jet1, ttag2, antitag, merged)

        return weights

    def _add_ttag_flep_weights(self, weights, jet0, jet1, ttag2, antitag, merged=None):
        """SF of the leptonic-top cut f_lep < 0.59 (factorised: SF_total = SF_TopvsQCD x SF_flep).

        The T&P measures SF_flep = data/MC of P(f_lep < cut | score passes) for fully merged
        hadronic tops, separately for tagged and band jets. jet0 and a tagged jet1 take the
        'tag' SF, an antitag jet1 the 'antitag' SF; pt_edges are the lower bin edges (last bin
        open, jets below the first edge keep 1), per block if the block has its own pt_edges
        (e.g. an inclusive band value). Two nuisances, one per block (independent probes),
        each correlated across pT bins and between the jets. Above a block's extrapolate_above
        the deviations are scaled by extrapolation_factor; up never exceeds the physical bound
        1/eps_mc. Tables without a 'flep' block: weight 1."""
        table = self.ttag_sf["iovs"][self.iov]
        jet0_pt = ak.to_numpy(jet0.p4.pt)
        jet1_pt = ak.to_numpy(jet1.p4.pt)
        flep = table.get("flep")
        if flep is None:
            ones = np.ones(len(jet0_pt))
            for name in FLEP_NUISANCES.values():
                weights.add(name, weight=ones, weightUp=ones, weightDown=ones)
            return
        if table.get("applies_to", "all") == "fully_merged":
            if merged is None:
                raise ValueError(f"top-tag SFs for {self.iov} apply to fully merged tops only: "
                                 "pass the per-jet gen merge flags")
            merged0, merged1 = (np.asarray(m, dtype=bool) for m in merged)
        else:
            merged0 = merged1 = np.ones(len(jet0_pt), bool)
        ttag2 = np.asarray(ak.to_numpy(ttag2), dtype=bool)
        antitag = np.asarray(ak.to_numpy(antitag), dtype=bool)
        factor = flep.get("extrapolation_factor", 1.0)

        def jet_sf(pt, block, sel):
            b = flep[block]
            edges = b.get("pt_edges", flep.get("pt_edges"))
            ib = np.digitize(pt, edges) - 1
            ibc = np.clip(ib, 0, len(edges) - 1)
            nom, up, down = (np.asarray(b[v], dtype=float)[ibc] for v in ("nominal", "up", "down"))
            k = np.where(pt > b.get("extrapolate_above", np.inf), factor, 1.0)
            up, down = nom + k * (up - nom), nom + k * (down - nom)
            if "eps_mc" in b:
                up = np.minimum(up, 1.0 / np.asarray(b["eps_mc"], dtype=float)[ibc])
            return [np.where((ib >= 0) & sel, v, 1.0) for v in (nom, up, down)]

        sf0 = jet_sf(jet0_pt, "tag", merged0)
        sf1t = jet_sf(jet1_pt, "tag", ttag2 & merged1)
        sf1a = jet_sf(jet1_pt, "antitag", antitag & merged1)
        nom, up, down = (a * b for a, b in zip(sf0, sf1t))
        weights.add(FLEP_NUISANCES["tag"], weight=nom, weightUp=up, weightDown=down)
        weights.add(FLEP_NUISANCES["antitag"], weight=sf1a[0], weightUp=sf1a[1], weightDown=sf1a[2])

    def _add_ttag_drift_weights(self, weights, jet0, jet1, ttag2, antitag, merged=None):
        """Year-specific drift of the real-top tag efficiency against the year the SFs were
        measured in (table entry 'drift', keyed by IOV). 2026 aliases the 2024 SFs, but the T&P
        finds the 2026 data efficiency lower with the same Summer24 MC, so the aliased SFs of
        fully merged jets are multiplied by the drift factor: jet0 and a tagged jet1 by 'tag', an
        antitag jet1 by 'antitag' (factor 1 if the block is absent), per pT bin (pt_edges are
        lower edges, last bin open, jets below the first edge keep 1).

        Each block lists one nuisance name per pT bin (from DRIFT_NUISANCES): the tight ratio has
        one nuisance per bin (bin-to-bin correlations are small), the band one for all bins. A
        nuisance is correlated between the jets and between signal and tt-bar. Deviations are
        scaled by extrapolation_factor above extrapolate_above unless the block sets
        "extrapolate": false. IOVs without a drift entry: weight 1 for every drift nuisance."""
        drift = self.ttag_sf.get("drift", {}).get(self.iov)
        jet0_pt = ak.to_numpy(jet0.p4.pt)
        jet1_pt = ak.to_numpy(jet1.p4.pt)
        ones = np.ones(len(jet0_pt))
        if drift is None:
            for name in DRIFT_NUISANCES:
                weights.add(name, weight=ones, weightUp=ones, weightDown=ones)
            return
        if merged is None:
            raise ValueError(f"the {self.iov} top-tag drift applies to fully merged tops only: "
                             "pass the per-jet gen merge flags")
        merged0, merged1 = (np.asarray(m, dtype=bool) for m in merged)
        ttag2 = np.asarray(ak.to_numpy(ttag2), dtype=bool)
        antitag = np.asarray(ak.to_numpy(antitag), dtype=bool)
        edges = drift["pt_edges"]
        factor = drift.get("extrapolation_factor", 1.0)

        nominal = ones.copy()
        up = {name: ones.copy() for name in DRIFT_NUISANCES}
        down = {name: ones.copy() for name in DRIFT_NUISANCES}
        for pt, block, sel in ((jet0_pt, "tag", merged0), (jet1_pt, "tag", ttag2 & merged1),
                               (jet1_pt, "antitag", antitag & merged1)):
            if block not in drift:
                continue
            b = drift[block]
            unknown = set(b["nuisances"]) - set(DRIFT_NUISANCES)
            if unknown:
                raise KeyError(f"drift nuisances {sorted(unknown)} are not in DRIFT_NUISANCES")
            ib = np.digitize(pt, edges) - 1
            valid = (ib >= 0) & sel
            ibc = np.clip(ib, 0, len(edges) - 1)
            nom, vup, vdown = (np.asarray(b[v], dtype=float)[ibc] for v in ("nominal", "up", "down"))
            k = np.where((pt > drift.get("extrapolate_above", np.inf)) & b.get("extrapolate", True), factor, 1.0)
            vup, vdown = nom + k * (vup - nom), nom + k * (vdown - nom)
            nominal *= np.where(valid, nom, 1.0)
            for i, name in enumerate(b["nuisances"]):
                in_bin = valid & (ibc == i)
                up[name] *= np.where(in_bin, vup / nom, 1.0)
                down[name] *= np.where(in_bin, vdown / nom, 1.0)
        for j, name in enumerate(DRIFT_NUISANCES):
            w = nominal if j == 0 else ones   # the drift nominal enters the event weight once
            weights.add(name, weight=w, weightUp=w * up[name], weightDown=w * down[name])

    def _add_ttag_nonmerged_weights(self, weights, jet0, jet1, ttag2, antitag, merged=None):
        """One flat nuisance for the tagged/antitag jets that are not fully merged tops.

        The SFs were measured for fully merged tops only, so these jets keep SF 1; each of
        them gets +-nonmerged_unc (same size for all pT bins, tag and antitag alike).
        Tables applied to all jets (legacy Run 2) carry no such uncertainty: weight 1."""
        table = self.ttag_sf["iovs"][self.iov]
        n = len(ak.to_numpy(jet0.p4.pt))
        unc = table.get("nonmerged_unc") if table.get("applies_to", "all") == "fully_merged" else None
        if not unc:
            ones = np.ones(n)
            weights.add("ttag_nonmerged", weight=ones, weightUp=ones, weightDown=ones)
            return
        if merged is None:
            raise ValueError(f"top-tag SFs for {self.iov} apply to fully merged tops only: "
                             "pass the per-jet gen merge flags")
        merged0, merged1 = (np.asarray(m, dtype=bool) for m in merged)
        ptbins = self.ttag_sf["pt_edges"]
        # same jets as the ttag_pt SFs: jet0, and jet1 when tagged or antitag, in the SF pT range
        in0 = np.digitize(ak.to_numpy(jet0.p4.pt), ptbins) - 1 >= 1
        in1 = np.digitize(ak.to_numpy(jet1.p4.pt), ptbins) - 1 >= 1
        sel1 = np.asarray(ak.to_numpy(ttag2), dtype=bool) | np.asarray(ak.to_numpy(antitag), dtype=bool)
        n_nonmerged = (in0 & ~merged0).astype(int) + (in1 & sel1 & ~merged1).astype(int)
        weights.add("ttag_nonmerged", weight=np.ones(n),
                    weightUp=(1.0 + unc) ** n_nonmerged, weightDown=(1.0 - unc) ** n_nonmerged)

    def _add_ttag_pt_weights(self, weights, jet0, jet1, ttag2, antitag, merged=None):
        table = self.ttag_sf["iovs"][self.iov]
        if self.deepak8_cut not in table["wps"]:
            raise KeyError(f"top-tag SF table {self.ttag_sf['version']} has no '{self.deepak8_cut}' "
                           f"WP for {self.iov} (have {sorted(table['wps'])})")
        wp = table["wps"][self.deepak8_cut]
        ptbins = self.ttag_sf["pt_edges"]

        # The T&P measured the SFs for fully merged tops only (other merge categories fixed
        # to MC), so with applies_to = fully_merged every other jet keeps SF 1.
        if table.get("applies_to", "all") == "fully_merged":
            if merged is None:
                raise ValueError(f"top-tag SFs for {self.iov} apply to fully merged tops only: "
                                 "pass the per-jet gen merge flags")
            merged0, merged1 = (np.asarray(m, dtype=bool) for m in merged)
        else:
            merged0 = merged1 = True

        # Extrapolation: the T&P data run out at pt_measured_max, so jets beyond it get
        # the top-bin SF with its uncertainty doubled (same nuisance). null = no doubling
        # (Run-2 SFs keep their original treatment).
        pt_measured_max = table["pt_measured_max"] or np.inf

        nomsf, upsf, downsf = (np.array(wp["tag"][v]) for v in ("nominal", "up", "down"))
        # antitag jet: v1.2 2024 has the SF measured in its own score band [low WP, WP);
        # the legacy tables give it the 'loose' SF
        nomsf_fail, upsf_fail, downsf_fail = (np.array(wp["antitag"][v]) for v in ("nominal", "up", "down"))

        jet0_pt = ak.to_numpy(jet0.p4.pt)
        jet1_pt = ak.to_numpy(jet1.p4.pt)
        jet0_ptbins = np.digitize(jet0_pt, ptbins) - 1
        jet1_ptbins = np.digitize(jet1_pt, ptbins) - 1
        jet0_k = np.where(jet0_pt > pt_measured_max, 2.0, 1.0)
        jet1_k = np.where(jet1_pt > pt_measured_max, 2.0, 1.0)

        # Interim high-pT band SF (antitag_hipt): band jets at or above its first edge take
        # its SF and its own nuisance instead of the ttag_pt band term.
        hipt = table.get("antitag_hipt")
        antitag = np.asarray(ak.to_numpy(antitag), dtype=bool)
        jet1_hipt = np.zeros(len(jet1_pt), bool) if hipt is None else jet1_pt >= hipt["pt_edges"][0]

        def jet_sf(jet_ptbins, k, ibin, sf_nom, sf_var, sel=True):
            # per-jet SF in pT bin ibin: nominal + k * (variation - nominal)
            return np.where((jet_ptbins == ibin) & sel, sf_nom[ibin] + k * (sf_var[ibin] - sf_nom[ibin]), 1.0)

        for ibin in range(1, 4):
            nom, up, down = (
                jet_sf(jet0_ptbins, jet0_k, ibin, nomsf, var, merged0)
                * jet_sf(jet1_ptbins, jet1_k, ibin, nomsf, var, ttag2 & merged1)
                * jet_sf(jet1_ptbins, jet1_k, ibin, nomsf_fail, var_fail, antitag & ~jet1_hipt & merged1)
                for var, var_fail in ((nomsf, nomsf_fail), (upsf, upsf_fail), (downsf, downsf_fail))
            )
            weights.add(f"ttag_pt{ibin}", weight=nom, weightUp=up, weightDown=down)

        name = "ttag_band_hipt" if hipt is None else hipt["nuisance"]
        if hipt is None:
            ones = np.ones(len(jet1_pt))
            weights.add(name, weight=ones, weightUp=ones, weightDown=ones)
        else:
            ib = np.clip(np.digitize(jet1_pt, hipt["pt_edges"]) - 1, 0, len(hipt["nominal"]) - 1)
            sel = antitag & jet1_hipt & merged1
            nom, up, down = (np.where(sel, np.asarray(hipt[v], dtype=float)[ib], 1.0)
                             for v in ("nominal", "up", "down"))
            weights.add(name, weight=nom, weightUp=up, weightDown=down)