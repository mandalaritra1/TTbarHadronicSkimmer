import json
from pathlib import Path

import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights

from corrections import GetPDFWeights, GetPSWeights, GetPUSF, GetQ2weights, pTReweighting

# Versioned top-tag SF table (data/toptag/ttag_sf_<version>.json). v1.1 is the table that
# was hard-coded here up to v1.1; v1.2 has the 2024 SFs re-measured on the v1.2 objects.
TTAG_SF_FILE = Path(__file__).resolve().parent.parent / "data" / "toptag" / "ttag_sf_v1.2.json"


def load_ttag_sf(path=TTAG_SF_FILE):
    with open(path) as f:
        table = json.load(f)
    for alias, target in table["aliases"].items():
        table["iovs"][alias] = table["iovs"][target]
    return table


class Run3WeightManager:
    """Handle event-weight construction and systematic variations."""

    def __init__(self, iov, systematics, no_syst, deepak8_cut):
        self.iov = iov
        self.systematics = systematics
        self.no_syst = no_syst
        self.deepak8_cut = deepak8_cut
        self.ttag_sf = load_ttag_sf()

    def build_weights(self, dataset, events, evtweights, is_data, jet0, jet1, ttag2, antitag):
        weights = Weights(len(evtweights))
        weights.add("genWeight", evtweights)

        if "TTbar" in dataset:
            ttbar_wgt = pTReweighting(jet0.pt, jet1.pt)
            weights.add("ptReweighting", ttbar_wgt)

        if self.no_syst or is_data:
            return weights

        if "pileup" in self.systematics:
            puNom, puUp, puDown = GetPUSF(events, self.iov)
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

        if "ttag_pt1" in self.systematics:
            self._add_ttag_pt_weights(weights, jet0, jet1, ttag2, antitag)

        return weights

    def _add_ttag_pt_weights(self, weights, jet0, jet1, ttag2, antitag):
        table = self.ttag_sf["iovs"][self.iov]
        if self.deepak8_cut not in table["wps"]:
            raise KeyError(f"top-tag SF table {self.ttag_sf['version']} has no '{self.deepak8_cut}' "
                           f"WP for {self.iov} (have {sorted(table['wps'])})")
        wp = table["wps"][self.deepak8_cut]
        ptbins = self.ttag_sf["pt_edges"]

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

        def jet_sf(jet_ptbins, k, ibin, sf_nom, sf_var, sel=True):
            # per-jet SF in pT bin ibin: nominal + k * (variation - nominal)
            return np.where((jet_ptbins == ibin) & sel, sf_nom[ibin] + k * (sf_var[ibin] - sf_nom[ibin]), 1.0)

        for ibin in range(1, 4):
            nom, up, down = (
                jet_sf(jet0_ptbins, jet0_k, ibin, nomsf, var)
                * jet_sf(jet1_ptbins, jet1_k, ibin, nomsf, var, ttag2)
                * jet_sf(jet1_ptbins, jet1_k, ibin, nomsf_fail, var_fail, antitag)
                for var, var_fail in ((nomsf, nomsf_fail), (upsf, upsf_fail), (downsf, downsf_fail))
            )
            weights.add(f"ttag_pt{ibin}", weight=nom, weightUp=up, weightDown=down)