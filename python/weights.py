import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights

from corrections import GetPDFWeights, GetPUSF, GetQ2weights, pTReweighting


class Run3WeightManager:
    """Handle event-weight construction and systematic variations."""

    def __init__(self, iov, systematics, no_syst, deepak8_cut):
        self.iov = iov
        self.systematics = systematics
        self.no_syst = no_syst
        self.deepak8_cut = deepak8_cut

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

        if "ttag_pt1" in self.systematics:
            self._add_ttag_pt_weights(weights, jet0, jet1, ttag2, antitag)

        return weights

    def _add_ttag_pt_weights(self, weights, jet0, jet1, ttag2, antitag):
        ptbins = [0, 400, 480, 600]

        ttag_scale_factors = {
            "2016": {
                "tight": {"nominal": [0.92, 1.01, 0.84, 1.00], "up": [1.04, 1.18, 0.9, 1.07], "down": [0.82, 0.84, 0.78, 0.94]},
                "medium": {"nominal": [0.89, 1.02, 0.93, 1.00], "up": [0.97, 1.07, 0.97, 1.05], "down": [0.81, 0.97, 0.89, 0.95]},
                "loose": {"nominal": [0.98, 0.95, 0.97, 1.00], "up": [0.98, 0.99, 1.0, 1.04], "down": [0.91, 0.91, 0.94, 0.96]},
            },
            "2016APV": {
                "tight": {"nominal": [0.92, 1.01, 0.84, 1.00], "up": [1.04, 1.18, 0.9, 1.07], "down": [0.82, 0.84, 0.78, 0.94]},
                "medium": {"nominal": [0.89, 1.02, 0.93, 1.00], "up": [0.97, 1.07, 0.97, 1.05], "down": [0.81, 0.97, 0.89, 0.95]},
                "loose": {"nominal": [0.98, 0.95, 0.97, 1.00], "up": [0.98, 0.99, 1.0, 1.04], "down": [0.91, 0.91, 0.94, 0.96]},
            },
            "2017": {
                "tight": {"nominal": [0.88, 0.9, 0.95, 0.97], "up": [0.96, 0.95, 1.0, 1.03], "down": [0.8, 0.85, 0.9, 0.91]},
                "medium": {"nominal": [0.95, 1.0, 0.98, 0.98], "up": [1.01, 1.04, 1.02, 1.02], "down": [0.89, 0.96, 0.94, 0.94]},
                "loose": {"nominal": [0.95, 0.98, 0.97, 0.97], "up": [1.0, 1.01, 1.0, 1.01], "down": [0.9, 0.95, 0.94, 0.93]},
            },
            "2018": {
                "tight": {"nominal": [0.81, 0.93, 0.96, 0.93], "up": [0.88, 0.98, 1.02, 1.01], "down": [0.74, 0.88, 0.92, 0.85]},
                "medium": {"nominal": [0.90, 0.97, 0.98, 0.95], "up": [0.95, 1.0, 1.01, 0.98], "down": [0.85, 0.94, 0.95, 0.92]},
                "loose": {"nominal": [0.96, 1.00, 0.98, 0.99], "up": [1.0, 1.03, 1.0, 1.02], "down": [0.92, 0.97, 0.96, 0.96]},
            },
            ## 2024 top-tag SF: assumed FLAT 0.90 per tag (medium WP) for testing.
            ## NOT a measured value -- placeholder until the real 2024 GloParT-v3
            ## efficiency SF is derived. up/down = 0.90 +/- 0.10.
            "2024": {
                "tight": {"nominal": [0.90, 0.90, 0.90, 0.90], "up": [1.00, 1.00, 1.00, 1.00], "down": [0.80, 0.80, 0.80, 0.80]},
                "medium": {"nominal": [0.90, 0.90, 0.90, 0.90], "up": [1.00, 1.00, 1.00, 1.00], "down": [0.80, 0.80, 0.80, 0.80]},
                "loose": {"nominal": [0.96, 1.00, 0.98, 0.99], "up": [1.0, 1.03, 1.0, 1.02], "down": [0.92, 0.97, 0.96, 0.96]},
            }
        }

        # 2022/2023 v15 sub-eras: reuse the 2024 placeholder top-tag SF (same
        # GloParTv3 tagger) until per-year SFs are derived.
        for _subera in ("2022preEE", "2022postEE", "2023preBPix", "2023postBPix"):
            ttag_scale_factors[_subera] = ttag_scale_factors["2024"]

        nomsf = np.array(ttag_scale_factors[self.iov][self.deepak8_cut]["nominal"])
        upsf = np.array(ttag_scale_factors[self.iov][self.deepak8_cut]["up"])
        downsf = np.array(ttag_scale_factors[self.iov][self.deepak8_cut]["down"])

        nomsf_fail = np.array(ttag_scale_factors[self.iov]["loose"]["nominal"])
        upsf_fail = np.array(ttag_scale_factors[self.iov]["loose"]["up"])
        downsf_fail = np.array(ttag_scale_factors[self.iov]["loose"]["down"])

        jet0_ptbins = np.digitize(ak.to_numpy(jet0.p4.pt), ptbins) - 1
        jet1_ptbins = np.digitize(ak.to_numpy(jet1.p4.pt), ptbins) - 1

        for ibin in range(1, 4):
            nom = (
                np.where(jet0_ptbins == ibin, nomsf[ibin], 1.0)
                * np.where((jet1_ptbins == ibin) & ttag2, nomsf[ibin], 1.0)
                * np.where((jet1_ptbins == ibin) & antitag, nomsf_fail[ibin], 1.0)
            )
            up = (
                np.where(jet0_ptbins == ibin, upsf[ibin], 1.0)
                * np.where((jet1_ptbins == ibin) & ttag2, upsf[ibin], 1.0)
                * np.where((jet1_ptbins == ibin) & antitag, upsf_fail[ibin], 1.0)
            )
            down = (
                np.where(jet0_ptbins == ibin, downsf[ibin], 1.0)
                * np.where((jet1_ptbins == ibin) & ttag2, downsf[ibin], 1.0)
                * np.where((jet1_ptbins == ibin) & antitag, downsf_fail[ibin], 1.0)
            )
            weights.add(f"ttag_pt{ibin}", weight=nom, weightUp=up, weightDown=down)