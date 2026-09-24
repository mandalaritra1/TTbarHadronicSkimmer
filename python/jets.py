import awkward as ak
import numpy as np

from corrections import GetJECUncertainties, GetJetVetoMapMask
from functions import getRapidity

# AK4 jet selection thresholds (used in both HT sum and baseline masks)
_AK4_PT_MIN  = 30    # GeV
_AK4_ETA_MAX = 3.0

# FatJet (AK8) acceptance boundary — applied on rapidity, not pseudo-rapidity
_AK8_RAPIDITY_MAX = 2.4


class Run3JetManager:
    """Utility class to keep jet/fatjet preparation, corrections, and baseline selections together."""

    def __init__(self, iov, systematics, no_syst, ak8_pt_min, ht_cut):
        self.iov = iov
        self.systematics = systematics
        self.no_syst = no_syst
        self.ak8_pt_min = ak8_pt_min
        self.ht_cut = ht_cut

    @staticmethod
    def _add_p4(collection):
        collection["p4"] = ak.with_name(
            collection[["pt", "eta", "phi", "mass"]],
            "PtEtaPhiMLorentzVector",
        )
        return collection

    def prepare_for_corrections(self, events, is_data):
        fatjets = self._add_p4(events.FatJet)
        jets = self._add_p4(events.Jet)

        if not is_data:
            genjets = self._add_p4(events.GenJet)
            _matched_gen = fatjets.p4.nearest(genjets.p4, threshold=0.2)
            fatjets["pt_gen"] = ak.values_astype(ak.fill_none(_matched_gen.pt, 0), np.float32)

        return fatjets, jets

    def build_corrections(self, events, is_data):
        no_corrections = "jes" not in self.systematics and "jer" not in self.systematics
        if no_corrections or self.no_syst or is_data:
            return None

        fatjets, jets = self.prepare_for_corrections(events, is_data)
        corrected_fatjets = GetJECUncertainties(fatjets, events, self.iov, R="AK8", isData=is_data)
        corrected_jets = GetJECUncertainties(jets, events, self.iov, R="AK4", isData=is_data)

        corrections = []
        if "jes" in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets,             "FatJet": corrected_fatjets},             "nominal"),
                ({"Jet": corrected_jets.JES_jes.up,  "FatJet": corrected_fatjets.JES_jes.up},  "jesUp"),
                ({"Jet": corrected_jets.JES_jes.down,"FatJet": corrected_fatjets.JES_jes.down}, "jesDown"),
            ])
        if "jer" in self.systematics:
            if not corrections:
                corrections.append(({"Jet": corrected_jets, "FatJet": corrected_fatjets}, "nominal"))
            corrections.extend([
                ({"Jet": corrected_jets.JER.up,  "FatJet": corrected_fatjets.JER.up},  "jerUp"),
                ({"Jet": corrected_jets.JER.down, "FatJet": corrected_fatjets.JER.down}, "jerDown"),
            ])

        return corrections

    def prepare_analysis_objects(self, events, is_data):
        fatjets     = self._add_p4(events.FatJet)
        subjets     = self._add_p4(events.SubJet)
        jets        = self._add_p4(events.Jet)
        genjets     = None
        genjetak8   = None
        subgenjetak8 = None
        if not is_data:
            genjets = self._add_p4(events.GenJet)
            if "GenJetAK8" in events.fields:
                genjetak8 = self._add_p4(events.GenJetAK8)
            if "SubGenJetAK8" in events.fields:
                subgenjetak8 = self._add_p4(events.SubGenJetAK8)
        return fatjets, subjets, jets, genjets, genjetak8, subgenjetak8

    def baseline_masks(self, events, fatjets, jets):
        jet_veto_map_mask = GetJetVetoMapMask(jets, self.iov)
        ht_mask = (
            ak.sum(jets[(jets.pt > _AK4_PT_MIN) & (np.abs(jets.eta) < _AK4_ETA_MAX)].pt, axis=1)
            > self.ht_cut
        )

        # No separate analysis-level jet-ID cut is applied here. TightLepVeto is
        # evaluated only for the official jet-veto-map eligibility above.

        jetkin = (fatjets.pt > self.ak8_pt_min) & (np.abs(getRapidity(fatjets.p4)) < _AK8_RAPIDITY_MAX)

        # jetkin_mask: ≥1 FatJet passes kinematics — used as a PackedSelection cut.
        # twofat_mask: ≥2 FatJets survive after filtering — the actual physics requirement.
        # Both are needed: jetkin_mask enters the selection before the filtered collection exists.
        jetkin_mask      = ak.any(jetkin, axis=1)
        fatjets_selected = fatjets[jetkin]
        twofat_mask      = ak.num(fatjets_selected) > 1

        event_mask = ht_mask & jetkin_mask & twofat_mask

        masks = {
            "jetVetoMap": jet_veto_map_mask,
            "htCut":      ht_mask,
            "jetkincut":  jetkin_mask,
            "twoFatJets": twofat_mask,
            "event":      jet_veto_map_mask & event_mask,
        }
        return fatjets_selected, masks
