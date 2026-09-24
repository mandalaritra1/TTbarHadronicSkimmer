import awkward as ak
import numpy as np

from corrections import _JSONPOG_JME_DIR, GetJECUncertainties, GetJetVetoMapMask
from functions import getRapidity

# Jet-collection variations run as separate passes of the processor.
_JET_VARIATIONS = {"jes", "jer", "jms", "jmr"}
# Soft-drop mass scale and resolution uncertainties (nominal JMS = JMR = 1.000).
_JMS_UNC = 0.01
_JMR_UNC = 0.02


def _jmr_seed(events):
    """Seed the JMR smearing from the chunk's event numbers so reruns reproduce it."""
    ids = ak.to_numpy(events.event[:16]).astype(np.uint64)
    return int(np.bitwise_xor.reduce(ids)) if len(ids) else 0

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
        variations = (
            set() if (self.no_syst or is_data)
            else _JET_VARIATIONS.intersection(self.systematics)
        )
        # Run-3 IOVs re-apply the vendored JEC (and JER for MC) on every pass, data
        # and --noSyst included, so data and MC carry the same JEC version. Run 2
        # keeps the NanoAOD jets unless variations are requested (its legacy data
        # JEC path is not usable).
        run3 = self.iov in _JSONPOG_JME_DIR
        if not variations and not run3:
            return None

        fatjets, jets = self.prepare_for_corrections(events, is_data)
        corrected_fatjets = GetJECUncertainties(fatjets, events, self.iov, R="AK8", isData=is_data)
        corrected_jets = GetJECUncertainties(jets, events, self.iov, R="AK4", isData=is_data)

        # msoftdrop is not re-corrected: NanoAOD builds it from subjets that already
        # carry AK4 PUPPI corrections, and scaling it by the AK8 JEC shifts the data
        # top-mass peak (see toptag_sf_processor.py). Nominal JMS = JMR = 1.000, so the
        # nominal msoftdrop stays on the NanoAOD value; only the variations move it.
        nominal_msd = corrected_fatjets.msoftdrop

        def with_msd(fatjet_collection, msd):
            return ak.with_field(fatjet_collection, msd, "msoftdrop")

        def follow_pt(varied):
            # JES/JER variations move msoftdrop with the jet's own pT ratio.
            ratio = ak.where(corrected_fatjets.pt > 0, varied.pt / corrected_fatjets.pt, 1.0)
            return with_msd(varied, nominal_msd * ratio)

        corrections = [({"Jet": corrected_jets, "FatJet": corrected_fatjets}, "nominal")]
        if not variations:
            return corrections
        if "jes" in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets.JES_jes.up,   "FatJet": follow_pt(corrected_fatjets.JES_jes.up)},   "jesUp"),
                ({"Jet": corrected_jets.JES_jes.down, "FatJet": follow_pt(corrected_fatjets.JES_jes.down)}, "jesDown"),
            ])
        if "jer" in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets.JER.up,   "FatJet": follow_pt(corrected_fatjets.JER.up)},   "jerUp"),
                ({"Jet": corrected_jets.JER.down, "FatJet": follow_pt(corrected_fatjets.JER.down)}, "jerDown"),
            ])
        if "jms" in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets, "FatJet": with_msd(corrected_fatjets, nominal_msd * (1 + _JMS_UNC))}, "jmsUp"),
                ({"Jet": corrected_jets, "FatJet": with_msd(corrected_fatjets, nominal_msd * (1 - _JMS_UNC))}, "jmsDown"),
            ])
        if "jmr" in self.systematics:
            # Gaussian smearing can only widen the resolution: jmrUp is smeared,
            # jmrDown is filled with the nominal mass and mirrored around nominal
            # (2*nom - up) when the 2DAlphabet templates are written.
            counts = ak.num(nominal_msd, axis=1)
            flat_msd = ak.to_numpy(ak.flatten(nominal_msd, axis=1))
            rng = np.random.default_rng(_jmr_seed(events))
            smeared = flat_msd * (1 + _JMR_UNC * rng.standard_normal(len(flat_msd)))
            corrections.extend([
                ({"Jet": corrected_jets, "FatJet": with_msd(corrected_fatjets, ak.unflatten(smeared, counts))}, "jmrUp"),
                ({"Jet": corrected_jets, "FatJet": corrected_fatjets}, "jmrDown"),
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
