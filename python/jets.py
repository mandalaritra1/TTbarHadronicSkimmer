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


def _softdrop_mass(fatjets, subjets):
    """Invariant mass of each fat jet's two soft-drop subjets (from their current
    pt/eta/phi/mass). Fat jets without two subjets keep their NanoAOD msoftdrop."""
    idx1 = ak.mask(fatjets.subJetIdx1, fatjets.subJetIdx1 >= 0)
    idx2 = ak.mask(fatjets.subJetIdx2, fatjets.subJetIdx2 >= 0)
    s1, s2 = subjets[idx1], subjets[idx2]

    def four_momentum(s):
        px, py = s.pt * np.cos(s.phi), s.pt * np.sin(s.phi)
        pz = s.pt * np.sinh(s.eta)
        return px, py, pz, np.sqrt(px**2 + py**2 + pz**2 + s.mass**2)

    px1, py1, pz1, e1 = four_momentum(s1)
    px2, py2, pz2, e2 = four_momentum(s2)
    m2 = (e1 + e2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
    msd = ak.values_astype(ak.fill_none(np.sqrt(np.maximum(m2, 0)), -1.0), np.float32)
    return ak.where(msd < 0, fatjets.msoftdrop, msd)


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

        # msoftdrop is the mass of the two soft-drop subjets, which NanoAOD corrects
        # with the AK4 PUPPI JEC of its production. For Run 3 the subjets are
        # un-corrected (rawFactor) and re-corrected with the vendored JEC (+ JER for
        # MC), and msoftdrop is rebuilt from them, nominal and JES/JER-varied alike
        # (JMAR recipe, as in smp_jetmass_run2). Nominal JMS = JMR = 1.000.
        if run3:
            corrected_subjets = self._corrected_subjets(events, is_data)

            def msd_from(varied_subjets, varied_fatjets):
                return with_msd(varied_fatjets, _softdrop_mass(varied_fatjets, varied_subjets))
        else:
            corrected_subjets = None

            def msd_from(varied_subjets, varied_fatjets):
                # Run 2: keep NanoAOD msoftdrop, scaled by the jet's own pT ratio.
                ratio = ak.where(corrected_fatjets.pt > 0, varied_fatjets.pt / corrected_fatjets.pt, 1.0)
                return with_msd(varied_fatjets, corrected_fatjets.msoftdrop * ratio)

        def with_msd(fatjet_collection, msd):
            return ak.with_field(fatjet_collection, msd, "msoftdrop")

        def sub(attr=None, direction=None):
            if corrected_subjets is None:
                return None
            return corrected_subjets if attr is None else getattr(corrected_subjets[attr], direction)

        corrected_fatjets = msd_from(sub(), corrected_fatjets)
        nominal_msd = corrected_fatjets.msoftdrop

        corrections = [({"Jet": corrected_jets, "FatJet": corrected_fatjets}, "nominal")]
        if not variations:
            return corrections
        if "jes" in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets.JES_jes.up,   "FatJet": msd_from(sub("JES_jes", "up"), corrected_fatjets.JES_jes.up)},     "jesUp"),
                ({"Jet": corrected_jets.JES_jes.down, "FatJet": msd_from(sub("JES_jes", "down"), corrected_fatjets.JES_jes.down)}, "jesDown"),
            ])
        if "jer" in self.systematics:
            corrections.extend([
                ({"Jet": corrected_jets.JER.up,   "FatJet": msd_from(sub("JER", "up"), corrected_fatjets.JER.up)},     "jerUp"),
                ({"Jet": corrected_jets.JER.down, "FatJet": msd_from(sub("JER", "down"), corrected_fatjets.JER.down)}, "jerDown"),
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

    def _corrected_subjets(self, events, is_data):
        subjets = self._add_p4(events.SubJet)
        if not is_data:
            # JER smearing needs the matched gen subjet pT (as smp_jetmass_run2).
            gen = self._add_p4(events.SubGenJetAK8)
            matched = subjets.p4.nearest(gen.p4, threshold=0.4)
            subjets["pt_gen"] = ak.values_astype(ak.fill_none(matched.pt, 0), np.float32)
        return GetJECUncertainties(subjets, events, self.iov, R="AK4", isData=is_data)

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
