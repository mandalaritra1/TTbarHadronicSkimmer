"""Coffea processor for the GloParTv3 top-tag efficiency scale-factor measurement.

Semileptonic ttbar **tag-and-probe**, separate from both ``TTbarResProcessor``
(the all-hadronic analysis) and ``TopTagWPProcessor`` (the WP-threshold
derivation). It selects a clean semileptonic-ttbar event on the *tag* side
(``t -> b l nu``: one high-pT lepton + MET + a leptonic-side b-tagged AK4) and
then treats the recoiling boosted AK8 as the *probe* top (``t -> b q q'``).
**No top-tag cut defines the probe** -- the tagger score and its per-WP pass/fail
are stored per probe so the efficiency ``eps = N(pass)/N(all)`` can be measured
downstream in every WP and pT binning.

The output is a small per-probe flat ntuple (one row per selected probe),
mirroring the WP-processor's skinny-ntuple pattern: nested
``{dataset -> {field -> column_accumulator}}``, merged by concatenation across
chunks, kept per-dataset so cross-section weighting happens at save time.

Per-probe record (all numeric so column-accumulators concatenate cleanly; the
sample name / IOV live in ``datasets_metadata``):
  - ``probe_pt, probe_eta, probe_msd`` : probe AK8 kinematics.
  - ``D``                              : TopvsQCD discriminant (``_tscore`` formula).
  - ``pass_very_tight..pass_very_loose`` : int8, D >= per-pT-bin WP threshold.
  - ``merge_cat``  : -1 data/none, 0 not-merged, 1 partially, 2 fully merged.
  - ``lepton_channel`` : 0 = muon (P0 is mu-only).
  - ``mu_pt, mu_eta, met, dphi_mu_probe, dr_probe_blep, n_bjet`` : tag-side
    kinematics kept for building systematic variations.
  - ``genweight`` : per-event weight (genWeight[* PU] for MC, 1.0 for data),
    RAW -- cross-section normalization (xsec*lumi/sumw) is applied at save time.

Bookkeeping accumulators: ``sumw`` (sum genWeight over ALL events, pre-selection,
for the xsec norm), ``nevents``, ``nevents_raw``, and a ``cutflow`` dict.

Year handling is data-driven via the per-IOV config dicts so adding an IOV is a
localized change; there is no hard-coded ``'2024'`` in the selection logic.
"""

import os
import sys
import json

import numpy as np
import awkward as ak
from coffea import processor

# coffea 2025.x (lxplus LCG_107) stripped the accumulator classes out of
# ``coffea.processor``; 2026.x (Mac) still ships them. Import the real classes
# when present, otherwise provide lightweight, API-compatible shims: a plain
# ``defaultdict`` (merged by the driver's recursive dict walk) and a
# concatenating column container with a ``.value`` array.
try:
    column_accumulator = processor.column_accumulator
    defaultdict_accumulator = processor.defaultdict_accumulator
except AttributeError:  # pragma: no cover - only on the stripped coffea build
    from collections import defaultdict as _defaultdict

    def defaultdict_accumulator(factory):
        return _defaultdict(factory)

    class column_accumulator:  # noqa: N801 - mirror coffea's lowercase name
        def __init__(self, value):
            self._value = np.asarray(value)

        @property
        def value(self):
            return self._value

        def __add__(self, other):
            return column_accumulator(
                np.concatenate([self._value, other._value]))

        def __len__(self):
            return len(self._value)

# Mirror ttbarprocessor.py's import convention so this works both locally and on
# Dask workers (where the package path is not importable).
sys.path.append(os.getcwd() + '/python/')
from truthstudy import get_hadronic_tops  # noqa: E402

try:  # corrections pull in correctionlib + vendored json; keep import soft.
    from corrections import getMETFilter, getLumiMask  # noqa: E402
    _HAVE_CORRECTIONS = True
except Exception:  # pragma: no cover - exercised only when corrections deps miss
    _HAVE_CORRECTIONS = False


# ---------------------------------------------------------------------------
# Tagger discriminant -- identical to TTbarResProcessor._tscore (v15 IOVs) and
# TopTagWPProcessor._topvsqcd_globalParT3, so the efficiency we calibrate is the
# efficiency the analysis applies.
# ---------------------------------------------------------------------------
def _topvsqcd_globalParT3(fatjets):
    """(TopbWqq + TopbWq) / (TopbWqq + TopbWq + QCD)."""
    num = fatjets.globalParT3_TopbWqq + fatjets.globalParT3_TopbWq
    return num / (num + fatjets.globalParT3_QCD)


# WP names in canonical order of increasing target QCD mis-tag (0.1% -> 5.0%).
WP_NAMES = ['very_tight', 'tight', 'medium', 'loose', 'very_loose']

# pT-binned WP thresholds (falls back to the derived 2024 table if the json is
# absent). Edges [400,500,600,800,1200,3000]; last edge stands in for infinity.
_WP_THRESHOLDS_2024 = {
    'pt_bin_edges': [400.0, 500.0, 600.0, 800.0, 1200.0, 3000.0],
    # Full-statistics derivation (outputs/toptag_wp_2024_full.json, swapped in
    # 2026-07-30). The previous table came from a single-QCD-bin local run and
    # carried spurious pT structure (up to +-50% off in per-pT MC closure; its
    # 1200-3000 thresholds were badly off) -- see research-notes
    # glopartv3_toptag_wp_data_mistag_2024.
    'thresholds': {
        'very_tight': [0.98162, 0.98090, 0.98084, 0.98110, 0.98055],
        'tight':      [0.92639, 0.92298, 0.91898, 0.91718, 0.91653],
        'medium':     [0.85537, 0.84771, 0.83933, 0.83672, 0.83743],
        'loose':      [0.64701, 0.62889, 0.61470, 0.61178, 0.62450],
        'very_loose': [0.39126, 0.37074, 0.35817, 0.36084, 0.38415],
    },
}


def load_wp_thresholds(iov='2024', path=None):
    """Return {'pt_bin_edges':[...], 'thresholds':{wp:[per-pt-bin]}} for ``iov``.

    Prefers the derived table ``data/toptag/toptag_wp_<iov>.json`` (the output of
    the companion WP derivation); falls back to the embedded 2024 values so the
    processor is self-contained on Dask workers where the file may be absent.
    """
    if path is None:
        path = f'data/toptag/toptag_wp_{iov}.json'
    try:
        with open(path) as f:
            tab = json.load(f)
        edges = tab['pt_bin_edges']
        wps = tab['working_points']
        thresholds = {wp: wps[wp]['threshold'] for wp in WP_NAMES if wp in wps}
        if thresholds:
            return {'pt_bin_edges': edges, 'thresholds': thresholds}
    except Exception:
        pass
    return _WP_THRESHOLDS_2024


# Per-IOV single-lepton HLT paths (OR'd; missing paths ignored). Mu50 is fully
# efficient for the pT>55 tag muon and unprescaled; IsoMu24 kept as a fallback.
TRIGGER_CONFIG = {
    'mu': {
        '2022': ['Mu50', 'IsoMu24'], '2023': ['Mu50', 'IsoMu24'],
        '2024': ['Mu50', 'IsoMu24'], '2025': ['Mu50', 'IsoMu24'],
        '2022preEE': ['Mu50', 'IsoMu24'], '2022postEE': ['Mu50', 'IsoMu24'],
        '2023preBPix': ['Mu50', 'IsoMu24'], '2023postBPix': ['Mu50', 'IsoMu24'],
    },
}

# AK4 b-tag: DeepJet (btagDeepFlavB) medium WP. NOTE: 0.3086 is the Run-3 2022
# DeepJet medium value used as a proxy -- update with the official Summer24 BTV
# medium WP when available. btagPNetB is used as a fallback if DeepJet is absent.
BTAG_CONFIG = {
    '2022': {'field': 'btagDeepFlavB', 'wp': 0.3086},
    '2023': {'field': 'btagDeepFlavB', 'wp': 0.3086},
    '2024': {'field': 'btagDeepFlavB', 'wp': 0.3086},
    '2025': {'field': 'btagDeepFlavB', 'wp': 0.3086},
}
_PNET_FALLBACK_WP = 0.2605  # Run-3 2022 ParticleNet-AK4 medium proxy

# --- Selection constants (semileptonic-ttbar T&P; standard boosted-top values) --
MU_PT_MIN = 55.0
MU_ETA_MAX = 2.4
MU_ISO_MAX = 0.15            # pfRelIso04_all
EL_PT_MIN = 55.0             # electron veto (reject dilepton ttbar)
EL_ETA_MAX = 2.5
MET_MIN = 50.0
AK4_PT_MIN = 30.0
AK4_ETA_MAX = 2.5
DR_LEP_B_MAX = 1.5           # leptonic-side b: close to the tag lepton
PROBE_PT_MIN = 400.0
PROBE_ETA_MAX = 2.5
PROBE_MSD_MIN = 105.0
PROBE_MSD_MAX = 210.0
DPHI_LEP_PROBE_MIN = 2.0     # probe recoils against the lepton
DR_PROBE_BLEP_MIN = 0.8      # probe is not the leptonic b
DR_MERGE = 0.8               # daughter-quark containment radius

# merge_cat codes -- follow CMS DP-2025/010 (four categories, AK8 dR(quark,axis)<0.8):
#   fully merged t : all three quarks (b + both W quarks) in the jet
#   fully merged W : both W quarks in, b NOT in
#   semi merged    : two quarks in, including the b (b + one W quark)
#   not merged     : none of the above (<=1 quark, or only b, or only one W quark)
MC_DATA = -1
MC_NOT = 0
MC_SEMI = 1
MC_W = 2
MC_FULL = 3

# lepton_channel codes
CH_MU = 0
CH_EL = 1

SF_NTUPLE_FIELDS = {
    'probe_pt': np.float32, 'probe_eta': np.float32, 'probe_msd': np.float32,
    'D': np.float32,
    'pass_very_tight': np.int8, 'pass_tight': np.int8, 'pass_medium': np.int8,
    'pass_loose': np.int8, 'pass_very_loose': np.int8,
    'merge_cat': np.int8, 'b_merged': np.int8, 'n_wq_merged': np.int8,
    'lepton_channel': np.int8,
    'mu_pt': np.float32, 'mu_eta': np.float32, 'met': np.float32,
    'dphi_mu_probe': np.float32, 'dr_probe_blep': np.float32, 'n_bjet': np.int8,
    'genweight': np.float32, 'pu_weight': np.float32,
    'jec_factor': np.float32,
}

SIGNAL_NAME_PREFIXES = ('TT', 'ZPRIME', 'ZPTOTT', 'RSG', 'RSGLUON')


def _delta_phi(a_phi, b_phi):
    """Wrapped phi difference in (-pi, pi]; broadcasts flat vs jagged."""
    return (a_phi - b_phi + np.pi) % (2 * np.pi) - np.pi


def _delta_r_flat_jagged(flat_eta, flat_phi, jag_eta, jag_phi):
    """dR between one flat (per-event) object and a jagged collection."""
    deta = flat_eta - jag_eta
    dphi = _delta_phi(flat_phi, jag_phi)
    return np.sqrt(deta * deta + dphi * dphi)


class TopTagSFProcessor(processor.ProcessorABC):
    """Semileptonic-ttbar tag-and-probe -> per-probe flat ntuple for top-tag SFs.

    Parameters
    ----------
    iov : str
        IOV key (e.g. ``'2024'``). Selects trigger / b-tag config and WP table.
    channel : str
        Lepton channel for the tag side. ``'mu'`` for P0 (only value supported).
    wp_thresholds : dict or None
        ``{'pt_bin_edges':[...], 'thresholds':{wp:[...]}}``. If None, loaded from
        the derived table (see :func:`load_wp_thresholds`) at construction so the
        arrays pickle to Dask workers.
    apply_pu : bool
        Multiply the stored genweight by the nominal PU weight (MC only). Off by
        default for P0 -- keeps worker file dependencies minimal; PU is a small
        correction added in P1.
    apply_lumimask : bool
        Apply the golden-JSON lumi mask to data (default True when corrections
        are importable).
    """

    def __init__(self, iov='2024', channel='mu', wp_thresholds=None,
                 apply_pu=False, apply_jec=False, apply_jec_syst=False,
                 apply_lumimask=True):
        if channel != 'mu':
            raise NotImplementedError(
                f"channel '{channel}' not implemented; P0 is mu-only.")
        self.iov = iov
        self.channel = channel
        self._trig_cfg = TRIGGER_CONFIG[channel].get(iov, [])
        self._btag_cfg = BTAG_CONFIG.get(iov, BTAG_CONFIG.get('2024'))
        wp = wp_thresholds or load_wp_thresholds(iov)
        self._wp_edges = np.asarray(wp['pt_bin_edges'], dtype=np.float64)
        # dense [n_wp, n_ptbin] threshold matrix in WP_NAMES order.
        self._wp_names = [w for w in WP_NAMES if w in wp['thresholds']]
        self._wp_thr = np.asarray(
            [wp['thresholds'][w] for w in self._wp_names], dtype=np.float64)
        self.apply_pu = bool(apply_pu)
        self.apply_jec = bool(apply_jec)
        self.jec_syst = bool(apply_jec_syst)
        self.apply_lumimask = bool(apply_lumimask) and _HAVE_CORRECTIONS

    # -- gen-level helpers --------------------------------------------------
    @staticmethod
    def _gen_top_decay_quarks(events):
        """Return (b_quarks, w_quarks) as separate per-event jagged GenPart
        collections: the b quarks straight from a top, and the light quarks from a
        hadronic W. Separating them lets the merge category distinguish "fully
        merged W" (W quarks in, b out) from a fully merged top.

        Identification is by *direct mother species*, not by walking the top ->
        W(first) -> W(last) -> qq' copy chain: a light quark whose direct mother is
        a W (pdgId 24) is a first-copy W daughter, and a b whose direct mother is a
        top is a first-copy top daughter. In ttbar every W descends from a top, so
        this needs no chain-walking and -- validated on real Summer24 v15 gen --
        recovers exactly two W quarks in 100% of semileptonic events, versus ~2/3
        for the multi-copy chain trace (which also mis-swept beam partons via the
        ak.max `-1` sentinel matching genPartIdxMother == -1). First-copy partons
        are also the right kinematics for dR merge matching (pre-shower direction).
        """
        genparts = events.GenPart
        pdg = abs(genparts.pdgId)
        mother = genparts.genPartIdxMother
        has_mother = mother >= 0
        # pdgId of each particle's mother (guard the -1 sentinel before gathering).
        mother_safe = ak.where(has_mother, mother, 0)
        mom_pdg = abs(pdg[mother_safe])

        b_mask = (pdg == 5) & (mom_pdg == 6) & has_mother
        wq_mask = (pdg >= 1) & (pdg <= 5) & (mom_pdg == 24) & has_mother
        return genparts[b_mask], genparts[wq_mask]

    def _probe_merge_cat(self, events, probe_eta, probe_phi):
        """Per-event merge classification from b- and W-quark containment.

        Returns (merge_cat, b_merged, n_wq_merged), all int8, following
        CMS DP-2025/010. ``probe_eta``/``probe_phi`` are flat awkward arrays (one
        probe per event). The leptonic-side b is geometrically far from the probe,
        so counting all top b-quarks within dR<0.8 effectively counts the hadronic
        top's b; W quarks come only from the hadronic W.
        """
        n = len(probe_eta)
        if n == 0 or 'GenPart' not in events.fields:
            z = np.zeros(n, dtype=np.int8)
            return z, z.copy(), z.copy()
        b_quarks, w_quarks = self._gen_top_decay_quarks(events)
        dr_b = _delta_r_flat_jagged(probe_eta, probe_phi, b_quarks.eta, b_quarks.phi)
        dr_w = _delta_r_flat_jagged(probe_eta, probe_phi, w_quarks.eta, w_quarks.phi)
        b_in = ak.to_numpy(ak.sum(dr_b < DR_MERGE, axis=1)) >= 1
        nwq = ak.to_numpy(ak.sum(dr_w < DR_MERGE, axis=1))

        cat = np.full(n, MC_NOT, dtype=np.int8)
        cat[(nwq >= 1) & b_in] = MC_SEMI          # b + >=1 W quark (>=2 quarks)
        cat[(nwq >= 2) & (~b_in)] = MC_W           # both W quarks, b out
        cat[(nwq >= 2) & b_in] = MC_FULL           # all three
        return (cat, b_in.astype(np.int8), nwq.astype(np.int8))

    # -- reco helpers -------------------------------------------------------
    def _trigger_mask(self, events):
        if 'HLT' not in events.fields or not self._trig_cfg:
            return np.ones(len(events), dtype=bool)
        avail = [p for p in self._trig_cfg if p in events.HLT.fields]
        if not avail:
            return np.ones(len(events), dtype=bool)
        m = events.HLT[avail[0]]
        for p in avail[1:]:
            m = m | events.HLT[p]
        return ak.to_numpy(m)

    @staticmethod
    def _met_pt(events):
        if 'PuppiMET' in events.fields:
            return events.PuppiMET.pt
        if 'MET' in events.fields:
            return events.MET.pt
        return ak.zeros_like(ak.num(events.Jet, axis=1), dtype=np.float64)

    def _btag_mask(self, jets):
        field = self._btag_cfg['field']
        wp = self._btag_cfg['wp']
        if field not in jets.fields:
            field, wp = 'btagPNetB', _PNET_FALLBACK_WP
        if field not in jets.fields:
            # no recognizable b-tag branch -> accept none (selection then empty)
            return ak.zeros_like(jets.pt, dtype=bool)
        return jets[field] > wp

    def _passwp_columns(self, probe_pt, D):
        """Return {'pass_<wp>': int8 array} using the per-pT-bin thresholds."""
        # bin index into edges; clip so <edge[0] and >=edge[-1] map to end bins.
        ib = np.clip(np.digitize(probe_pt, self._wp_edges) - 1,
                     0, self._wp_thr.shape[1] - 1)
        out = {}
        for k, name in enumerate(self._wp_names):
            thr = self._wp_thr[k][ib]
            out[f'pass_{name}'] = (D >= thr).astype(np.int8)
        # any WP absent from the table -> fail (keeps a stable 5-column schema).
        for name in WP_NAMES:
            out.setdefault(f'pass_{name}', np.zeros(len(D), dtype=np.int8))
        return out

    # -- coffea API ---------------------------------------------------------
    def process(self, events):
        dataset = events.metadata['dataset']
        meta = getattr(events, 'metadata', {}) or {}
        is_mc = bool(meta.get('is_mc', 'genWeight' in events.fields))

        output = {
            'ntuple': {},
            'sumw': defaultdict_accumulator(float),
            'nevents': defaultdict_accumulator(int),
            'nevents_raw': defaultdict_accumulator(int),
            'cutflow': defaultdict_accumulator(int),
        }
        n_raw = len(events)
        output['nevents_raw'][dataset] += n_raw
        output['cutflow'][f'{dataset}/all'] += n_raw

        # genWeight bookkeeping over ALL events (pre-selection) for the xsec norm.
        if is_mc and 'genWeight' in events.fields:
            w_evt_full = events.genWeight
        else:
            w_evt_full = ak.ones_like(ak.Array(np.ones(n_raw, dtype=np.float64)))
        output['sumw'][dataset] += float(ak.sum(w_evt_full))

        # ---- event-level preselection (no None: all whole-event booleans) ----
        keep = self._trigger_mask(events)
        keep = keep & getMETFilter(self.iov, events) if _HAVE_CORRECTIONS \
            else keep
        if (not is_mc) and self.apply_lumimask and 'run' in events.fields:
            try:
                lm = getLumiMask(self.iov)
                keep = keep & np.asarray(
                    lm(ak.to_numpy(events.run),
                       ak.to_numpy(events.luminosityBlock)))
            except Exception:
                pass

        muons = events.Muon
        mu_sel = ((muons.pt > MU_PT_MIN) & (abs(muons.eta) < MU_ETA_MAX)
                  & muons.tightId & (muons.pfRelIso04_all < MU_ISO_MAX))
        n_mu = ak.to_numpy(ak.sum(mu_sel, axis=1))
        # electron veto (suppress dilepton ttbar)
        if 'Electron' in events.fields and 'pt' in events.Electron.fields:
            els = events.Electron
            el_id = els.cutBased >= 4 if 'cutBased' in els.fields \
                else ak.ones_like(els.pt, dtype=bool)
            el_sel = (els.pt > EL_PT_MIN) & (abs(els.eta) < EL_ETA_MAX) & el_id
            n_el = ak.to_numpy(ak.sum(el_sel, axis=1))
        else:
            n_el = np.zeros(n_raw, dtype=np.int64)

        met = ak.to_numpy(self._met_pt(events))
        keep = keep & (n_mu == 1) & (n_el == 0) & (met > MET_MIN)
        output['cutflow'][f'{dataset}/presel'] += int(np.sum(keep))
        if not np.any(keep):
            output['nevents'][dataset] += 0
            return output

        # ---- reduce to preselected events, then do object-level probe/b ----
        ev = events[keep]
        w_evt = w_evt_full[keep]
        muons = ev.Muon
        mu_sel = ((muons.pt > MU_PT_MIN) & (abs(muons.eta) < MU_ETA_MAX)
                  & muons.tightId & (muons.pfRelIso04_all < MU_ISO_MAX))
        lead_mu = ak.firsts(muons[mu_sel])           # guaranteed present (n_mu==1)
        # keep muon geometry in AWKWARD space when combining with jagged
        # collections: a numpy operand on the LEFT would coerce a rectangular
        # awkward array to (n, k) and broadcast wrongly. Convert to numpy only
        # for flat-vs-flat arithmetic and final storage.
        mu_eta = lead_mu.eta
        mu_phi = lead_mu.phi

        variations = [('', None, None)]
        if is_mc and self.jec_syst and _HAVE_CORRECTIONS:
            # Exact JES/JER propagation: the FULL selection (lep-side b jet,
            # probe, thresholds) is re-run under each varied jet collection.
            # Fractional shifts from the official factory (V3 Total, JRV1
            # hybrid smearing with gen matching) are applied around the NanoAOD
            # pT so the nominal stays at the analysis jet scale. MET is left
            # nominal (no Type-1 recompute in this analysis); msoftdrop is
            # never scaled by jet-level factors (subjet-corrected quantity).
            try:
                from corrections import GetJECUncertainties
                fjc = GetJECUncertainties(ev.FatJet, ev, self.iov, R='AK8',
                                          isData=False)
                jc = GetJECUncertainties(ev.Jet, ev, self.iov, R='AK4',
                                         isData=False)
                fj0 = ev.FatJet.pt
                j0 = ev.Jet.pt
                for lbl, fv, jv in (
                        ('__jesUp', fjc.JES_jes.up.pt / fjc.pt,
                         jc.JES_jes.up.pt / jc.pt),
                        ('__jesDown', fjc.JES_jes.down.pt / fjc.pt,
                         jc.JES_jes.down.pt / jc.pt),
                        ('__jerUp', fjc.JER.up.pt / fjc.pt,
                         jc.JER.up.pt / jc.pt),
                        ('__jerDown', fjc.JER.down.pt / fjc.pt,
                         jc.JER.down.pt / jc.pt)):
                    variations.append((lbl, fj0 * fv, j0 * jv))
            except Exception as exc:
                print(f"[toptag_sf] WARNING: JEC-syst variations FAILED for "
                      f"{dataset} ({type(exc).__name__}: {exc}) -- writing "
                      f"NOMINAL ONLY", flush=True)
                variations = [('', None, None)]

        met_sel = met[keep]
        for lbl, fpt, jpt in variations:
            self._select_and_fill(ev, w_evt, met_sel, lead_mu, mu_eta, mu_phi,
                                  dataset, is_mc, output, suffix=lbl,
                                  fj_pt=fpt, jet_pt=jpt)
        return output

    def _select_and_fill(self, ev, w_evt, met_sel, lead_mu, mu_eta, mu_phi,
                         dataset, is_mc, output, suffix='', fj_pt=None,
                         jet_pt=None):
        """Object selection + per-probe fill for ONE jet-energy variation.

        ``fj_pt`` / ``jet_pt`` optionally replace the AK8/AK4 pT (same jagged
        layout); everything else (muon, MET, mSD, D) stays nominal. Output keys
        are suffixed (``<dataset>__jesUp`` etc.); ``suffix=''`` is the nominal
        path and byte-identical to the pre-refactor behavior.
        """
        # leptonic-side b-tagged AK4
        jets = ev.Jet
        if jet_pt is not None:
            jets = ak.with_field(jets, jet_pt, 'pt')
        jet_ok = (jets.pt > AK4_PT_MIN) & (abs(jets.eta) < AK4_ETA_MAX)
        if 'jetId' in jets.fields:
            jet_ok = jet_ok & (jets.jetId >= 2)
        bjets = jets[jet_ok & self._btag_mask(jets)]
        dr_mu_b = _delta_r_flat_jagged(mu_eta, mu_phi, bjets.eta, bjets.phi)
        lep_b = bjets[dr_mu_b < DR_LEP_B_MAX]
        n_lepb = ak.to_numpy(ak.num(lep_b, axis=1))
        b_lep = ak.firsts(lep_b)                     # leading pt lep-side b (or None)
        blep_eta = ak.fill_none(b_lep.eta, 999.0)
        blep_phi = ak.fill_none(b_lep.phi, 0.0)

        # probe AK8 candidates
        fj = ev.FatJet
        if fj_pt is not None:
            fj = ak.with_field(fj, fj_pt, 'pt')
        fj0 = fj
        if self.apply_jec and _HAVE_CORRECTIONS:
            try:
                from corrections import GetJECUncertainties
                pt_before = fj.pt
                # CMS recommendation: JEC on BOTH data and MC (data gets its own
                # run-dependent L2L3Residual chain); JER smearing on MC ONLY --
                # you smear simulation to match the data resolution, and the
                # JSON-POG file ships no DATA PtResolution to smear data with.
                # Raw pT/mass come from rawFactor inside GetJECUncertainties.
                fj = GetJECUncertainties(fj, ev, self.iov, R='AK8',
                                         isData=not is_mc)
                f = fj.pt / pt_before
                # msoftdrop is deliberately LEFT ALONE. It is not the AK8 mass
                # scaled by the AK8 JEC: NanoAOD builds it from subjets carrying
                # their own AK4 PUPPI corrections (JMAR: "subjets must be
                # corrected using AK4 PUPPI corrections"). Scaling it by the
                # jet-level JEC ratio was tried (v4) and is measurably WRONG --
                # it dragged the data top-mass peak from 175 to 167 GeV while MC
                # went 181 -> 183, i.e. the data/MC peak offset went -6 -> -16
                # GeV. The top mass is the same object in both, so that is
                # unphysical, and it faked an SF near 1 by changing which probes
                # survive the mSD window. Correcting mSD properly needs JMS/JMR
                # (unavailable), so it stays on the NanoAOD value.
                fj = ak.with_field(fj, f, 'jec_factor')
            except Exception as exc:
                print(f"[toptag_sf] WARNING: JEC/JER FAILED for {dataset} "
                      f"(is_mc={is_mc}, {type(exc).__name__}: {exc}) -- running "
                      f"with UNCORRECTED jets", flush=True)
                fj = ak.with_field(fj0, ak.ones_like(fj0.pt), 'jec_factor')
        else:
            fj = ak.with_field(fj, ak.ones_like(fj.pt), 'jec_factor')
        dphi_mu_fj = np.abs(_delta_phi(mu_phi, fj.phi))
        dr_fj_blep = _delta_r_flat_jagged(blep_eta, blep_phi, fj.eta, fj.phi)
        probe_ok = ((fj.pt > PROBE_PT_MIN) & (abs(fj.eta) < PROBE_ETA_MAX)
                    & (fj.msoftdrop > PROBE_MSD_MIN) & (fj.msoftdrop < PROBE_MSD_MAX)
                    & (dphi_mu_fj > DPHI_LEP_PROBE_MIN)
                    & (dr_fj_blep > DR_PROBE_BLEP_MIN))
        probes = fj[probe_ok]
        n_probe = ak.to_numpy(ak.num(probes, axis=1))

        final = (n_lepb >= 1) & (n_probe >= 1)
        output['cutflow'][f'{dataset}{suffix}/final'] += int(np.sum(final))
        output['nevents'][dataset + suffix] += int(np.sum(final))
        if not np.any(final):
            return

        # ---- gather the single probe (leading passing AK8) per final event ---
        ev = ev[final]
        w_evt = w_evt[final]
        probe = ak.firsts(probes[final])
        probe_eta = probe.eta                          # ak flat, kept for merge dR
        probe_phi = probe.phi
        probe_pt_np = ak.to_numpy(probe.pt).astype(np.float64)
        probe_eta_np = ak.to_numpy(probe_eta)
        probe_phi_np = ak.to_numpy(probe_phi)
        probe_msd = ak.to_numpy(probe.msoftdrop).astype(np.float32)
        D = ak.to_numpy(_topvsqcd_globalParT3(probe)).astype(np.float64)

        # tag-side kinematics (flat-vs-flat -> numpy is safe)
        mu_pt_np = ak.to_numpy(lead_mu.pt[final])
        mu_eta_np = ak.to_numpy(mu_eta[final])
        mu_phi_np = ak.to_numpy(mu_phi[final])
        met_f = met_sel[final]
        dphi_mu_probe = np.abs(_delta_phi(mu_phi_np, probe_phi_np))
        blep_eta_f = ak.to_numpy(blep_eta[final])
        blep_phi_f = ak.to_numpy(blep_phi[final])
        dr_probe_blep = _delta_r_flat_jagged(
            probe_eta_np, probe_phi_np, blep_eta_f, blep_phi_f)
        n_bjet = n_lepb[final]

        # weight (raw; xsec*lumi/sumw applied at save time)
        gw = ak.to_numpy(w_evt).astype(np.float64)
        pu_w = np.ones(len(gw), dtype=np.float64)
        if is_mc and self.apply_pu and 'Pileup' in ev.fields:
            try:
                from corrections import GetPUSF
                pu_w = np.asarray(GetPUSF(ev, self.iov)[0], dtype=np.float64)
                gw = gw * pu_w
            except Exception as exc:
                # Do NOT swallow this silently: a failed PU lookup used to leave
                # the run looking healthy while applying no pileup weight at all,
                # which is indistinguishable from success in the output. Shout,
                # and record pu_weight=1 so the ntuple itself shows it was off.
                print(f"[toptag_sf] WARNING: pileup reweighting FAILED for "
                      f"{ev.metadata.get('dataset', '?')} ({type(exc).__name__}: "
                      f"{exc}) -- running with NO pileup weight", flush=True)
                pu_w = np.ones(len(gw), dtype=np.float64)

        if is_mc:
            merge_cat, b_merged, n_wq_merged = self._probe_merge_cat(
                ev, probe_eta, probe_phi)
        else:
            merge_cat = np.full(len(D), MC_DATA, dtype=np.int8)
            b_merged = np.full(len(D), -1, dtype=np.int8)
            n_wq_merged = np.full(len(D), -1, dtype=np.int8)

        cols = {
            'probe_pt': probe_pt_np.astype(np.float32),
            'probe_eta': probe_eta_np.astype(np.float32),
            'probe_msd': probe_msd,
            'D': D.astype(np.float32),
            'merge_cat': merge_cat.astype(np.int8),
            'b_merged': b_merged.astype(np.int8),
            'n_wq_merged': n_wq_merged.astype(np.int8),
            'lepton_channel': np.full(len(D), CH_MU, dtype=np.int8),
            'mu_pt': mu_pt_np.astype(np.float32),
            'mu_eta': mu_eta_np.astype(np.float32),
            'met': met_f.astype(np.float32),
            'dphi_mu_probe': dphi_mu_probe.astype(np.float32),
            'dr_probe_blep': dr_probe_blep.astype(np.float32),
            'n_bjet': n_bjet.astype(np.int8),
            'genweight': gw.astype(np.float32),
            # 1.0 when pileup reweighting was not applied -- lets the
            # ntuple prove whether --apply-pu actually took effect.
            'pu_weight': pu_w.astype(np.float32),
            # corrected/uncorrected probe pT; 1.0 when JEC/JER off.
            'jec_factor': ak.to_numpy(probe.jec_factor).astype(np.float32),
        }
        cols.update({k: v for k, v in self._passwp_columns(
            probe_pt_np, D).items()})

        dest = output['ntuple'].setdefault(
            dataset + suffix,
            {f: column_accumulator(np.empty(0, dtype=dt))
             for f, dt in SF_NTUPLE_FIELDS.items()})
        for f in SF_NTUPLE_FIELDS:
            dest[f] = dest[f] + column_accumulator(
                np.asarray(cols[f], dtype=SF_NTUPLE_FIELDS[f]))

        return

    def postprocess(self, accumulator):
        return accumulator
