"""Synthetic unit test for TopTagSFProcessor.

Builds mock NanoAOD-like awkward arrays (the processor uses only field access and
its own manual delta-R helpers -- no coffea vector behavior -- so plain record
arrays are enough) and exercises the tag/probe selection + merge-category split.

Run:  coffea-dask/bin/python -m pytest tests/test_toptag_sf_processor.py -q
 or:  coffea-dask/bin/python tests/test_toptag_sf_processor.py
"""

import os
import sys
import math
import unittest

import numpy as np
import awkward as ak

sys.path.append(os.path.join(os.getcwd(), "python"))
import toptag_sf_processor as m  # noqa: E402

PI = math.pi
MET_FLAGS = [
    "goodVertices", "globalSuperTightHalo2016Filter",
    "EcalDeadCellTriggerPrimitiveFilter", "BadPFMuonFilter",
    "BadPFMuonDzFilter", "hfNoisyHitsFilter", "eeBadScFilter",
    "ecalBadCalibFilter",
]


class MockEvents:
    """Minimal stand-in for a coffea NanoEventsArray: an ak.Array plus metadata.

    Only the *original* events object needs ``.metadata``; ``events[mask]`` returns
    the bare ak.Array, and the processor only does field access after that.
    """

    def __init__(self, arr, metadata):
        object.__setattr__(self, "_a", arr)
        object.__setattr__(self, "metadata", metadata)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_a"), k)

    def __getitem__(self, k):
        return object.__getattribute__(self, "_a")[k]

    def __len__(self):
        return len(object.__getattribute__(self, "_a"))


def _mu(pt, eta, phi, tight=True, iso=0.02):
    return {"pt": pt, "eta": eta, "phi": phi, "tightId": tight,
            "pfRelIso04_all": iso}


def _jet(pt, eta, phi, btag, jetId=6):
    return {"pt": pt, "eta": eta, "phi": phi, "jetId": jetId,
            "btagDeepFlavB": btag}


def _fj(pt, eta, phi, msd, wqq, wq, qcd):
    return {"pt": pt, "eta": eta, "phi": phi, "msoftdrop": msd,
            "globalParT3_TopbWqq": wqq, "globalParT3_TopbWq": wq,
            "globalParT3_QCD": qcd}


def _gp(pdg, flags, mother, pt, eta, phi):
    return {"pdgId": pdg, "statusFlags": flags, "genPartIdxMother": mother,
            "pt": pt, "eta": eta, "phi": phi, "mass": 0.0}


LAST = 1 << 13  # isLastCopy


def _hadronic_top_genparts(probe_eta, probe_phi, b_near=True, n_wq_near=2):
    """One hadronic top with independently placed b and W quarks.

    Chain modeled as top -> W_first -> W_last -> q q', plus b straight from top,
    matching get_hadronic_tops' tracing. ``b_near`` puts the b inside dR<0.8 of
    the probe; ``n_wq_near`` (0-2) of the two W quarks are placed inside.
    """
    near = (probe_eta + 0.1, probe_phi + 0.1)   # dR ~ 0.14 < 0.8
    far = (probe_eta + 2.0, probe_phi + 2.0)
    be, bp = near if b_near else far
    (q1e, q1p) = near if n_wq_near >= 1 else far
    (q2e, q2p) = near if n_wq_near >= 2 else far
    return [
        _gp(6, LAST, -1, 500.0, probe_eta, probe_phi),   # 0 top (last copy)
        _gp(24, 0, 0, 200.0, probe_eta, probe_phi),      # 1 W first copy
        _gp(5, LAST, 0, 120.0, be, bp),                  # 2 b from top
        _gp(24, LAST, 1, 190.0, probe_eta, probe_phi),   # 3 W last copy
        _gp(1, LAST, 3, 90.0, q1e, q1p),                 # 4 q from W
        _gp(2, LAST, 3, 80.0, q2e, q2p),                 # 5 q' from W
    ]


GP_FIELDS = ("pdgId", "statusFlags", "genPartIdxMother", "pt", "eta", "phi", "mass")


def _genpart_col(events_records):
    """GenPart column that is always *typed*, like the real NanoAOD branch.

    ``ak.Array([[], []])`` has no fields at all, so ``genparts.pdgId`` raises --
    an artifact of the mock, not of real files, where NanoAODSchema always
    supplies the full field set even for a zero-length collection. Build the
    record type explicitly when every event is empty so tests exercise the
    realistic degenerate case (typed but empty) instead of a fieldless array.
    """
    lists = [r.get("GenPart", []) for r in events_records]
    if not any(lists):
        return ak.zip({f: ak.Array([[] for _ in lists]) for f in GP_FIELDS})
    return ak.Array(lists)


def _build(events_records, is_mc=True):
    n = len(events_records)
    cols = {
        "genWeight": ak.Array([r.get("genWeight", 1.0) for r in events_records]),
        "Muon": ak.Array([r["Muon"] for r in events_records]),
        "Electron": ak.Array([r.get("Electron", []) for r in events_records]),
        "Jet": ak.Array([r["Jet"] for r in events_records]),
        "FatJet": ak.Array([r["FatJet"] for r in events_records]),
        "HLT": ak.Array([r.get("HLT", {"Mu50": True, "IsoMu24": True})
                         for r in events_records]),
        "Flag": ak.Array([{f: True for f in MET_FLAGS} for _ in events_records]),
        "PuppiMET": ak.Array([{"pt": r["met"]} for r in events_records]),
    }
    if is_mc:
        cols["GenPart"] = _genpart_col(events_records)
    arr = ak.zip(cols, depth_limit=1)
    meta = {"dataset": "TTto2L2Nu" if is_mc else "Data_C", "is_mc": is_mc}
    return MockEvents(arr, meta)


def _run(events, is_mc=True):
    proc = m.TopTagSFProcessor(iov="2024", channel="mu",
                               apply_pu=False, apply_lumimask=False)
    out = proc.process(events)
    return out


class TopTagSFSelectionTest(unittest.TestCase):
    def _probe_recoil(self, msd=170.0, wqq=0.99, wq=0.0, qcd=0.01, pt=450.0):
        # AK8 back-to-back with a muon at (eta=0, phi=0)
        return _fj(pt, 0.0, PI, msd, wqq, wq, qcd)

    def _base_event(self, **kw):
        ev = {
            "Muon": [_mu(200.0, 0.0, 0.0)],
            "Jet": [_jet(70.0, 0.1, 0.3, btag=0.9)],   # b near muon (dR~0.32)
            "FatJet": [self._probe_recoil()],
            "met": 120.0,
        }
        ev.update(kw)
        return ev

    def test_fully_merged_signal_passes_and_tags(self):
        ev = self._base_event(
            GenPart=_hadronic_top_genparts(0.0, PI, b_near=True, n_wq_near=2))
        out = _run(_build([ev]))
        nt = out["ntuple"]["TTto2L2Nu"]
        self.assertEqual(len(nt["D"].value), 1)
        self.assertEqual(int(nt["merge_cat"].value[0]), m.MC_FULL)
        self.assertEqual(int(nt["b_merged"].value[0]), 1)
        self.assertEqual(int(nt["n_wq_merged"].value[0]), 2)
        self.assertAlmostEqual(float(nt["D"].value[0]), 0.99, places=4)
        # D=0.99 clears every WP threshold in the 400-500 bin
        for wp in m.WP_NAMES:
            self.assertEqual(int(nt[f"pass_{wp}"].value[0]), 1, wp)
        self.assertEqual(int(nt["lepton_channel"].value[0]), m.CH_MU)

    def test_fully_merged_w_only(self):
        # both W quarks in, b outside the cone -> fully merged W
        ev = self._base_event(
            GenPart=_hadronic_top_genparts(0.0, PI, b_near=False, n_wq_near=2))
        out = _run(_build([ev]))
        nt = out["ntuple"]["TTto2L2Nu"]
        self.assertEqual(int(nt["merge_cat"].value[0]), m.MC_W)
        self.assertEqual(int(nt["b_merged"].value[0]), 0)

    def test_semi_merged(self):
        # b + one W quark in -> semi merged
        ev = self._base_event(
            GenPart=_hadronic_top_genparts(0.0, PI, b_near=True, n_wq_near=1))
        out = _run(_build([ev]))
        nt = out["ntuple"]["TTto2L2Nu"]
        self.assertEqual(int(nt["merge_cat"].value[0]), m.MC_SEMI)

    def test_not_merged(self):
        # only the b in the cone (one quark) -> not merged
        ev = self._base_event(
            GenPart=_hadronic_top_genparts(0.0, PI, b_near=True, n_wq_near=0))
        out = _run(_build([ev]))
        nt = out["ntuple"]["TTto2L2Nu"]
        self.assertEqual(int(nt["merge_cat"].value[0]), m.MC_NOT)

    def test_low_score_probe_fails_tight_wp(self):
        ev = self._base_event(
            FatJet=[self._probe_recoil(wqq=0.30, wq=0.0, qcd=0.70)],  # D=0.30
            GenPart=_hadronic_top_genparts(0.0, PI, b_near=True, n_wq_near=2))
        out = _run(_build([ev]))
        nt = out["ntuple"]["TTto2L2Nu"]
        self.assertAlmostEqual(float(nt["D"].value[0]), 0.30, places=4)
        self.assertEqual(int(nt["pass_tight"].value[0]), 0)
        self.assertEqual(int(nt["pass_very_loose"].value[0]), 0)  # thr 0.576

    def test_no_leptonic_bjet_rejects_event(self):
        ev = self._base_event(Jet=[_jet(70.0, 0.1, 0.3, btag=0.01)])  # not b-tagged
        out = _run(_build([ev]))
        self.assertNotIn("TTto2L2Nu", out["ntuple"])
        self.assertEqual(out["nevents"]["TTto2L2Nu"], 0)

    def test_low_met_rejects_event(self):
        ev = self._base_event(met=20.0)
        out = _run(_build([ev]))
        self.assertNotIn("TTto2L2Nu", out["ntuple"])

    def test_probe_collinear_with_lepton_rejected(self):
        # AK8 at phi=0 (same side as muon) -> dphi ~ 0 < 2.0
        ev = self._base_event(FatJet=[_fj(450.0, 0.0, 0.05, 170.0, 0.99, 0.0, 0.01)])
        out = _run(_build([ev]))
        self.assertNotIn("TTto2L2Nu", out["ntuple"])

    def test_msd_window(self):
        ev = self._base_event(FatJet=[self._probe_recoil(msd=300.0)])  # > 210
        out = _run(_build([ev]))
        self.assertNotIn("TTto2L2Nu", out["ntuple"])

    def test_two_tight_muons_rejected(self):
        ev = self._base_event(Muon=[_mu(200.0, 0.0, 0.0), _mu(90.0, 0.5, 1.0)])
        out = _run(_build([ev]))
        self.assertNotIn("TTto2L2Nu", out["ntuple"])

    def test_data_merge_cat_is_none(self):
        ev = self._base_event()
        out = _run(_build([ev], is_mc=False), is_mc=False)
        nt = out["ntuple"]["Data_C"]
        self.assertEqual(int(nt["merge_cat"].value[0]), m.MC_DATA)
        self.assertEqual(float(nt["genweight"].value[0]), 1.0)

    def test_multi_event_bookkeeping(self):
        evs = [
            self._base_event(GenPart=_hadronic_top_genparts(0.0, PI, True, 2)),   # pass
            self._base_event(met=10.0),                                           # fail MET
            self._base_event(GenPart=_hadronic_top_genparts(0.0, PI, True, 1)),   # pass
        ]
        out = _run(_build(evs))
        self.assertEqual(out["nevents_raw"]["TTto2L2Nu"], 3)
        self.assertEqual(out["nevents"]["TTto2L2Nu"], 2)
        self.assertEqual(len(out["ntuple"]["TTto2L2Nu"]["D"].value), 2)


def _wjets_genparts(probe_eta, probe_phi, quark_mother=-1):
    """W+jets-like gen record: a *leptonic* W and light partons, and no top.

    This is the topology of the W+jets background samples added for the P0
    background subtraction (WJets.json). Nothing here should register as a top
    or W decay quark: the W decays to mu nu, and the recoiling light parton is a
    hard-process/ISR parton, not a W daughter. ``quark_mother`` lets a test point
    that parton's mother at the -1 sentinel (index 0 in NanoAOD terms: no mother)
    or at a gluon, both of which must stay unmatched.
    """
    near = (probe_eta + 0.1, probe_phi + 0.1)   # dR ~ 0.14 -- inside the cone
    return [
        _gp(24, LAST, -1, 250.0, 0.0, 0.0),          # 0 leptonic W (near the muon)
        _gp(13, LAST, 0, 200.0, 0.0, 0.0),           # 1 muon from the W
        _gp(14, LAST, 0, 120.0, 0.05, 0.05),         # 2 neutrino from the W
        _gp(21, LAST, -1, 80.0, 1.5, 1.0),           # 3 gluon
        _gp(1, LAST, quark_mother, 400.0, *near),    # 4 light parton -> the probe
    ]


class WJetsBackgroundGenTest(unittest.TestCase):
    """The gen path on samples with no top quark at all.

    The processor and its merge classification were only ever exercised on ttbar
    and data; W+jets is a new code path (is_mc=True, zero tops, zero hadronic W
    quarks), so every gen reduction runs over a fully empty jagged collection.
    """

    def _probe_recoil(self):
        return _fj(450.0, 0.0, PI, 170.0, 0.99, 0.0, 0.01)

    def _base_event(self, **kw):
        ev = {
            "Muon": [_mu(200.0, 0.0, 0.0)],
            "Jet": [_jet(70.0, 0.1, 0.3, btag=0.9)],
            "FatJet": [self._probe_recoil()],
            "met": 120.0,
        }
        ev.update(kw)
        return ev

    def test_wjets_probe_is_not_merged(self):
        # A hard parton inside the cone with NO W/top mother must not be counted.
        # This is also the regression guard for the beam-parton bug: the parton's
        # genPartIdxMother is the -1 sentinel.
        ev = self._base_event(GenPart=_wjets_genparts(0.0, PI, quark_mother=-1))
        out = _run(_build([ev]))
        nt = out["ntuple"]["TTto2L2Nu"]
        self.assertEqual(len(nt["D"].value), 1)          # probe still recorded
        self.assertEqual(int(nt["merge_cat"].value[0]), m.MC_NOT)
        self.assertEqual(int(nt["b_merged"].value[0]), 0)
        self.assertEqual(int(nt["n_wq_merged"].value[0]), 0)

    def test_wjets_gluon_mothered_parton_not_merged(self):
        # Same, but the parton's mother is a real particle (the gluon at index 3).
        # Only mom_pdg == 24 may count, so this must stay unmatched.
        ev = self._base_event(GenPart=_wjets_genparts(0.0, PI, quark_mother=3))
        out = _run(_build([ev]))
        nt = out["ntuple"]["TTto2L2Nu"]
        self.assertEqual(int(nt["merge_cat"].value[0]), m.MC_NOT)
        self.assertEqual(int(nt["n_wq_merged"].value[0]), 0)

    def test_empty_genpart_does_not_crash(self):
        # Degenerate case: MC event with an empty GenPart collection. Every gen
        # reduction runs over a zero-length jagged array.
        ev = self._base_event(GenPart=[])
        out = _run(_build([ev]))
        nt = out["ntuple"]["TTto2L2Nu"]
        self.assertEqual(len(nt["D"].value), 1)
        self.assertEqual(int(nt["merge_cat"].value[0]), m.MC_NOT)

    def test_mixed_wjets_and_ttbar_events(self):
        # W+jets-like and ttbar-like events in one chunk: the empty-gen events
        # must not disturb the classification of the real top event.
        evs = [
            self._base_event(GenPart=_wjets_genparts(0.0, PI)),
            self._base_event(GenPart=_hadronic_top_genparts(0.0, PI, True, 2)),
            self._base_event(GenPart=[]),
        ]
        out = _run(_build(evs))
        nt = out["ntuple"]["TTto2L2Nu"]
        cats = [int(c) for c in nt["merge_cat"].value]
        self.assertEqual(cats, [m.MC_NOT, m.MC_FULL, m.MC_NOT])


if __name__ == "__main__":
    unittest.main(verbosity=2)
