import awkward as ak
import numpy as np
import os
import subprocess
import sys
import unittest

from coffea.analysis_tools import Weights

sys.path.append(os.path.join(os.getcwd(), "python"))
import weights
from truthstudy import MERGE_FULL, MERGE_NOT, MERGE_SEMI, MERGE_W, top_merge_category
from truthstudy import (WLEP_E, WLEP_MU, WLEP_NONE, WLEP_TAUE, WLEP_TAUH, WLEP_TAUMU,
                        w_lepton_in_jet)
from weights import Run3WeightManager, load_ttag_sf


def _jets(pts):
    return ak.Array({"p4": {"pt": np.asarray(pts, dtype=np.float64)}})


# data/toptag/ttag_sf_v1.2.json, 2024 tight, pT bin 3 (>= 600 GeV): tagged jet, antitag band
TIGHT3 = (0.821, 0.8956, 0.7486)
BAND3 = (1.578, 1.7657, 1.3895)


class TopTagSFWeightsTest(unittest.TestCase):
    def _variations(self, iov, jet0_pt, jet1_pt, ttag2, antitag, merged=None):
        manager = Run3WeightManager(iov=iov, systematics=["ttag_pt3"], no_syst=False, deepak8_cut="tight")
        weights = Weights(len(jet0_pt))
        if merged is None:
            merged = (np.ones(len(jet0_pt), bool), np.ones(len(jet0_pt), bool))
        manager._add_ttag_pt_weights(
            weights, _jets(jet0_pt), _jets(jet1_pt), np.asarray(ttag2), np.asarray(antitag), merged
        )
        return weights.weight(), weights.weight("ttag_pt3Up"), weights.weight("ttag_pt3Down")

    def test_uncertainty_doubled_only_above_measured_range(self):
        # event 0: both jets inside the measured range; event 1: both beyond 1.2 TeV
        nom, up, down = self._variations("2024", [700.0, 1500.0], [800.0, 1600.0], [True, True], [False, False])
        sf, sf_up, sf_down = TIGHT3
        np.testing.assert_allclose(nom, [sf**2, sf**2])
        np.testing.assert_allclose(up, [sf_up**2, (sf + 2 * (sf_up - sf)) ** 2])
        np.testing.assert_allclose(down, [sf_down**2, (sf - 2 * (sf - sf_down)) ** 2])

    def test_antitag_jet_uses_band_sf_with_same_extrapolation(self):
        nom, up, _ = self._variations("2025", [700.0], [1500.0], [False], [True])
        np.testing.assert_allclose(nom, [TIGHT3[0] * BAND3[0]])
        np.testing.assert_allclose(up, [TIGHT3[1] * (BAND3[0] + 2 * (BAND3[1] - BAND3[0]))])

    def test_sf_only_on_fully_merged_jets(self):
        # 2024 SFs were measured for fully merged tops: other jets keep SF 1
        # event 0: tagged jet0 merged, antitag jet1 not; event 1: the reverse; event 2: 2t, jet1 not merged
        nom, up, _ = self._variations(
            "2024", [700.0, 700.0, 700.0], [700.0, 700.0, 700.0],
            [False, False, True], [True, True, False],
            merged=(np.array([True, False, True]), np.array([False, True, False])),
        )
        np.testing.assert_allclose(nom, [TIGHT3[0], BAND3[0], TIGHT3[0]])
        np.testing.assert_allclose(up, [TIGHT3[1], BAND3[1], TIGHT3[1]])

    def test_merge_flags_required_for_fully_merged_tables(self):
        manager = Run3WeightManager(iov="2025", systematics=["ttag_pt1"], no_syst=False, deepak8_cut="tight")
        with self.assertRaisesRegex(ValueError, "fully merged"):
            manager._add_ttag_pt_weights(Weights(1), _jets([450.0]), _jets([450.0]),
                                         np.array([True]), np.array([False]))

    def test_run2_tables_ignore_merge_flags(self):
        _, up, _ = self._variations("2018", [1500.0], [1500.0], [True], [False],
                                    merged=(np.array([False]), np.array([False])))
        np.testing.assert_allclose(up, [1.01**2])

    def _nonmerged(self, iov, ttag2, antitag, merged, pts=(700.0, 700.0)):
        manager = Run3WeightManager(iov=iov, systematics=["ttag_nonmerged"], no_syst=False, deepak8_cut="tight")
        n = len(ttag2)
        weights = Weights(n)
        manager._add_ttag_nonmerged_weights(
            weights, _jets([pts[0]] * n), _jets([pts[1]] * n), np.asarray(ttag2), np.asarray(antitag), merged
        )
        return weights.weight(), weights.weight("ttag_nonmergedUp"), weights.weight("ttag_nonmergedDown")

    def test_nonmerged_jets_get_flat_uncertainty(self):
        # 0: 2t, jet1 not merged; 1: antitag, both not merged; 2: jet1 neither tagged nor
        # antitag (only jet0 counts); 3: all merged
        nom, up, down = self._nonmerged(
            "2024", [True, False, False, True], [False, True, False, False],
            (np.array([True, False, False, True]), np.array([False, False, False, True])),
        )
        u = load_ttag_sf()["iovs"]["2024"]["nonmerged_unc"]
        np.testing.assert_allclose(nom, 1.0)
        np.testing.assert_allclose(up, [1 + u, (1 + u) ** 2, 1 + u, 1.0])
        np.testing.assert_allclose(down, [1 - u, (1 - u) ** 2, 1 - u, 1.0])

    def test_nonmerged_uncertainty_only_in_sf_pt_range(self):
        _, up, _ = self._nonmerged("2024", [True], [False], (np.array([False]), np.array([False])),
                                   pts=(380.0, 700.0))
        np.testing.assert_allclose(up, [1 + load_ttag_sf()["iovs"]["2024"]["nonmerged_unc"]])

    def test_run2_tables_have_no_nonmerged_uncertainty(self):
        _, up, down = self._nonmerged("2018", [True], [False], None)
        np.testing.assert_allclose([up[0], down[0]], [1.0, 1.0])

    def test_unmeasured_wp_raises(self):
        manager = Run3WeightManager(iov="2024", systematics=["ttag_pt1"], no_syst=False, deepak8_cut="medium")
        with self.assertRaisesRegex(KeyError, "no 'medium' WP for 2024"):
            manager._add_ttag_pt_weights(Weights(1), _jets([450.0]), _jets([450.0]),
                                         np.array([True]), np.array([False]))

    def test_v11_file_reproduces_the_old_hard_coded_table(self):
        src = subprocess.run(["git", "show", "047db9f:python/weights.py"],
                             capture_output=True, text=True, check=True).stdout
        start = src.index("ttag_scale_factors = {") + len("ttag_scale_factors = ")
        old = eval(src[start:src.index("\n        }\n", start) + len("\n        }")])
        v11 = load_ttag_sf(weights.TTAG_SF_FILE.with_name("ttag_sf_v1.1.json"))
        for iov, table in old.items():
            for wp, sf in table.items():
                self.assertEqual(v11["iovs"][iov]["wps"][wp]["tag"], sf)
                self.assertEqual(v11["iovs"][iov]["wps"][wp]["antitag"], table["loose"])

    def test_run2_keeps_original_uncertainty(self):
        # 2018 tight, pT bin 3: nominal 0.93, up 1.01
        _, up, _ = self._variations("2018", [1500.0], [1500.0], [True], [False])
        np.testing.assert_allclose(up, [1.01**2])


def _genparts(events):
    """events: list of [(pdgId, mother index, eta, phi), ...]"""
    return ak.Array([
        [{"pdgId": p, "genPartIdxMother": m, "eta": eta, "phi": phi} for p, m, eta, phi in ev]
        for ev in events
    ])


class TopMergeCategoryTest(unittest.TestCase):
    # t (0) -> b (1) + W (2) -> q (3) q' (4); the jet axis is at eta = phi = 0
    def _event(self, b, q1, q2):
        return [(6, -1, 0.0, 0.0), (5, 0, *b), (24, 0, 0.0, 0.0), (1, 2, *q1), (-2, 2, *q2)]

    def test_categories_match_the_tp_definition(self):
        inside, outside = (0.3, 0.3), (1.5, 0.0)
        gp = _genparts([
            self._event(inside, inside, (0.1, -0.5)),   # all three in
            self._event(outside, inside, inside),       # W only
            self._event(inside, inside, outside),       # b + one W quark
            self._event(inside, outside, outside),      # b only
            [(21, -1, 0.0, 0.0), (1, 0, 0.0, 0.0)],     # QCD: gluon -> quark, no top
        ])
        cat = top_merge_category(gp, np.zeros(5), np.zeros(5))
        np.testing.assert_array_equal(cat, [MERGE_FULL, MERGE_W, MERGE_SEMI, MERGE_NOT, MERGE_NOT])

    def test_phi_wraps_around(self):
        near_pi = (0.0, np.pi - 0.1)
        gp = _genparts([self._event(near_pi, near_pi, near_pi)])
        cat = top_merge_category(gp, np.zeros(1), np.array([-np.pi + 0.1]))
        np.testing.assert_array_equal(cat, [MERGE_FULL])



class WLeptonInJetTest(unittest.TestCase):
    # t (0) -> b (1) + W (2) -> lepton (3) + neutrino (4); jet axis at eta = phi = 0
    def _event(self, lep_pdg, lep, extra=()):
        return [(6, -1, 0.0, 0.0), (5, 0, 0.1, 0.1), (24, 0, 0.0, 0.0),
                (lep_pdg, 2, *lep), (-(abs(lep_pdg) + 1), 2, 0.2, 0.2), *extra]

    def test_flavour_codes(self):
        inside, outside = (0.3, 0.3), (1.5, 0.0)
        gp = _genparts([
            self._event(11, inside),
            self._event(-13, inside),
            self._event(15, inside),                              # tau -> hadrons
            self._event(15, inside, [(11, 3, 0.35, 0.3)]),        # tau -> e
            self._event(-15, inside, [(-13, 3, 0.35, 0.3)]),      # tau -> mu
            self._event(13, outside),                             # lepton outside the jet
            [(6, -1, 0.0, 0.0), (5, 0, 0.1, 0.1), (24, 0, 0.0, 0.0),
             (1, 2, 0.2, 0.2), (-2, 2, -0.2, 0.1)],               # hadronic top
            [(21, -1, 0.0, 0.0), (13, 0, 0.1, 0.1)],              # QCD muon, not from a W
        ])
        code = w_lepton_in_jet(gp, np.zeros(8), np.zeros(8))
        np.testing.assert_array_equal(code, [WLEP_E, WLEP_MU, WLEP_TAUH, WLEP_TAUE, WLEP_TAUMU,
                                             WLEP_NONE, WLEP_NONE, WLEP_NONE])

    def test_tau_label_follows_its_own_decay(self):
        inside, outside = (0.3, 0.3), (1.5, 0.0)
        gp = _genparts([
            # tau -> e with the electron outside the jet: still tau -> e
            self._event(15, inside, [(11, 3, *outside)]),
            # tau copy chain: tau (3) -> tau (5) -> mu
            self._event(15, inside, [(15, 3, 0.31, 0.3), (13, 5, *outside)]),
            # hadronic tau in the jet; an electron from a second, distant tau lands inside
            self._event(15, inside, [(24, 0, 2.0, 2.0), (15, 5, 2.0, 2.0), (11, 6, 0.3, 0.35)]),
        ])
        code = w_lepton_in_jet(gp, np.zeros(3), np.zeros(3))
        np.testing.assert_array_equal(code, [WLEP_TAUE, WLEP_TAUMU, WLEP_TAUH])



class LeptonicTopFractionTest(unittest.TestCase):
    def test_flep(self):
        from categories import leptonic_top_fraction
        jets = ak.Array([
            # hadronic top, leptonic e top, leptonic tau top, all heads zero
            {"globalParT3_TopbWqq": 0.80, "globalParT3_TopbWq": 0.06, "globalParT3_TopbWev": 0.0004,
             "globalParT3_TopbWmv": 0.0, "globalParT3_TopbWtauhv": 0.0, "globalParT3_QCD": 0.006},
            {"globalParT3_TopbWqq": 0.0004, "globalParT3_TopbWq": 0.0002, "globalParT3_TopbWev": 0.94,
             "globalParT3_TopbWmv": 0.0, "globalParT3_TopbWtauhv": 0.0, "globalParT3_QCD": 0.00002},
            {"globalParT3_TopbWqq": 0.02, "globalParT3_TopbWq": 0.005, "globalParT3_TopbWev": 0.0,
             "globalParT3_TopbWmv": 0.0, "globalParT3_TopbWtauhv": 0.83, "globalParT3_QCD": 0.0006},
            {"globalParT3_TopbWqq": 0.0, "globalParT3_TopbWq": 0.0, "globalParT3_TopbWev": 0.0,
             "globalParT3_TopbWmv": 0.0, "globalParT3_TopbWtauhv": 0.0, "globalParT3_QCD": 1.0},
        ])
        f = np.asarray(leptonic_top_fraction(jets))
        np.testing.assert_allclose(f, [0.0004 / 0.8604, 0.94 / 0.9406, 0.83 / 0.855, 0.0], rtol=1e-6)
        # the hadronic top passes c = 0.59, the leptonic ones do not
        np.testing.assert_array_equal(f < 0.59, [True, False, False, True])


if __name__ == "__main__":
    unittest.main()
