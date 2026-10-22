import os
import sys
import unittest

import awkward as ak
import numpy as np
from coffea.analysis_tools import Weights

sys.path.append(os.path.join(os.getcwd(), "python"))
import weights
from corrections import GetTopPtResidual, top_pt_residual
from weights import EXTRAP_NUISANCES, TTAG_EIG_NUISANCES, TTAG_PT_NUISANCES, Run3WeightManager, load_ttag_sf

V13 = os.path.join(os.getcwd(), "data", "toptag", "ttag_sf_v1.3.json")
V12 = os.path.join(os.getcwd(), "data", "toptag", "ttag_sf_v1.2.json")
ALL = list(TTAG_PT_NUISANCES + TTAG_EIG_NUISANCES + tuple(EXTRAP_NUISANCES.values()))


def _jets(pts):
    return ak.Array({"p4": {"pt": np.asarray(pts, dtype=np.float64)}})


def _weights(jet0_pt, jet1_pt, ttag2, antitag, systematics=ALL, sf_file=V13, iov="2024"):
    manager = Run3WeightManager(iov=iov, systematics=systematics, no_syst=False, deepak8_cut="tight",
                                ttag_sf_file=sf_file)
    w = Weights(len(jet0_pt))
    merged = (np.ones(len(jet0_pt), bool), np.ones(len(jet0_pt), bool))
    manager._add_ttag_pt_weights(w, _jets(jet0_pt), _jets(jet1_pt), np.asarray(ttag2), np.asarray(antitag), merged)
    return w


class V13TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = load_ttag_sf(V13)["iovs"]["2024"]
        cls.tag = [np.asarray(cls.t["wps"]["tight"]["tag"][v]) for v in ("nominal", "up", "down")]
        cls.band = [np.asarray(cls.t["wps"]["tight"]["antitag"][v]) for v in ("nominal", "up", "down")]

    def test_default_table_is_v13(self):
        self.assertTrue(str(weights.TTAG_SF_FILE).endswith("ttag_sf_v1.3.json"))
        self.assertEqual(self.t["pt_edges"], [0.0, 400.0, 480.0, 600.0, 800.0])
        self.assertNotIn("antitag_hipt", self.t)
        self.assertNotIn("flep", self.t)

    def test_fourth_bin_and_binned_band_sf(self):
        # jet0 900 GeV tagged (bin 4), jet1 700 GeV band (bin 3): no interim hipt SF any more
        w = _weights([900.0], [700.0], [False], [True])
        np.testing.assert_allclose(w.weight(), [self.tag[0][4] * self.band[0][3]])
        np.testing.assert_allclose(w.weight("ttag_pt4Up"), [self.tag[1][4] * self.band[0][3]])
        np.testing.assert_allclose(w.weight("ttag_pt3Down"), [self.tag[0][4] * self.band[2][3]])
        self.assertNotIn("ttag_band_hiptUp", w.variations)

    def test_beyond_measured_range_not_doubled_extra_goes_to_extrap(self):
        # both jets tagged at 1.5 TeV: the last bin's deviation once in ttag_pt4, the rest in ttag_extrap_tag
        w = _weights([1500.0], [1500.0], [True], [False])
        n, u, d = (v[4] for v in self.tag)
        np.testing.assert_allclose(w.weight(), [n * n])
        np.testing.assert_allclose(w.weight("ttag_pt4Up"), [u * u])
        ex = self.t["extrapolation"]["tag"]
        sf_up = n + np.sqrt(ex["factor_up"] ** 2 - 1) * (u - n)
        sf_dn = max(n - np.sqrt(ex["factor_down"] ** 2 - 1) * (n - d), 0.0)
        np.testing.assert_allclose(w.weight("ttag_extrap_tagUp"), [sf_up * sf_up])
        np.testing.assert_allclose(w.weight("ttag_extrap_tagDown"), [sf_dn * sf_dn])
        # quadrature: the total down deviation of one jet is factor_down x the last-bin deviation
        np.testing.assert_allclose(np.hypot(n - d, n - sf_dn), ex["factor_down"] * (n - d))
        np.testing.assert_allclose(w.weight("ttag_extrap_bandUp"), w.weight())

    def test_extrap_band_only_on_band_jets_beyond_range(self):
        w = _weights([1500.0, 1000.0], [1300.0, 1100.0], [False, False], [True, True])
        ex = self.t["extrapolation"]["antitag"]
        n, u, d = (v[4] for v in self.band)
        ratio_up = (n + np.sqrt(ex["factor"] ** 2 - 1) * (u - n)) / n
        np.testing.assert_allclose(w.weight("ttag_extrap_bandUp") / w.weight(), [ratio_up, 1.0])
        # the tagged jet0 at 1.5 TeV moves ttag_extrap_tag, the 1 TeV one does not
        self.assertGreater(w.weight("ttag_extrap_tagUp")[0] / w.weight()[0], 1.0)
        np.testing.assert_allclose(w.weight("ttag_extrap_tagUp")[1], w.weight()[1])

    def test_eigen_shifts_reproduce_the_covariance_diagonal(self):
        # one jet per (block, bin): the quadrature sum of the eigen shifts of its SF = sqrt(cov_ii)
        cov = np.asarray(self.t["covariance"]["total"])
        pts = [440.0, 540.0, 700.0, 1000.0]
        for block, ttag2, antitag, offset in (("tag", [True], [False], 0), ("antitag", [False], [True], 4)):
            for ib, pt in enumerate(pts, start=1):
                # jet0 below the SF range (weight 1), so only jet1 carries the shift
                w = _weights([300.0], [pt], ttag2, antitag, systematics=list(TTAG_EIG_NUISANCES))
                nom = w.weight()[0]
                shifts = [w.weight(f"{name}Up")[0] - nom for name in TTAG_EIG_NUISANCES]
                np.testing.assert_allclose(np.sqrt(np.sum(np.square(shifts))), np.sqrt(cov[ib - 1 + offset, ib - 1 + offset]),
                                           rtol=2e-3, err_msg=f"{block} bin {ib}")

    def test_unrequested_nuisance_keeps_nominal_only(self):
        w = _weights([900.0], [700.0], [False], [True], systematics=["ttag_pt3"])
        self.assertEqual(set(w.variations), {"ttag_pt3Up", "ttag_pt3Down"})
        np.testing.assert_allclose(w.weight(), [self.tag[0][4] * self.band[0][3]])

    def test_v12_table_keeps_doubling_and_flat_new_nuisances(self):
        w = _weights([1500.0], [700.0], [True], [False], sf_file=V12)
        t12 = load_ttag_sf(V12)["iovs"]["2024"]["wps"]["tight"]["tag"]
        n, u = t12["nominal"][3], t12["up"][3]
        np.testing.assert_allclose(w.weight("ttag_pt3Up"), [(n + 2 * (u - n)) * u])
        for name in ("ttag_pt4", "ttag_eig1", "ttag_extrap_tag", "ttag_extrap_band"):
            np.testing.assert_allclose(w.weight(f"{name}Up"), w.weight())

    def test_aliased_iovs_use_the_2024_entry(self):
        for iov in ("2022preEE", "2023postBPix", "2025", "2026"):
            w = _weights([900.0], [700.0], [False], [True], iov=iov)
            np.testing.assert_allclose(w.weight(), [self.tag[0][4] * self.band[0][3]])


class TopPtResidualTest(unittest.TestCase):
    def test_residual_held_at_1tev(self):
        np.testing.assert_allclose(top_pt_residual([500.0, 1000.0, 2000.0]),
                                   np.exp(0.0416 - 0.0003 * np.array([500.0, 1000.0, 1000.0])))

    def test_event_weight_from_last_copy_tops(self):
        last = 1 << 13
        gp = ak.Array([
            {"pdgId": [6, -6, 6], "pt": [500.0, 1500.0, 900.0], "statusFlags": [last, last, 0]},
            {"pdgId": [21], "pt": [300.0], "statusFlags": [last]},
        ])
        r = GetTopPtResidual(gp)
        np.testing.assert_allclose(r, [np.sqrt(np.exp(0.0416 - 0.15) * np.exp(0.0416 - 0.3)), 1.0])


if __name__ == "__main__":
    unittest.main()
