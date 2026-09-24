import awkward as ak
import numpy as np
import os
import subprocess
import sys
import unittest

from coffea.analysis_tools import Weights

sys.path.append(os.path.join(os.getcwd(), "python"))
import weights
from weights import Run3WeightManager, load_ttag_sf


def _jets(pts):
    return ak.Array({"p4": {"pt": np.asarray(pts, dtype=np.float64)}})


# data/toptag/ttag_sf_v1.2.json, 2024 tight, pT bin 3 (>= 600 GeV): tagged jet, antitag band
TIGHT3 = (0.821, 0.8956, 0.7486)
BAND3 = (1.578, 1.7657, 1.3895)


class TopTagSFWeightsTest(unittest.TestCase):
    def _variations(self, iov, jet0_pt, jet1_pt, ttag2, antitag):
        manager = Run3WeightManager(iov=iov, systematics=["ttag_pt3"], no_syst=False, deepak8_cut="tight")
        weights = Weights(len(jet0_pt))
        manager._add_ttag_pt_weights(
            weights, _jets(jet0_pt), _jets(jet1_pt), np.asarray(ttag2), np.asarray(antitag)
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


if __name__ == "__main__":
    unittest.main()
