import awkward as ak
import numpy as np
import os
import sys
import unittest

from coffea.analysis_tools import Weights

sys.path.append(os.path.join(os.getcwd(), "python"))
from weights import Run3WeightManager


def _jets(pts):
    return ak.Array({"p4": {"pt": np.asarray(pts, dtype=np.float64)}})


# 2024 tight / loose SF tables in weights.py, pT bin 3 (>= 600 GeV)
TIGHT3 = (0.880, 0.9516, 0.8084)
LOOSE3 = (0.947, 1.0316, 0.8624)


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

    def test_antitag_jet_uses_fail_sf_with_same_extrapolation(self):
        nom, up, _ = self._variations("2025", [700.0], [1500.0], [False], [True])
        np.testing.assert_allclose(nom, [TIGHT3[0] * LOOSE3[0]])
        np.testing.assert_allclose(up, [TIGHT3[1] * (LOOSE3[0] + 2 * (LOOSE3[1] - LOOSE3[0]))])

    def test_run2_keeps_original_uncertainty(self):
        # 2018 tight, pT bin 3: nominal 0.93, up 1.01
        _, up, _ = self._variations("2018", [1500.0], [1500.0], [True], [False])
        np.testing.assert_allclose(up, [1.01**2])


if __name__ == "__main__":
    unittest.main()
