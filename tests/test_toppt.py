import awkward as ak
import numpy as np
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from corrections import GetTopPtWeight, top_pt_sf

LAST = 1 << 13   # isLastCopy


def _nnlo(pt):
    return 0.103 * np.exp(-0.0118 * pt) - 0.000134 * pt + 0.973


def _ext(pt):
    return 0.991 + 0.000075 * pt


class TopPtWeightTest(unittest.TestCase):
    def test_sf_is_nnlo_times_extrapolation_held_at_fit_edges(self):
        pt = np.array([0.0, 500.0, 1000.0, 1500.0, 2500.0])
        expected = [_nnlo(0) * _ext(0), _nnlo(500) * _ext(500), _nnlo(1000) * _ext(1000),
                    _nnlo(1500) * _ext(1000), _nnlo(2000) * _ext(1000)]
        np.testing.assert_allclose(top_pt_sf(pt), expected)

    def test_event_weight_uses_last_copy_tops_only(self):
        gp = ak.Array([
            # t (first copy, 900 GeV) -> t (last copy, 800 GeV); tbar last copy 600 GeV; a b quark
            [{"pdgId": 6, "statusFlags": 0, "pt": 900.0}, {"pdgId": 6, "statusFlags": LAST, "pt": 800.0},
             {"pdgId": -6, "statusFlags": LAST, "pt": 600.0}, {"pdgId": 5, "statusFlags": LAST, "pt": 300.0}],
            # no tops (e.g. QCD): weight 1
            [{"pdgId": 21, "statusFlags": LAST, "pt": 500.0}],
        ])
        w = GetTopPtWeight(gp)
        np.testing.assert_allclose(w, [np.sqrt(top_pt_sf(800.0) * top_pt_sf(600.0)), 1.0])


if __name__ == "__main__":
    unittest.main()
