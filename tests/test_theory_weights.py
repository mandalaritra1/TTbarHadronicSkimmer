import os
import sys
import unittest

import awkward as ak
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "python"))
from corrections import GetPDFWeights, GetQ2weights


def _events(**branches):
    n = len(next(iter(branches.values()))) if branches else 2
    fields = {"event": np.arange(n)}
    fields.update({k: ak.Array(v) for k, v in branches.items()})
    return ak.zip(fields, depth_limit=1)


class TheoryWeightsTest(unittest.TestCase):
    def test_q2_envelope_from_nine_scale_weights(self):
        scales = [[0.8, 0.9, 1.0, 0.95, 1.0, 1.05, 1.0, 1.1, 1.2]] * 2
        _, up, down = GetQ2weights(_events(LHEScaleWeight=scales))
        np.testing.assert_allclose(up, [1.2, 1.2])
        np.testing.assert_allclose(down, [0.8, 0.8])

    def test_malformed_scale_weights_raise(self):
        with self.assertRaises(ValueError):
            GetQ2weights(_events(LHEScaleWeight=[[1.0] * 9, [1.0] * 7]))

    def test_ragged_pdf_weights_raise(self):
        with self.assertRaises(ValueError):
            GetPDFWeights(_events(LHEPdfWeight=[[1.0] * 103, []]))

    def test_missing_lhe_branches_give_flat_variations(self):
        # pure-Pythia samples (Run-3 QCD_PT); the processor records these
        events = _events()
        for getter in (GetQ2weights, GetPDFWeights):
            _, up, down = getter(events)
            np.testing.assert_allclose(up, 1.0)
            np.testing.assert_allclose(down, 1.0)


if __name__ == "__main__":
    unittest.main()
