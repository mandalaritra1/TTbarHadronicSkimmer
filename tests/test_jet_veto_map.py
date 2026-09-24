import awkward as ak
import numpy as np
import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.join(os.getcwd(), "python"))
import corrections


class _FakeJetVetoCorrection:
    def evaluate(self, map_name, eta, phi):
        assert map_name == "jetvetomap"
        # Mark positive eta as a vetoed detector region.
        return np.asarray(eta) > 0.0


class JetVetoMapTest(unittest.TestCase):
    def test_run3_jet_veto_map_uses_only_eligible_jets(self):
        jets = ak.Array([
            # Eligible and in the mapped region -> event vetoed.
            [{"pt": 50.0, "eta": 0.5, "phi": 0.1, "jetId": 6, "chEmEF": 0.1, "neEmEF": 0.1}],
            # Mapped, but below 15 GeV -> ignored.
            [{"pt": 10.0, "eta": 0.5, "phi": 0.2, "jetId": 6, "chEmEF": 0.1, "neEmEF": 0.1}],
            # Mapped, but fails TightLepVeto -> ignored.
            [{"pt": 50.0, "eta": 0.5, "phi": 0.3, "jetId": 2, "chEmEF": 0.1, "neEmEF": 0.1}],
            # Mapped, but dominated by EM energy -> ignored.
            [{"pt": 50.0, "eta": 0.5, "phi": 0.4, "jetId": 6, "chEmEF": 0.5, "neEmEF": 0.5}],
            # Eligible but outside the mapped region -> retained.
            [{"pt": 50.0, "eta": -0.5, "phi": 0.5, "jetId": 6, "chEmEF": 0.1, "neEmEF": 0.1}],
            [],
        ])

        with (
            mock.patch.object(
                corrections,
                "_load_jet_veto_correction",
                return_value=_FakeJetVetoCorrection(),
            ),
            mock.patch.object(
                corrections,
                "_tight_lepton_veto_mask",
                side_effect=lambda jet_collection, iov: (jet_collection.jetId & 4) != 0,
            ),
        ):
            result = corrections.GetJetVetoMapMask(jets, "2024")

        self.assertEqual(result.tolist(), [False, True, True, True, True, True])

    def test_run2_iov_passes_without_a_run3_map(self):
        jets = ak.Array([[], []])
        self.assertEqual(
            corrections.GetJetVetoMapMask(jets, "2018").tolist(),
            [True, True],
        )


if __name__ == "__main__":
    unittest.main()
