import json
import os
import sys
import unittest
from types import SimpleNamespace

import correctionlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from corrections import GetPUSF
from weights import _mc_campaign

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DERIVED = os.path.join(ROOT, "data/corrections/puWeights/Summer24_to_2022_2023")
IOVS = ("2022preEE", "2022postEE", "2023preBPix", "2023postBPix")


def _events(ntrue):
    return SimpleNamespace(Pileup=SimpleNamespace(nTrueInt=np.asarray(ntrue, dtype=float)))


class Summer24PileupTest(unittest.TestCase):
    def setUp(self):
        self.profile = np.array(json.load(open(os.path.join(DERIVED, "derivation_summary.json")))
                                ["summer24_profile_measured"])
        self.x = np.arange(len(self.profile)) + 0.5

    def test_derived_weights_preserve_normalisation_on_summer24(self):
        # the point of the derivation: <w> = 1 on the Summer24 profile (central 2022/23: 0.72-1.40)
        for iov in IOVS:
            nom, up, down = GetPUSF(_events(self.x), iov, mc_campaign="Summer24")
            for w in (nom, up, down):
                self.assertAlmostEqual(float(np.sum(w * self.profile)), 1.0, delta=0.003, msg=iov)

    def test_without_campaign_the_central_weights_are_used(self):
        central = correctionlib.CorrectionSet.from_file(
            os.path.join(ROOT, "data/corrections/puWeights/2022_Summer22/puWeights.json.gz"))
        nom, _, _ = GetPUSF(_events(self.x), "2022preEE")
        np.testing.assert_allclose(
            nom, central["Collisions2022_355100_357900_eraBCD_GoldenJson"].evaluate(self.x, "nominal"))

    def test_2024_ignores_the_campaign_flag(self):
        # Summer24 is the native MC of 2024/2025: always the central 2024 weights
        a, _, _ = GetPUSF(_events(self.x), "2024")
        b, _, _ = GetPUSF(_events(self.x), "2024", mc_campaign="Summer24")
        np.testing.assert_allclose(a, b)

    def test_campaign_read_from_the_input_file_name(self):
        ev = SimpleNamespace(metadata={"filename": "root://x//store/mc/RunIII2024Summer24NanoAODv15/ZPrimeToTT/a.root"})
        self.assertEqual(_mc_campaign(ev), "Summer24")
        ev = SimpleNamespace(metadata={"filename": "root://x//store/mc/Run3Summer22NanoAODv15/TTto4Q/a.root"})
        self.assertIsNone(_mc_campaign(ev))
        self.assertIsNone(_mc_campaign(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
