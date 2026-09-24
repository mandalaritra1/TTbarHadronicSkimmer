import os
import sys
import unittest

sys.path.append(os.path.join(os.getcwd(), "python"))
from hists import build_output_histograms


class TruthHistogramLayoutTest(unittest.TestCase):
    def test_nominal_only_truth_hists_have_no_systematic_axis(self):
        output = build_output_histograms(
            anacats=["atcen", "atfwd", "2tcen", "2tfwd"],
            systematics=["nominal", "jes", "jer", "pileup", "pdf", "q2"],
            no_syst=False,
        )
        for name in ("gen_jetmsd_reco_jetmsd", "jet_mass_resolution"):
            self.assertNotIn("systematic", output[name].axes.name, name)
        # with the systematic axis this hist was 21.6M bins, 346 MB per chunk output
        self.assertLess(output["gen_jetmsd_reco_jetmsd"].values(flow=True).size, 1_000_000)

    def test_event_list_only_on_request(self):
        kwargs = dict(anacats=["atcen"], systematics=["nominal"], no_syst=True)
        self.assertNotIn("event_list", build_output_histograms(**kwargs))
        self.assertIn("event_list", build_output_histograms(store_event_list=True, **kwargs))


if __name__ == "__main__":
    unittest.main()
