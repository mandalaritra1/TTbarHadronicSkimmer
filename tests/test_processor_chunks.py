"""Per-chunk bookkeeping of TTbarResProcessor on a local Z' NanoAOD file.

Small chunks produce the two cases where the processor used to lose events:
a category with events but no gen-matched tops (its weight variations were
skipped), and a chunk with only a few events after the baseline selection
(returned early after sumw was recorded). Skipped when the file is absent.
"""
import copy
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.getcwd())

ZPRIME = os.path.expanduser(
    "~/Projects/rootfiles/ttbar/2024/mc/ZPrime2000_W10/ZPrime2000_W10_0.root"
)
QCD = os.path.expanduser(
    "~/Projects/rootfiles/ttbar/2024/mc/QCD_PT1000to1500/QCD_PT1000to1500_0.root"
)
CATS = ["atcen", "atfwd", "2tcen", "2tfwd"]
WEIGHT_SYSTEMATICS = ["pileup", "isr", "fsr", "ttag_pt1", "ttag_pt2", "ttag_pt3"]
CHUNK = 15
NCHUNKS = 20


def _process(start, stop, path=ZPRIME, dataset="ZPrime2000_10", systematics=None):
    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
    from ttbarprocessor import TTbarResProcessor

    events = NanoEventsFactory.from_root(
        {path: "Events"}, schemaclass=NanoAODSchema, entry_start=start, entry_stop=stop,
        metadata={"dataset": dataset}, mode="eager",
    ).events()
    # weight systematics only: one pass per chunk, no jet variations
    processor = TTbarResProcessor(
        iov="2024", htCut=1400.0, deepAK8Cut="tight", topTagger="topvsqcd",
        anacats=CATS, systematics=systematics or ["nominal"] + WEIGHT_SYSTEMATICS,
    )
    return processor.process(events)


@unittest.skipUnless(os.path.exists(ZPRIME), "local Z' NanoAOD test file not available")
class ProcessorChunkBookkeepingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.outputs = [_process(i * CHUNK, (i + 1) * CHUNK) for i in range(NCHUNKS)]

    def test_weight_variations_filled_for_every_category_with_events(self):
        for output in self.outputs:
            cutflow = output["cutflow"]
            n_categories = sum(cutflow.get(c, 0) for c in CATS)
            self.assertEqual(output["systematics"].get("nominal", 0), n_categories)
            h = output["mtt_vs_mt"].project("systematic", "anacat")
            for i, cat in enumerate(CATS):
                if cutflow.get(cat, 0) == 0:
                    continue
                for syst in WEIGHT_SYSTEMATICS:
                    for direction in ("Up", "Down"):
                        filled = h[{"systematic": syst + direction, "anacat": i}]
                        self.assertGreater(filled.variance, 0.0, f"{cat} {syst}{direction}")

    def test_chunk_outputs_merge(self):
        # coffea's accumulate (futures/iterative executors) could not merge 3+
        # outputs carrying event_list, which is now off by default
        from coffea.processor import accumulate

        merged = accumulate(copy.deepcopy(o) for o in self.outputs)
        self.assertEqual(merged["cutflow"]["all events"], CHUNK * NCHUNKS)

    def test_chunk_with_few_selected_events_is_processed(self):
        few = [o for o in self.outputs if 0 < o["cutflow"].get("after_eventCut", 0) < 10]
        self.assertTrue(few, "no chunk with 1-9 events after the baseline selection")
        for output in few:
            self.assertIn("after_ttbarcandCuts", output["cutflow"])


@unittest.skipUnless(os.path.exists(QCD), "local QCD NanoAOD test file not available")
class FlatTheoryVariationTest(unittest.TestCase):
    def test_sample_without_lhe_weights_is_recorded(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = _process(0, 200, path=QCD, dataset="QCD_PT1000to1500",
                              systematics=["nominal", "pdf", "q2"])
        flat = output["flat_theory_variations"]
        self.assertEqual(sorted(k.split("|")[-1] for k in flat), ["pdf", "q2"])
        self.assertTrue(all(v == 200 for v in flat.values()))


if __name__ == "__main__":
    unittest.main()
