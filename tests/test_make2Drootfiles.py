import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import hist
from coffea import util

from plots.make2Drootfiles import (
    _check_unique_signal_specs,
    _resolve_signal_outputs,
    _zprime_signal_point_specs,
    _zprime_signal_specs,
    _template_name,
    _year_label,
)

import signal_grouped as sg  # noqa: E402  (plots.make2Drootfiles puts python/ on sys.path)


class ZPrimeSignalSpecTest(unittest.TestCase):
    def test_one_percent_width_omits_width_from_output_label(self):
        specs = _zprime_signal_specs(["900", "4000"], ["1"], "2024")

        self.assertEqual(
            specs,
            [
                ("signalZPrime900", "900", "1"),
                ("signalZPrime4000", "4000", "1"),
            ],
        )

    def test_non_one_percent_width_keeps_width_in_output_label(self):
        specs = _zprime_signal_specs(["4000"], ["10", "30"], "2024")

        self.assertEqual(
            specs,
            [
                ("signalZPrime4000_10", "4000", "10"),
                ("signalZPrime4000_30", "4000", "30"),
            ],
        )

    def test_percent_suffix_is_accepted_for_width_labels(self):
        specs = _zprime_signal_specs(["4000"], ["1%", "10%"], "2024")

        self.assertEqual(
            specs,
            [
                ("signalZPrime4000", "4000", "1"),
                ("signalZPrime4000_10", "4000", "10"),
            ],
        )

    def test_exact_points_default_to_one_percent_width(self):
        specs = _zprime_signal_point_specs(
            ["900", "1000", "4000", "4000:10", "4000:30"],
            "2024",
        )

        self.assertEqual(
            specs,
            [
                ("signalZPrime900", "900", "1"),
                ("signalZPrime1000", "1000", "1"),
                ("signalZPrime4000", "4000", "1"),
                ("signalZPrime4000_10", "4000", "10"),
                ("signalZPrime4000_30", "4000", "30"),
            ],
        )

    def test_duplicate_output_labels_are_rejected(self):
        specs = _zprime_signal_point_specs(["4000", "4000:1"], "2024")

        with self.assertRaisesRegex(ValueError, "Duplicate signal output label"):
            _check_unique_signal_specs(specs)


def _signal_output(mass, weight):
    """One ZPrime point in the legacy per-mass layout (mtt_vs_mt, no dataset axis),
    filled once at m_tt = mass with the given weight."""
    mtt_vs_mt = hist.Hist(
        hist.axis.StrCategory(["nominal"], name="systematic"),
        hist.axis.IntCategory([0, 1], name="anacat"),
        hist.axis.Regular(100, 0, 500, name="jetmass"),
        hist.axis.Regular(92, 800, 10000, name="ttbarmass"),
        storage="weight",
    )
    mtt_vs_mt.fill(systematic="nominal", anacat=0, jetmass=175.0, ttbarmass=mass, weight=weight)
    return {"mtt_vs_mt": mtt_vs_mt}


class ResolveSignalOutputsTest(unittest.TestCase):
    """Locating one ZPrime point: legacy per-mass file or grouped width-file."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.coffea_dir = Path(tmp.name)

    def _save_grouped(self, name, width, masses):
        """Write a grouped width-file holding ``masses`` on its dataset axis. Each
        point is filled at its own m_tt with weight = width, so picking the wrong
        mass or width changes the histogram."""
        per_mass = {
            f"ZPrime{mass}_{width}": _signal_output(mass, weight=float(width))
            for mass in masses
        }
        util.save(sg.stack_masses(per_mass), self.coffea_dir / name)
        return per_mass

    def _resolve(self, mass, width):
        with contextlib.redirect_stdout(io.StringIO()):
            return _resolve_signal_outputs(self.coffea_dir, "2024", mass, width, util)

    def test_grouped_width_files_are_searched_for_the_requested_point(self):
        # --signal-batch splits one width over several files, and the 10% file
        # holds the same masses as the 1% one.
        low = self._save_grouped("ZPrime1_2024_500to900.coffea", "1", [500, 900])
        high = self._save_grouped("ZPrime1_2024_1000to4000.coffea", "1", [1000, 4000])
        wide = self._save_grouped("ZPrime10_2024_1000to4000.coffea", "10", [1000, 4000])

        for mass, width, expected in [
            ("900", "1", low["ZPrime900_1"]),
            ("4000", "1", high["ZPrime4000_1"]),
            ("4000", "10", wide["ZPrime4000_10"]),
        ]:
            with self.subTest(mass=mass, width=width):
                (output,) = self._resolve(mass, width)
                # sliced to the one point: legacy layout, no dataset axis left
                self.assertEqual(output["mtt_vs_mt"], expected["mtt_vs_mt"])

    def test_legacy_per_mass_file_takes_precedence_over_grouped(self):
        legacy = _signal_output(4000, weight=0.5)
        util.save(legacy, self.coffea_dir / "ZPrime4000_1_2024_.coffea")
        self._save_grouped("ZPrime1_2024_1000to4000.coffea", "1", [1000, 4000])

        (output,) = self._resolve("4000", "1")

        self.assertEqual(output["mtt_vs_mt"], legacy["mtt_vs_mt"])

    def test_point_missing_from_every_file_raises(self):
        self._save_grouped("ZPrime1_2024_1000to4000.coffea", "1", [1000, 4000])

        with self.assertRaisesRegex(FileNotFoundError, "ZPrime900_1"):
            self._resolve("900", "1")


class TemplateNameTest(unittest.TestCase):
    """The 2DAlphabet input keys are the interface contract with bgestimation."""

    def test_nominal_and_varied_keys(self):
        cases = {
            ("cen", "2024", "Pass", "nominal"): "MttvsMtCen24Pass",
            ("fwd", "2024", "Fail", "nominal"): "MttvsMtFwd24Fail",
            ("cen", "2025", "Pass", "jesUp"): "MttvsMtCen25PassJESup",
            ("fwd", "2025", "Fail", "ttag_pt2Down"): "MttvsMtFwd25FailTTAG_PT2down",
            ("cen", "2024", "Fail", "isrUp"): "MttvsMtCen24FailISRup",
        }
        for (cat, year, region, syst), name in cases.items():
            self.assertEqual(_template_name(cat, _year_label(year), region, syst), name)

    def test_unknown_category_raises(self):
        with self.assertRaises(KeyError):
            _template_name("central", "24", "Pass", "nominal")


if __name__ == "__main__":
    unittest.main()
