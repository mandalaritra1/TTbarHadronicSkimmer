import unittest

from plots.make2Drootfiles import (
    _check_unique_signal_specs,
    _zprime_signal_point_specs,
    _zprime_signal_specs,
)


class ZPrimeSignalSpecTest(unittest.TestCase):
    def test_one_percent_width_omits_width_from_output_label(self):
        specs = _zprime_signal_specs(["900", "4000"], ["1"], "2024")

        self.assertEqual(
            specs,
            [
                ("signalZPrime900", "ZPrime900_1_2024*.coffea"),
                ("signalZPrime4000", "ZPrime4000_1_2024*.coffea"),
            ],
        )

    def test_non_one_percent_width_keeps_width_in_output_label(self):
        specs = _zprime_signal_specs(["4000"], ["10", "30"], "2024")

        self.assertEqual(
            specs,
            [
                ("signalZPrime4000_10", "ZPrime4000_10_2024*.coffea"),
                ("signalZPrime4000_30", "ZPrime4000_30_2024*.coffea"),
            ],
        )

    def test_percent_suffix_is_accepted_for_width_labels(self):
        specs = _zprime_signal_specs(["4000"], ["1%", "10%"], "2024")

        self.assertEqual(
            specs,
            [
                ("signalZPrime4000", "ZPrime4000_1_2024*.coffea"),
                ("signalZPrime4000_10", "ZPrime4000_10_2024*.coffea"),
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
                ("signalZPrime900", "ZPrime900_1_2024*.coffea"),
                ("signalZPrime1000", "ZPrime1000_1_2024*.coffea"),
                ("signalZPrime4000", "ZPrime4000_1_2024*.coffea"),
                ("signalZPrime4000_10", "ZPrime4000_10_2024*.coffea"),
                ("signalZPrime4000_30", "ZPrime4000_30_2024*.coffea"),
            ],
        )

    def test_duplicate_output_labels_are_rejected(self):
        specs = _zprime_signal_point_specs(["4000", "4000:1"], "2024")

        with self.assertRaisesRegex(ValueError, "Duplicate signal output label"):
            _check_unique_signal_specs(specs)


if __name__ == "__main__":
    unittest.main()
