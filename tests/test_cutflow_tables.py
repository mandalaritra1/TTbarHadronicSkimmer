import os
import sys
import unittest

sys.path.append(os.path.join(os.getcwd(), "python"))

from cutflow import (  # noqa: E402
    build_cutflow_matrix,
    build_cutflow_rows,
    category_label,
    format_latex_matrix,
    format_latex_table,
    format_markdown_matrix,
    format_markdown_table,
)


class CutflowTableFormattingTest(unittest.TestCase):
    def test_build_rows_uses_scaled_weights_and_efficiencies(self):
        output = {
            "cutflow_table_steps": [
                {"key": "input_events", "label": "Input events", "group": "Bookkeeping"},
                {"key": "trigger", "label": "Trigger", "group": "Preselection"},
                {"key": "tag_2tag", "label": "Two top-tagged jets", "group": "Tagging"},
            ],
            "cutflow_unweighted": {
                "input_events": 100,
                "trigger": 80,
                "tag_2tag": 20,
            },
            "cutflow_weighted": {
                "input_events": 50.0,
                "trigger": 40.0,
                "tag_2tag": 10.0,
            },
            "cutflow_weighted_scaled": {
                "input_events": 500.0,
                "trigger": 400.0,
                "tag_2tag": 100.0,
            },
        }

        rows = build_cutflow_rows(output)

        self.assertEqual([row["key"] for row in rows], ["input_events", "trigger", "tag_2tag"])
        self.assertEqual(rows[1]["weighted"], 400.0)
        self.assertAlmostEqual(rows[1]["eff_prev_percent"], 80.0)
        self.assertAlmostEqual(rows[2]["eff_total_percent"], 20.0)

    def test_markdown_and_latex_render_core_columns(self):
        rows = [
            {
                "group": "Bookkeeping",
                "step": "Input events",
                "events": 100.0,
                "weighted": 500.0,
                "eff_prev_percent": None,
                "eff_total_percent": 100.0,
                "weight_label": "Yield",
            }
        ]

        markdown = format_markdown_table(rows, title="Cutflow")
        latex = format_latex_table(rows, caption="Cutflow")

        self.assertIn("| Step | Events | Yield |", markdown)
        self.assertNotIn("| Group |", markdown)
        self.assertIn("Input events", markdown)
        self.assertIn(r"\begin{table}", latex)
        self.assertIn("Input events", latex)

    def test_legacy_cutflow_keys_are_adapted(self):
        output = {
            "cutflow": {
                "all events 1": 100,
                "all events": 90,
                "trigger": 80,
                "after_eventCut": 40,
                "after_ttbarcandCuts": 20,
                "2t0bcen": 5,
            },
            "cutflow_scaled": {
                "all events 1": 1000.0,
                "all events": 900.0,
                "trigger": 800.0,
                "after_eventCut": 400.0,
                "after_ttbarcandCuts": 200.0,
                "2t0bcen": 50.0,
            },
        }

        rows = build_cutflow_rows(output)

        self.assertIn("preselection", [row["key"] for row in rows])
        self.assertIn("ttbarcand", [row["key"] for row in rows])
        self.assertIn("category_2t0bcen", [row["key"] for row in rows])
        self.assertEqual(rows[-1]["step"], "Two top-tagged jets, 0 b-tags, central rapidity")
        self.assertEqual(rows[-1]["weighted"], 50.0)
        self.assertAlmostEqual(rows[-1]["eff_prev_percent"], 25.0)

    def test_compact_category_labels_are_spelled_out(self):
        self.assertEqual(category_label("atcen"), "Antitag, central rapidity")
        self.assertEqual(category_label("atfwd"), "Antitag, forward rapidity")
        self.assertEqual(category_label("2tcen"), "Two top-tagged jets, central rapidity")
        self.assertEqual(category_label("2tfwd"), "Two top-tagged jets, forward rapidity")


class CutflowMatrixTest(unittest.TestCase):
    STEPS = [
        {"key": "input_events", "label": "All events", "group": "Bookkeeping"},
        {"key": "trigger", "label": "Trigger", "group": "Preselection"},
        {"key": "tag_2tag", "label": "Two top-tagged jets", "group": "Tagging"},
    ]

    def _output(self, unweighted, scaled):
        return {
            "cutflow_table_steps": self.STEPS,
            "cutflow_unweighted": unweighted,
            "cutflow_weighted_scaled": scaled,
        }

    def test_events_matrix_columns_and_total(self):
        a = self._output(
            {"input_events": 100, "trigger": 80, "tag_2tag": 20},
            {"input_events": 0.0, "trigger": 0.0, "tag_2tag": 0.0},
        )
        b = self._output(
            {"input_events": 200, "trigger": 150, "tag_2tag": 40},
            {"input_events": 0.0, "trigger": 0.0, "tag_2tag": 0.0},
        )

        matrix = build_cutflow_matrix(
            [a, b], labels=["2016", "2017"], value="events", add_total=True
        )

        self.assertEqual(matrix["columns"], ["2016", "2017", "Total"])
        trigger = next(r for r in matrix["rows"] if r["key"] == "trigger")
        self.assertEqual(trigger["values"], [80.0, 150.0])
        self.assertEqual(trigger["total"], 230.0)

    def test_yield_matrix_uses_scaled_weights(self):
        a = self._output(
            {"input_events": 100, "trigger": 80, "tag_2tag": 20},
            {"input_events": 59740.0, "trigger": 3030.5, "tag_2tag": 29.0},
        )

        matrix = build_cutflow_matrix(
            [a], labels=["1000 GeV"], value="yield"
        )
        trigger = next(r for r in matrix["rows"] if r["key"] == "trigger")
        self.assertEqual(trigger["values"], [3030.5])

        markdown = format_markdown_matrix(matrix, title="Signal cutflow")
        latex = format_latex_matrix(matrix, caption="Signal cutflow")
        self.assertIn("| Cut | 1000 GeV |", markdown)
        self.assertIn("3030.5", markdown)
        self.assertIn(r"\begin{table}", latex)
        self.assertIn("3030.5", latex)

    def test_mismatched_labels_raise(self):
        a = self._output({"input_events": 1}, {"input_events": 1.0})
        with self.assertRaises(ValueError):
            build_cutflow_matrix([a], labels=["x", "y"])


if __name__ == "__main__":
    unittest.main()
