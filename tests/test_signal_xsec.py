import ast
import os
import sys
import unittest

sys.path.insert(0, os.getcwd())


class SignalXsecTest(unittest.TestCase):
    def test_mass_outside_the_table_uses_the_1pb_reference(self):
        from ttbaranalysis import _signal_xsec

        # functions.xs covers ZPrime1 1000-4500 only; 400 GeV used to return None,
        # which saved the template at raw weighted counts
        self.assertEqual(_signal_xsec("ZPrime1", "400"), 1.0)

    def test_notebook_runner_uses_the_cli_helper(self):
        # ttbar_notebook builds widgets at import, so inspect its source instead
        with open("ttbar_notebook.py") as f:
            tree = ast.parse(f.read())
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        imported = {
            (n.module, a.name)
            for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            for a in n.names
        }
        self.assertNotIn("_signal_xsec", defined)
        self.assertIn(("ttbaranalysis", "_signal_xsec"), imported)


if __name__ == "__main__":
    unittest.main()
