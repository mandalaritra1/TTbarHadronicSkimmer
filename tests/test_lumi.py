import os
import sys
import unittest

sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "python"))


class LumiTableTest(unittest.TestCase):
    def test_every_reader_uses_the_one_table(self):
        import lumi
        import ttbarprocessor
        import functions
        from python import functions as package_functions

        self.assertIs(ttbarprocessor._LUMI_PB, lumi.LUMI_PB)
        self.assertIs(functions.lumi, lumi.LUMI_PB)
        # imported via the package path it is a second module object, same values
        self.assertEqual(package_functions.lumi, lumi.LUMI_PB)
        # PPD Run3-2025 table, eras C-G
        self.assertEqual(lumi.LUMI_PB["2025"], 110370.0)


if __name__ == "__main__":
    unittest.main()
