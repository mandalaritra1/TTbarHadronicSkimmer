import json
import tempfile
import unittest
from pathlib import Path

import run_toptag_wp


class TopTagRunnerFilesetTest(unittest.TestCase):
    def test_local_fileset_uses_rootdir_and_maxfiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            rootdir = Path(tmp)
            sample_dir = rootdir / "2024" / "mc" / "TTto4Q"
            sample_dir.mkdir(parents=True)
            for idx in range(3):
                (sample_dir / f"file{idx}.root").touch()

            fileset = run_toptag_wp.build_local_fileset(rootdir, "2024", maxfiles=2)

        self.assertEqual(list(fileset), ["TTto4Q"])
        self.assertEqual(len(fileset["TTto4Q"]["files"]), 2)
        self.assertEqual(fileset["TTto4Q"]["metadata"]["xsec_pb"], 419.9)

    def test_manifest_fileset_uses_redirector_metadata_and_maxfiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            qcd_json = tmpdir / "QCD.json"
            ttbar_json = tmpdir / "TTbar.json"
            data_json = tmpdir / "data.json"
            qcd_json.write_text(json.dumps({
                "2024": {
                    "QCD_PT-170to300": {
                        "files": ["/store/qcd/low.root"],
                        "metadata": {
                            "sample": "QCD",
                            "subsample": "QCD_PT-170to300",
                            "year": "2024",
                            "is_mc": True,
                            "xsec_pb": 113300.0,
                        },
                    },
                    "QCD_PT-600to800": {
                        "files": ["/store/qcd/a.root", "/store/qcd/b.root", "/store/qcd/c.root"],
                        "metadata": {
                            "sample": "QCD",
                            "subsample": "QCD_PT-600to800",
                            "year": "2024",
                            "is_mc": True,
                            "xsec_pb": 178.7,
                        },
                    }
                }
            }))
            ttbar_json.write_text(json.dumps({
                "2024": {
                    "inclusive": {
                        "files": ["/store/tt/a.root", "/store/tt/b.root"],
                        "metadata": {
                            "sample": "TTbar",
                            "subsample": "inclusive",
                            "year": "2024",
                            "is_mc": True,
                            "xsec_pb": 419.9,
                        },
                    }
                }
            }))
            data_json.write_text(json.dumps({
                "2024": {
                    "C": ["/store/data/c.root", "/store/data/d.root"]
                }
            }))

            fileset = run_toptag_wp.build_manifest_fileset(
                qcd_json,
                ttbar_json,
                "2024",
                redirector="root://cmsxrootd.fnal.gov/",
                maxfiles=1,
                data_json=data_json,
            )

        self.assertEqual(set(fileset), {"QCD_PT-600to800", "TTbar"})
        self.assertEqual(fileset["QCD_PT-600to800"]["files"], [
            "root://cmsxrootd.fnal.gov//store/qcd/a.root"
        ])
        self.assertEqual(fileset["TTbar"]["metadata"]["xsec_pb"], 419.9)

    def test_local_fileset_skips_low_qcd_pt_bins(self):
        with tempfile.TemporaryDirectory() as tmp:
            rootdir = Path(tmp)
            low_dir = rootdir / "2024" / "mc" / "QCD_PT-170to300"
            high_dir = rootdir / "2024" / "mc" / "QCD_PT-300to470"
            low_dir.mkdir(parents=True)
            high_dir.mkdir(parents=True)
            (low_dir / "low.root").touch()
            (high_dir / "high.root").touch()

            fileset = run_toptag_wp.build_local_fileset(
                rootdir,
                "2024",
                samples=["QCD"],
            )

        self.assertEqual(set(fileset), {"QCD_PT-300to470"})

    def test_manifest_fileset_can_run_data_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            qcd_json = tmpdir / "QCD.json"
            ttbar_json = tmpdir / "TTbar.json"
            data_json = tmpdir / "data.json"
            qcd_json.write_text(json.dumps({"2024": {}}))
            ttbar_json.write_text(json.dumps({"2024": {}}))
            data_json.write_text(json.dumps({
                "2024": {
                    "C": ["/store/data/c.root", "/store/data/d.root"]
                }
            }))

            fileset = run_toptag_wp.build_manifest_fileset(
                qcd_json,
                ttbar_json,
                "2024",
                redirector="root://cmsxrootd.fnal.gov/",
                maxfiles=1,
                samples=["Data"],
                data_json=data_json,
            )

        self.assertEqual(set(fileset), {"Data_C"})
        self.assertEqual(fileset["Data_C"]["files"], [
            "root://cmsxrootd.fnal.gov//store/data/c.root"
        ])
        self.assertFalse(fileset["Data_C"]["metadata"]["is_mc"])


if __name__ == "__main__":
    unittest.main()
