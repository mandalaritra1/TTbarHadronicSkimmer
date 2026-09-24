import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.getcwd())
from python import provenance


class ProvenanceTest(unittest.TestCase):
    def test_records_revision_config_and_payloads(self):
        record = provenance.collect({"iov": "2024", "systematics": ["nominal", "jes"]})
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.assertEqual(record["git_sha"], head)
        self.assertEqual(record["config"]["iov"], "2024")
        self.assertEqual(record["config"]["systematics"], repr(["nominal", "jes"]))
        self.assertTrue(any(k.startswith("data/corrections/") for k in record["payload_sha256"]))
        self.assertTrue(all(len(v) == 64 for v in record["payload_sha256"].values()))


if __name__ == "__main__":
    unittest.main()
