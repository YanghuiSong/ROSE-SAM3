import tempfile
import unittest
from pathlib import Path

from core.promptbank_compiler import SUPPORTED_DATASETS, build_recovery_inputs


class PotsdamRecoveryTests(unittest.TestCase):
    def test_potsdam_recovery_entry_accepts_the_dataset(self):
        self.assertIn('potsdam', SUPPORTED_DATASETS)

    def test_historical_potsdam_graph_supplies_21_response_queries(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo) as temporary:
            inputs = build_recovery_inputs(repo, 'potsdam', Path(temporary))
            phrases = {p for row in inputs['query_bank'].read_text().splitlines()
                       for p in row.split(',')}
            self.assertEqual(len(phrases), 21)
            self.assertIn('forest tree', phrases)


if __name__ == '__main__':
    unittest.main()
