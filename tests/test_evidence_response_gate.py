import json
import tempfile
import unittest
from pathlib import Path

import torch

from core.dag_evidence_search import run_search


class EvidenceResponseGateTests(unittest.TestCase):
    def test_changing_only_responses_changes_the_selected_phrase(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo) as temporary:
            root = Path(temporary)
            cache = root / 'cache'
            cache.mkdir()
            (root / 'pool.json').write_text(json.dumps({'images': ['unlabeled.png']}))
            (root / 'classes.json').write_text(json.dumps({
                'classes': ['road', 'building'], 'native_bank': {'road': ['road'], 'building': ['building']}}))
            (root / 'candidates.json').write_text(json.dumps({'road': ['path'], 'building': []}))

            def save_candidate(value, presence):
                clean = {p: {'fused': torch.full((128, 128), v), 'presence': torch.tensor(pr)}
                         for p, v, pr in [('road', .60, .9), ('building', .10, .9), ('path', value, presence)]}
                torch.save({'clean': clean}, cache / '0000.pt')

            def select(name):
                return run_search(cache, root / 'pool.json', root / 'candidates.json',
                                  root / 'classes.json', root / name, root / name / 'bank.txt')['final_bank']

            save_candidate(.61, .9)
            self.assertEqual(select('supported'), [['road', 'path'], ['building']])
            save_candidate(0., 0.)
            self.assertEqual(select('silent'), [['road'], ['building']])


if __name__ == '__main__':
    unittest.main()
