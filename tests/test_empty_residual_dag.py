import json
import tempfile
import unittest
from pathlib import Path

import torch

from core.dag_residual_search import run_search


class EmptyResidualDAGTests(unittest.TestCase):
    def test_saturated_base_bank_is_valid_but_still_checks_every_image(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo) as temporary:
            root = Path(temporary)
            cache = root / 'cache'
            cache.mkdir()
            (root / 'base.txt').write_text('road,path\nbuilding\n')
            nodes = [
                {'id': 'road', 'class_index': 0, 'phrase': 'road', 'immutable': True},
                {'id': 'building', 'class_index': 1, 'phrase': 'building', 'immutable': True},
                {'id': 'path', 'class_index': 0, 'phrase': 'path', 'depth': 1, 'immutable': False},
            ]
            (root / 'dag.json').write_text(json.dumps({'dataset': 'fixture', 'nodes': nodes}))
            (root / 'folds.json').write_text(json.dumps({'partitions': {str(i): [i] for i in range(5)}}))
            responses = {phrase: {branch: torch.ones((256, 256)) * value
                                 for branch in ('semantic', 'instance', 'fused')}
                         for phrase, value in (('road', .8), ('path', .7), ('building', .3))}
            for i in range(5):
                torch.save({'responses': responses}, cache / f'{i:05d}.pt')
            bank, cert = run_search(cache, root / 'dag.json', root / 'base.txt',
                                    root / 'folds.json', root / 'result')
            self.assertEqual(bank, [['road', 'path'], ['building']])
            self.assertEqual(cert['candidate_count'], 0)
            self.assertEqual(cert['calibration_image_count'], 5)
            self.assertEqual(cert['final_residual_nodes'], {})
            self.assertEqual(cert['native_anchor_core']['0']['images'], 5)
            responses.pop('road')
            torch.save({'responses': responses}, cache / '00004.pt')
            with self.assertRaisesRegex(RuntimeError, 'missing PromptBank phrase: road'):
                run_search(cache, root / 'dag.json', root / 'base.txt',
                           root / 'folds.json', root / 'bad')


if __name__ == '__main__':
    unittest.main()
