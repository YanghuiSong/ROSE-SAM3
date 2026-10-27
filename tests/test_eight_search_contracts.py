import json
import tempfile
import unittest
from pathlib import Path

from core.promptbank_compiler import SUPPORTED_DATASETS, build_recovery_inputs


class EightSearchContractTests(unittest.TestCase):
    def test_all_eight_have_a_loadable_finite_search_input(self):
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual(set(SUPPORTED_DATASETS), {'openearthmap', 'udd5', 'vdd', 'vaihingen',
                                                   'potsdam', 'uavid', 'loveda', 'isaid'})
        for dataset in SUPPORTED_DATASETS:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory(dir=repo) as temporary:
                inputs = build_recovery_inputs(repo, dataset, Path(temporary))
                classes = json.loads(inputs['classes'].read_text())['classes']
                rows = inputs['query_bank'].read_text().splitlines()
                self.assertEqual(len(classes), len(rows))
                self.assertTrue(all(rows))

    def test_cohort_partitions_match_their_actual_stage(self):
        root = Path(__file__).resolve().parents[1] / 'calibrated_prompts/recovery_inputs'
        for dataset in SUPPORTED_DATASETS:
            for stage in ('base_calibration', 'residual_calibration'):
                with self.subTest(dataset=dataset, stage=stage):
                    manifest = json.loads((root / stage / f'{dataset}.json').read_text())
                    count = len(manifest['relative_images'])
                    self.assertEqual(count, manifest['image_count'])
                    if stage == 'base_calibration' and dataset == 'loveda':
                        self.assertFalse(manifest['selection_uses_folds'])
                        self.assertEqual(manifest['partitions'], {})
                    else:
                        self.assertEqual(len(manifest['partitions']), 5)
                        indices = [i for fold in manifest['partitions'].values() for i in fold]
                        self.assertEqual(sorted(indices), list(range(count)))
                    if dataset in ('uavid', 'isaid'):
                        self.assertEqual(manifest['calibration_split'], 'train')
                        self.assertTrue(all('test' not in p.casefold() for p in manifest['relative_images']))


if __name__ == '__main__':
    unittest.main()
