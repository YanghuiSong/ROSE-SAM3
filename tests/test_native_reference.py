"""The released search uses the fixed native reference for each dataset."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / 'work_dirs/test_native_reference'
TEST_TMP.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('rebuild_native_test', ROOT / 'core/promptbank_compiler.py')
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)


class NativeReferenceTests(unittest.TestCase):
    def test_existing_default_preserves_old_input_protocol(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as raw:
            inputs = rebuild.build_recovery_inputs(ROOT, 'udd5', Path(raw))
            classes = json.loads(inputs['classes'].read_text())
            self.assertEqual(classes['native_bank']['vegetation'], ['vegetation'])

    def test_uavid_default_uses_the_verified_proc_reference(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as raw:
            inputs = rebuild.build_recovery_inputs(ROOT, 'uavid', Path(raw))
            spec = json.loads(inputs['classes'].read_text())
            self.assertEqual(sum(map(len, spec['native_bank'].values())), 25)
            self.assertEqual(spec['native_bank']['background'], ['background', 'other', 'cluttered ground'])
            self.assertEqual(inputs['native_source'].parent.name, 'native_references')

    def test_isaid_default_uses_the_verified_24_word_proc_reference(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as raw:
            inputs = rebuild.build_recovery_inputs(ROOT, 'isaid', Path(raw))
            spec = json.loads(inputs['classes'].read_text())
            self.assertEqual(len(spec['classes']), 16)
            self.assertEqual(sum(map(len, spec['native_bank'].values())), 24)
            self.assertEqual(spec['native_bank']['harbor'], ['harbor'])
            self.assertEqual(inputs['native_source'].parent.name, 'native_references')

    def test_isaid_base_search_has_the_verified_512_image_cohort(self):
        path = ROOT / 'calibrated_prompts/recovery_inputs/base_calibration/isaid.json'
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest['image_count'], 512)
        self.assertEqual(len(manifest['relative_images']), 512)

if __name__ == '__main__':
    unittest.main()
