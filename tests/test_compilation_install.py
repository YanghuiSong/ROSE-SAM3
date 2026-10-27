import contextlib
import io
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'work_dirs/test_compilation_install'
TMP.mkdir(parents=True, exist_ok=True)
from core import promptbank_compiler as rebuild


class CompilationInstallTests(unittest.TestCase):
    def test_matching_search_output_installs_without_an_existing_final_file(self):
        with tempfile.TemporaryDirectory(dir=TMP) as raw:
            work = Path(raw)
            selected, final = work / 'searched.txt', work / 'final/compiled.txt'
            selected.write_bytes(b'a,b\nc\n')
            with contextlib.redirect_stdout(io.StringIO()):
                certificate = rebuild.finish_compilation(selected, final, work, {}, rebuild.sha256(selected))
            self.assertEqual(final.read_bytes(), selected.read_bytes())
            self.assertTrue(certificate['installed'])

    def test_mismatch_is_not_installed(self):
        with tempfile.TemporaryDirectory(dir=TMP) as raw:
            work = Path(raw)
            selected, final = work / 'searched.txt', work / 'final.txt'
            selected.write_bytes(b'new\n')
            with self.assertRaisesRegex(RuntimeError, 'no file was installed'):
                rebuild.finish_compilation(selected, final, work, {}, '0' * 64)
            self.assertFalse(final.exists())
            self.assertFalse(json.loads((work / 'rebuild_certificate.json').read_text())['installed'])

    def test_mismatch_preserves_an_existing_output(self):
        with tempfile.TemporaryDirectory(dir=TMP) as raw:
            work = Path(raw)
            selected, final = work / 'searched.txt', work / 'existing.txt'
            selected.write_bytes(b'new\n')
            final.write_bytes(b'existing\n')
            with self.assertRaises(RuntimeError):
                rebuild.finish_compilation(selected, final, work, {}, '0' * 64)
            self.assertEqual(final.read_bytes(), b'existing\n')

    def test_isaid_entry_freezes_base_dag_output_without_a_residual_stage(self):
        with tempfile.TemporaryDirectory(dir=TMP) as raw:
            directory = Path(raw)
            fixture = (ROOT / 'calibrated_prompts/final_promptbank/cls_isaid.txt').read_bytes()
            source = directory / 'source'
            shutil.copytree(ROOT / 'calibrated_prompts/recovery_inputs', source / 'calibrated_prompts/recovery_inputs')
            images = [directory / f'{i}.png' for i in range(512)]
            folds = {str(k): list(range(k, 512, 5)) for k in range(5)}
            base_manifest = ROOT / 'calibrated_prompts/recovery_inputs/base_calibration/isaid.json'
            residual_manifest = ROOT / 'calibrated_prompts/recovery_inputs/residual_calibration/isaid.json'
            def select(settings):
                settings.prompt_out.write_bytes(fixture)

            with patch('core.dag_responses.collect_responses') as collect, \
                 patch('core.dag_base_search.run_search', side_effect=select) as search, \
                 patch('core.dag_residual_search.run_search') as residual, \
                 patch.object(rebuild, 'list_calibration_images', return_value=images), \
                 patch.object(rebuild, 'stage1_calibration', return_value=(images, folds, 'train', base_manifest)), \
                 patch.object(rebuild, 'validate_residual_calibration', return_value=residual_manifest), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = rebuild.compile_promptbank('isaid', directory, work_dir=directory / 'work', repo=source)
            self.assertEqual(collect.call_count, 1)
            self.assertEqual(collect.call_args.kwargs['stage'], 'base')
            search.assert_called_once()
            residual.assert_not_called()
            self.assertTrue(result['installed'])
            certificate = json.loads((directory / 'work/rebuild_certificate.json').read_text())
            self.assertIsNone(certificate['stage2'])
            self.assertTrue(certificate['matches_frozen_promptbank'])
            self.assertEqual(certificate['native_kind'], 'dataset_proc_reference')


if __name__ == '__main__':
    unittest.main()
