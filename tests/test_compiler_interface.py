"""Protect the single compilation interface independently of its internal layout."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from unittest.mock import patch

from core.promptbank_compiler import SUPPORTED_DATASETS, compile_promptbank

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT/'work_dirs/test_compiler_interface'
TMP.mkdir(parents=True,exist_ok=True)


class CompilerInterfaceTests(unittest.TestCase):
    def test_existing_banks_are_verified_without_images_or_model_imports(self):
        for dataset in SUPPORTED_DATASETS:
            with self.subTest(dataset=dataset):
                result = compile_promptbank(dataset, ROOT/'does_not_exist')
                self.assertEqual(result['status'], 'verified_existing')

    def test_existing_wrong_bank_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=TMP) as raw:
            root = Path(raw)
            destination = root/'calibrated_prompts'
            shutil.copytree(ROOT/'calibrated_prompts', destination)
            bank = destination/'final_promptbank/cls_udd5.txt'
            bank.write_bytes(b'wrong\n')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                compile_promptbank('udd5', root/'missing', repo=root)
            self.assertEqual(bank.read_bytes(), b'wrong\n')

    def test_unsupported_dataset_fails_before_search(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported dataset'):
            compile_promptbank('unknown', ROOT)

    def test_cli_does_not_expose_internal_search_tuning(self):
        result = subprocess.run([sys.executable,str(ROOT/'scripts/rebuild_promptbank.py'),'--help'],
                                text=True,capture_output=True,check=True)
        self.assertIn('--dataset',result.stdout)
        for option in ('--query-budget','--proposal-soft-margin','--metric-stride','--config'):
            self.assertNotIn(option,result.stdout)

    def test_compilation_uses_functions_not_a_subprocess_command_chain(self):
        text = (ROOT/'core/promptbank_compiler.py').read_text()
        self.assertNotIn('subprocess',text)
        self.assertNotIn('sys.executable',text)

    def test_integrity_verifier_routes_all_missing_banks_to_compilation(self):
        spec=importlib.util.spec_from_file_location('integrity_check',ROOT/'scripts/verify_promptbank.py')
        verifier=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        with tempfile.TemporaryDirectory(dir=TMP) as raw:
            source=Path(raw)
            shutil.copytree(ROOT/'calibrated_prompts/recovery_inputs',source/'calibrated_prompts/recovery_inputs')
            for dataset in SUPPORTED_DATASETS:
                with self.subTest(dataset=dataset), \
                     patch.object(verifier,'__file__',str(source/'scripts/verify_promptbank.py')), \
                     patch('sys.argv',['verify_promptbank.py','--dataset',dataset]):
                    with self.assertRaisesRegex(FileNotFoundError,'rebuild_promptbank.py'):
                        verifier.main()


if __name__ == '__main__':
    unittest.main()
