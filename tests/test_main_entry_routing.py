import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'work_dirs/test_main_entry_routing'
TMP.mkdir(parents=True, exist_ok=True)


@unittest.skipUnless(os.name != 'nt' and shutil.which('bash'), 'Linux bash entry point')
class MainEntryRoutingTests(unittest.TestCase):
    def run_entry(self, dataset, bank_exists):
        with tempfile.TemporaryDirectory(dir=TMP) as raw:
            folder = Path(raw)
            source = folder / 'source'
            (source / 'scripts').mkdir(parents=True)
            shutil.copy2(ROOT / 'scripts/run_main_table.sh', source / 'scripts/run_main_table.sh')
            executable = folder / 'bin/python'
            executable.parent.mkdir()
            executable.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
with Path(os.environ['ROUTE_LOG']).open('a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if Path(sys.argv[1]).name == 'rebuild_promptbank.py':
    dataset = sys.argv[sys.argv.index('--dataset') + 1]
    target = Path.cwd() / f'calibrated_prompts/final_promptbank/cls_{dataset}.txt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'routing fixture\\n')
''')
            executable.chmod(0o755)
            if bank_exists:
                target = source / f'calibrated_prompts/final_promptbank/cls_{dataset}.txt'
                target.parent.mkdir(parents=True)
                target.write_bytes(b'routing fixture\n')
            env = {k: v for k, v in os.environ.items() if not (
                k.startswith('CALIBRATION_') or k == 'DAG_WORK_ROOT')}
            dataset_root = folder / 'data'
            env.update(PATH=str(executable.parent) + os.pathsep + env['PATH'],
                       SAM3_CHECKPOINT='/unused/checkpoint', SAM3_BPE_PATH='/unused/bpe',
                       MAIN_RUN_ROOT=str(folder / 'metrics'), ROUTE_LOG=str(folder / 'calls.jsonl'))
            env[dataset.upper() + '_ROOT'] = str(dataset_root)
            result = subprocess.run(['bash', str(source / 'scripts/run_main_table.sh'), dataset],
                                    cwd=source, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in (folder / 'calls.jsonl').read_text().splitlines()]
            return calls, str(dataset_root)

    def test_missing_bank_uses_dataset_root_and_runs_rebuild_before_verification(self):
        for dataset in ('udd5', 'vdd', 'loveda', 'isaid'):
            with self.subTest(dataset=dataset):
                calls, root = self.run_entry(dataset, False)
                self.assertEqual([Path(row[0]).name for row in calls],
                                 ['rebuild_promptbank.py', 'verify_promptbank.py', 'eval_rose_sam3.py'])
                actual = calls[0][calls[0].index('--calibration-image-dir') + 1]
                self.assertEqual(actual, root + ('/train/src' if dataset in ('udd5', 'vdd') else '/img_dir/train'))

    def test_existing_bank_is_verified_without_research(self):
        calls, _ = self.run_entry('isaid', True)
        self.assertEqual([Path(row[0]).name for row in calls], ['verify_promptbank.py', 'eval_rose_sam3.py'])


if __name__ == '__main__':
    unittest.main()
