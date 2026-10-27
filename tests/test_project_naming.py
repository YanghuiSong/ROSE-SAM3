from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectNamingTests(unittest.TestCase):
    def test_sources_and_asset_names_use_the_project_name(self):
        previous_name = ''.join(map(chr, (114, 112, 115)))
        excluded = {'.git', '__pycache__', 'work_dirs'}
        stale = []
        for path in ROOT.rglob('*'):
            relative = path.relative_to(ROOT)
            if not path.is_file() or set(relative.parts) & excluded:
                continue
            if previous_name in relative.as_posix().lower():
                stale.append(str(relative))
            if path.suffix == '.png':
                continue
            if previous_name in path.read_text(encoding='utf-8').lower():
                stale.append(str(relative))
        self.assertEqual(sorted(set(stale)), [])


if __name__ == '__main__':
    unittest.main()
