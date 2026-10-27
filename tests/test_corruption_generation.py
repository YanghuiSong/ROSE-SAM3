import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from rs_ovss_c.corruptions import CORRUPTIONS, corrupt_array
from rs_ovss_c.generate import main

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'work_dirs/test_corruption_generation'
TMP.mkdir(parents=True, exist_ok=True)
GOLDEN = {
    'gaussian': 'b41e4f81e2befcebe9f0f4676cf6147401e9de48d23fb2a6f6cb75b05a23a847',
    'motion_blur': 'fc100292f811cc3b6854eebe260d2891566ee4229b6168abe581d4263b21e342',
    'downsample': '2348d295beeec1ecf2339f3c78dd88bd2cf5d5dc8f04b874036c03206f5de5c1',
    'fog': 'd423db729a48ca256c97fd9194da232b87bc14c5444ead3adea798074f801836',
    'shadow': '77c04b37393e1a5ee03e68106a73a683e680c6821ee0adb506f81085ea53559e',
    'stripe': '2f1d8cea05d89463ac7bd43a28c6ebfc52db181decfce9710597466058a606db',
}


class CorruptionGenerationTests(unittest.TestCase):
    def test_reported_operators_keep_the_recorded_pixels_across_five_severities(self):
        self.assertEqual(set(CORRUPTIONS), set(GOLDEN))
        image = np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)
        for name, expected in GOLDEN.items():
            payload = b''.join(corrupt_array(image, name, severity, 3407, 'nested/scene.png').tobytes()
                               for severity in range(1, 6))
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)

    def test_generator_preserves_layout_sources_labels_and_existing_outputs(self):
        with tempfile.TemporaryDirectory(dir=TMP) as raw:
            folder = Path(raw)
            source = folder / 'img_dir/val/nested/scene.png'
            label = folder / 'ann_dir/val/nested/scene.png'
            source.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            Image.fromarray(np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)).save(source)
            Image.fromarray(np.zeros((32, 40), np.uint8)).save(label)
            initial = source.read_bytes(), label.read_bytes()
            output = folder / 'degraded'
            args = ['generate', str(folder / 'img_dir/val'), str(output), '--severity', '3', '--seed', '3407']
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            generated = {name: (output / f'{name}_s3/nested/scene.png').read_bytes() for name in CORRUPTIONS}
            self.assertEqual(len(list(output.rglob('*.png'))), 6)
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(initial, (source.read_bytes(), label.read_bytes()))
            self.assertTrue(all((output / f'{name}_s3/nested/scene.png').read_bytes() == data
                                for name, data in generated.items()))


if __name__ == '__main__':
    unittest.main()
