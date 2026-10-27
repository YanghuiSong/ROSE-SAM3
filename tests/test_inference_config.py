import ast
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / 'configs').glob('cfg_*.py'))


class InferenceConfigTests(unittest.TestCase):
    def test_all_eight_model_configs_use_supported_arguments(self):
        tree = ast.parse((ROOT / 'rose_sam3_segmentor.py').read_text())
        segmentor = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                         and node.name == 'ROSESAM3Segmentor')
        constructor = next(node for node in segmentor.body if isinstance(node, ast.FunctionDef)
                           and node.name == '__init__')
        supported = {arg.arg for arg in constructor.args.args} - {'self'}
        self.assertEqual(len(CONFIGS), 8)
        for path in CONFIGS:
            with self.subTest(config=path.name):
                model = runpy.run_path(str(path))['model']
                self.assertEqual(set(model) - supported, {'type'})

    def test_configs_only_declare_the_active_test_data_pipeline(self):
        for path in CONFIGS:
            with self.subTest(config=path.name):
                config = runpy.run_path(str(path))
                self.assertIn('test_dataloader', config)
                for field in ('data', 'class_names', 'train_dataloader', 'val_dataloader',
                              'train_cfg', 'val_cfg', 'optim_wrapper', 'param_scheduler'):
                    self.assertFalse(field in config, f"{path.name} declares unused {field}")

    def test_evaluator_does_not_declare_instance_iou_thresholds(self):
        for path in [ROOT / 'configs/base_config.py', *CONFIGS]:
            with self.subTest(config=path.name):
                config = runpy.run_path(str(path))
                self.assertNotIn('iou_thresholds', config.get('test_evaluator', {}))

    def test_runtime_config_has_no_training_only_hooks(self):
        config = runpy.run_path(str(ROOT / 'configs/base_config.py'))
        self.assertFalse({'checkpoint', 'param_scheduler', 'sampler_seed'}
                         & set(config.get('default_hooks', {})))
        self.assertNotIn('load_from', config)
        self.assertNotIn('resume', config)

    def test_inference_source_and_configs_have_no_boundary_switch(self):
        paths = [ROOT / 'rose_sam3_segmentor.py', ROOT / 'core/fusion_heads.py',
                 ROOT / 'configs/base_config.py', *CONFIGS]
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                source = path.read_text()
                for token in ('boundary_enabled', 'sich_boundary', 'boundary_term', 'max_pool2d'):
                    self.assertNotIn(token, source)

    def test_prediction_export_only_sets_the_evaluator_output_directory(self):
        source = (ROOT / 'eval_rose_sam3.py').read_text()
        self.assertIn('cfg.test_evaluator["output_dir"] = args.out', source)
        self.assertNotIn('keep_results', source)


if __name__ == '__main__':
    unittest.main()
