"""Keep the release limited to its supported inference and search interfaces."""
import ast
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[1]

def arguments(path, function, owner=None):
    tree=ast.parse((ROOT/path).read_text())
    body=tree.body if owner is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==function)
    assert node.args.vararg is None and node.args.kwarg is None
    return [a.arg for a in node.args.posonlyargs+node.args.args+node.args.kwonlyargs]


class ReleaseInterfacesTests(unittest.TestCase):
    def test_single_image_engine_has_no_optional_execution_modes(self):
        self.assertEqual(arguments('core/grounding_execution_engine.py','infer_image','GroundingExecutionEngine'),['self','image'])
        self.assertEqual(arguments('core/grounding_execution_engine.py','__init__','GroundingExecutionEngine'),
                         ['self','processor','prompt_bank','device','prompt_batch_size','confidence_threshold'])

    def test_reducer_always_returns_the_required_response_branches(self):
        self.assertEqual(arguments('core/result_reducer.py','reduce_slot_outputs','GroundingResultReducer'),
                         ['self','outputs','slot_count','output_size'])

    def test_model_only_exposes_active_inference_controls(self):
        self.assertEqual(arguments('rose_sam3_segmentor.py','__init__','ROSESAM3Segmentor'),
                         ['self','classname_path','device','prob_thd','bg_idx','confidence_threshold','prompt_batch_size'])
        self.assertEqual(arguments('core/prompt_bank.py','__init__','PromptBank'),
                         ['self','processor','classname_path','device'])

    def test_dataset_configs_only_set_dataset_specific_model_values(self):
        for path in (ROOT/'configs').glob('cfg_*.py'):
            with self.subTest(dataset=path.name):
                model=runpy.run_path(str(path))['model']
                self.assertEqual(set(model),{'type','classname_path','prob_thd','bg_idx','confidence_threshold'})

    def test_search_uses_fixed_reference_and_prepared_partition(self):
        self.assertEqual(arguments('core/promptbank_compiler.py','build_recovery_inputs'),['repo','dataset','work'])
        self.assertEqual(arguments('core/promptbank_compiler.py','stage1_calibration'),['repo','dataset','train_dir'])
        self.assertEqual(arguments('core/concept_search_metrics.py','evaluate_candidate'),
                         ['response','native_bank','phrase','class_id','prepared'])
        self.assertEqual(arguments('core/concept_search_metrics.py','load_response_cache'),['cache_dir','n'])


if __name__=='__main__':
    unittest.main()
