"""Exercise both collection stages through their shared interface."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from core.dag_responses import collect_responses


class Processor:
    def __init__(self):
        self.views = []

    def set_image(self, image):
        self.views.append(image.tobytes())
        return {'image':image}

    def reset_all_prompts(self, state):
        pass

    def set_text_prompt(self, state, prompt):
        width,height = state['image'].size
        image=state['image']
        values=torch.tensor([image.getpixel((x,0))[0]/255 for x in range(width)])
        return {'semantic_mask_logits':values.repeat(height,1),
                'presence_score':torch.tensor(.75)}


class CollectionTests(unittest.TestCase):
    def exercise(self, stage, clean_only=False):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw)
            images=[]
            original=Image.new('RGB',(3,2))
            for x in range(3):
                for y in range(2):
                    original.putpixel((x,y),(x*100,0,0))
            for index in range(5):
                path=root/f'{index}.png'
                original.save(path)
                images.append(path)
            prompts=root/'prompts.txt'
            prompts.write_text('building,building\nroad\n')
            processor=Processor()
            with patch('core.dag_responses.build_processor',return_value=(processor,{})) as build, contextlib.redirect_stdout(io.StringIO()):
                manifest=collect_responses(config_path=root/'config.py',prompt_path=prompts,image_paths=images,
                                           cache_dir=root/'cache',device='cpu',dataset='udd5',stage=stage,
                                           clean_only=clean_only)
            self.assertEqual(manifest['prompts'],['building','road'])
            self.assertEqual(manifest['image_count'],5)
            self.assertFalse(manifest['labels_used'])
            self.assertEqual(len(processor.views),10 if stage=='base' and not clean_only else 5)
            base=stage=='base'
            self.assertEqual(build.call_args.kwargs['confidence'],.5 if base else None)
            record=torch.load(root/'cache'/('0000.pt' if base else '00000.pt'),weights_only=False)
            maps=record['clean' if base else 'responses']
            self.assertEqual(tuple(maps['building']['fused'].shape),(128,128) if base else (256,256))
            if base and not clean_only:
                self.assertTrue(torch.equal(record['clean']['building']['fused'],record['flip_aligned']['building']['fused']))
            else:
                self.assertNotIn('flip_aligned',record)

    def test_base_clean_and_aligned_flip(self):
        self.exercise('base')

    def test_evidence_clean_only(self):
        self.exercise('base',clean_only=True)

    def test_residual_without_flip(self):
        self.exercise('residual')


if __name__=='__main__':
    unittest.main()
