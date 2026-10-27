import os

_base_ = ['./base_config.py']

custom_imports = dict(imports=['rose_sam3_segmentor'], allow_failed_imports=False)

data_root = os.environ.get('LOVEDA_ROOT', '/data/public/LoveDA')
dataset_type = 'LoveDADataset'

model = dict(
    type='ROSESAM3Segmentor',
    classname_path='./calibrated_prompts/final_promptbank/cls_loveda.txt',
    prob_thd=0.5,
    bg_idx=0,
    confidence_threshold=0.5,
)

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
    ),
)
