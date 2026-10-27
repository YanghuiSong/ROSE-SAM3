import os

_base_ = ['./base_config.py']

custom_imports = dict(imports=['rose_sam3_segmentor'], allow_failed_imports=False)

data_root = os.environ.get('POTSDAM_ROOT', '/data/public/Potsdam')
dataset_type = 'PotsdamDataset'

model = dict(
    type='ROSESAM3Segmentor',
    classname_path='./calibrated_prompts/final_promptbank/cls_potsdam.txt',
    prob_thd=0.1,
    bg_idx=5,
    confidence_threshold=0.4,
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
