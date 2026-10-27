import os

_base_ = ['./base_config.py']

custom_imports = dict(imports=['rose_sam3_segmentor'], allow_failed_imports=False)

data_root = os.environ.get('VDD_ROOT', '/data/public/VDD')
dataset_type = 'VDDDataset'

model = dict(
    type='ROSESAM3Segmentor',
    classname_path='./calibrated_prompts/final_promptbank/cls_vdd.txt',
    prob_thd=0.3,
    bg_idx=0,
    confidence_threshold=0.5,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/src', seg_map_path='test/gt'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
    ),
)
