import os

_base_ = ['./base_config.py']

custom_imports = dict(imports=['rose_sam3_segmentor'], allow_failed_imports=False)

data_root = os.environ.get('OPENEARTHMAP_ROOT', '/data/public/OpenEarthMap')
dataset_type = 'OpenEarthMapDataset'

model = dict(
    type='ROSESAM3Segmentor',
    classname_path='./calibrated_prompts/final_promptbank/cls_openearthmap.txt',
    prob_thd=0.02,
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
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(448, 448), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
    ),
)
