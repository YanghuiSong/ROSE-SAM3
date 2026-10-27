"""Inference runtime shared by the dataset configs."""

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], ignore_index=255)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False))
