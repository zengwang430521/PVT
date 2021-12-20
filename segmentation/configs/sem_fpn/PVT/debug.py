_base_ = [
    '../../_base_/models/fpn_r50.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py'
]
# model settings
# model = dict(
#     type='EncoderDecoder',
#     # pretrained='pretrained/pvt_small.pth',
#     pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth',
#     backbone=dict(
#         type='pvt_small',
#         style='pytorch'),
#     neck=dict(in_channels=[64, 128, 320, 512]),
#     decode_head=dict(num_classes=150))

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='../detection/models/checkpoint_tcformer.pth',
    pretrained=None,
    backbone=dict(
        type='tcformer_partpad_light',
        style='pytorch'),
    neck=dict(
        type='MTA',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_heads=[4, 4, 4, 4],
        mlp_ratios=[4, 4, 4, 4],
        num_outs=4),
    decode_head=dict(num_classes=150)
)

gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
