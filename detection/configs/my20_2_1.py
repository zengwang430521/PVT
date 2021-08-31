_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/default_runtime.py'
]
model = dict(
    pretrained='pretrained/my20_2_330.pth',
    backbone=dict(
        type='mypvt20_2_small',
        style='pytorch',
        pretrained='pretrained/my20_2_330.pth'
    ),
    neck=dict(
        type='TokenFPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        kernel_size=[7, 5, 3, 1],
        sigma=[2, 2, 2, 2],
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

# find_unused_parameters = True
work_dir = 'work_dirs/my20_2_d1'