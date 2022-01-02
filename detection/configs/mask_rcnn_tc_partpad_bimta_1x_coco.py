_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    # pretrained='pretrained/pvt_v2_b2.pth',
    # pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth',
    # pretrained=None,
    pretrained='models/checkpoint_tcformer.pth',
    backbone=dict(
        type='tcformer_partpad_small',
        style='pytorch'),
    neck=dict(
        type='BiMTA',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_heads=[4, 4, 4, 4],
        mlp_ratios=[4, 4, 4, 4],
        num_outs=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        # norm_cfg=dict(type='BN', requires_grad=True),
        no_norm_on_lateral=True
    ))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002/1.4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
