#!/usr/bin/env bash

srun -p mm_human \
srun -p pat_earth  \
srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=det_eval python -u test.py configs/retinanet_pvt_v2_b2_fpn_1x_coco.py models/mask_rcnn_pvt_s_fpn_1x_coco.pth \
    --work-dir=work_dirs/debug --eval=bbox --launcher="slurm"


    --job-name=det_part python -u test.py configs/mask_rcnn_pvt_v2_b2_fpn_1x_coco.py models/mask_rcnn_pvt_s_fpn_1x_coco.pth \
    --work-dir=work_dirs/debug --eval=bbox --launcher="slurm"