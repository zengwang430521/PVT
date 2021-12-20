#!/usr/bin/env bash

srun -p mm_human \
srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p pat_earth  \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=det_eval python -u test.py configs/mask_rcnn_pvt_s_fpn_1x_coco.py models/mask_rcnn_pvt_s_fpn_1x_coco.pth \
    --work-dir=work_dirs/debug --eval=bbox,seg --launcher="slurm"



srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human \
srun -p pat_earth  \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=seg python -u train.py configs/sem_fpn/PVT/fpn_pvt_v2_b2_ade20k_40k.py \
    --work-dir=work_dirs/pvtv2_b2 --launcher="slurm"



    --job-name=seg python -u train.py configs/sem_fpn/PVT/mta_tcpart_s_ade20k_40k.py \
    --work-dir=work_dirs/tc_part_s --launcher="slurm"



