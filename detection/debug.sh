#!/usr/bin/env bash

srun -p mm_human \
srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p pat_earth  \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=det_eval python -u test.py configs/mask_rcnn_tc_partpad_mta2_1x_coco.py work_dirs/mask_partpad/latest.pth \
    --work-dir=work_dirs/debug --launcher="slurm" --eval bbox segm


    --job-name=det_eval python -u test.py configs/mask_rcnn_tc_partpad2_mta_1x_coco.py work_dirs/mask_partpad/latest.pth \

    --job-name=det_eval python -u test.py configs/mask_rcnn_tc_partpad_mta_1x_coco.py work_dirs/mask_partpad2/latest.pth \

    --job-name=det_eval python -u test.py configs/mask_rcnn_pvt_s_fpn_1x_coco.py models/mask_rcnn_pvt_s_fpn_1x_coco.pth \



srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p mm_human \
srun -p mm_human --quotatype=auto\
srun -p pat_earth  \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
  --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
   --job-name=det python -u train.py --launcher="slurm" \
   configs/mask_rcnn_tc_partpad_hr2_1x_coco.py --work-dir=work_dirs/mask_partpad_hr2

    configs/mask_rcnn_tc_partpad_mta_1x_coco_2.py  --work-dir=work_dirs/mask_partpad_2

   configs/mask_rcnn_tc_partpad_hr_1x_coco.py --work-dir=work_dirs/mask_partpad_hr





   --job-name=det python -u train.py configs/mask_rcnn_tc_partpad_bimta_1x_coco.py \
    --work-dir=work_dirs/mask_partpad_bi --launcher="slurm"

    --job-name=det python -u train.py configs/mask_rcnn_tc_partpad_mta2_1x_coco.py \
    --work-dir=work_dirs/mask_partpad_mta2 --launcher="slurm" --resume-from=work_dirs/mask_partpad_mta2/latest.pth

   --job-name=det python -u train.py configs/mask_rcnn_tc_partpad_mta_1x_coco.py \
    --work-dir=work_dirs/mask_partpad --launcher="slurm"





    --job-name=det python -u train.py configs/retinanet_tc_partpad_small_mta_1x_coco.py \
    --work-dir=work_dirs/retina_partpad --launcher="slurm"



    --job-name=det_partpad python -u train.py configs/mask_rcnn_tc_partpad2_mta_1x_coco.py \
    --work-dir=work_dirs/mask_partpad2 --launcher="slurm"

    --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=det_partpad python -u train.py configs/mask_rcnn_tc_partpad2_mta_1x_coco.py \
    --work-dir=work_dirs/debug --launcher="slurm"

