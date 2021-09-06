#!/usr/bin/env bash

srun -p 3dv-share  -w SH-IDC1-10-198-6-129\
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u main.py --config configs/pvt/my20_2.py --output_dir work_dirs/debug  \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader\
    --resume work_dirs/my20_s2/my20_300_pre.pth  --eval

