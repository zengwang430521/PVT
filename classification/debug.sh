#!/usr/bin/env bash

srun -p 3dv-share  -w SH-IDC1-10-198-6-129\

-x SH-IDC1-10-198-4-[100-103,116-119] \

srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt4a_small --output_dir=work_dirs/my4a \
    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader

    python -u train.py --config configs/pvt_v2/my3.py \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my3/checkpoint.pth

    python -u train.py --config configs/pvt_v2/my4.py \
    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader

    python -u train.py --config configs/pvt_v2/pvt_v2_b2.py \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/my3.py \
    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my3/checkpoint.pth

    python -u train.py --model=mypvt20_2_small --output_dir work_dirs/debug  \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader\
    --resume work_dirs/my20_2_f2/checkpoint.pth  --eval






