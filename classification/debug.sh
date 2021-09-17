#!/usr/bin/env bash

export NCCL_LL_THRESHOLD=0
python -m torch.distributed.launch --nproc_per_node=8 --master_port=6333 --use_env \


    train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3c2_small --output_dir=work_dirs/my3c2_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112

    train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3b_small --output_dir=work_dirs/my3b_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112


srun -p 3dv-share  -w SH-IDC1-10-198-6-129\
srun -p mmpose \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human \
    --job-name=pvt --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f_small --output_dir=work_dirs/my3f_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f_LR/checkpoint.pth


    python -u train.py --config configs/pvt_v2/my3.py \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my3/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3d_small --output_dir=work_dirs/my3d_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3e_small --output_dir=work_dirs/my3e_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3c3_small --output_dir=work_dirs/my3c3_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3c_small --output_dir=work_dirs/my3c_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3b3_small --output_dir=work_dirs/my3b3_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3b2_small --output_dir=work_dirs/my3b2_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3a_small --output_dir=work_dirs/my3a \
    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader\

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt4a_small --output_dir=work_dirs/my4a \
    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader\
    --resume work_dirs/my4a/checkpoint.pth

    python -u train.py --config configs/pvt_v2/my3.py \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my3/checkpoint.pth

    python -u train.py --config configs/pvt_v2/my4.py \
    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader

    python -u train.py --config configs/pvt_v2/pvt_v2_b2.py \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader



    python -u train.py --model=mypvt20_2_small --output_dir work_dirs/debug  \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader\
    --resume work_dirs/my20_2_f2/checkpoint.pth  --eval






