#!/usr/bin/env bash

export NCCL_LL_THRESHOLD=0
python -m torch.distributed.launch --nproc_per_node=8 --master_port=6333 --use_env \
train.py --config configs/pvt_v2/debug.py \
    --batch-size 128 --data-path data/imagenet --input-size 112 \
    --model=mypvt3h7k3_small --output_dir=work_dirs/my3h7k3_LR --resume work_dirs/my3h7k3_LR/checkpoint.pth

    --model=mypvt3h10_small --output_dir=work_dirs/my3h10_LR --resume work_dirs/my3h10_LR/checkpoint.pth


srun -p 3dv-share  -w SH-IDC1-10-198-6-129\
srun -p mmpose \
srun -p mm_human \
srun -p pat_earth \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    --job-name=pvt --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --config configs/pvt_v2/debug.py \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader \
    --model=mypvt3h12_small --output_dir=work_dirs/my3h12_LR --resume work_dirs/my3h12_LR/checkpoint.pth


    --model=mypvt3h10_small --output_dir=work_dirs/my3h10_LR --resume work_dirs/my3h10_LR/checkpoint.pth

    --model=mypvt3h7k3_small --output_dir=work_dirs/my3h7k3_LR --resume work_dirs/my3h7k3_LR/checkpoint.pth

    --model=mypvt3h9_small --output_dir=work_dirs/my3h9_LR --resume work_dirs/my3h9_LR/checkpoint.pth

    --model=mypvt3h8_small --output_dir=work_dirs/my3h8_LR --resume work_dirs/my3h8_LR/checkpoint.pth

    --model=mypvt3h6_small --output_dir=work_dirs/my3h6_LR --resume work_dirs/my3h6_LR/checkpoint.pth

    --model=mypvt3h4_small --output_dir=work_dirs/my3h4_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3h4_LR/checkpoint.pth

    --model=mypvt3h2_1_small --output_dir=work_dirs/my3h2_1_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3h2_1_LR/checkpoint.pth

    --model=mypvt3h3_small --output_dir=work_dirs/my3h3_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3h3_LR/checkpoint.pth

    --model=mypvt3h2_small --output_dir=work_dirs/my3h2_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3h2_LR/checkpoint.pth

    --model=mypvt3h1_small --output_dir=work_dirs/my3h1_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3h1_LR/checkpoint.pth

    --model=mypvt3g1_small --output_dir=work_dirs/my3g1_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3g1_LR/checkpoint.pth

    --model=mypvt3h_small --output_dir=work_dirs/my3h_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3h_LR/checkpoint.pth

    --model=mypvt3f14_small --output_dir=work_dirs/my3f14_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f14_LR/checkpoint.pth

    --model=mypvt3f12_5_small --output_dir=work_dirs/my3f12_5_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f12_5_LR/checkpoint.pth

    --model=mypvt3f12_2_small --output_dir=work_dirs/my3f12_2new_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f12_2new_LR/checkpoint.pth

    --model=mypvt3f12_1_small --output_dir=work_dirs/my3f12_1new_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f12_1new_LR/checkpoint.pth

    --model=mypvt3f12_4_small --output_dir=work_dirs/my3f12_4_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f12_4_LR/checkpoint.pth

    --model=mypvt5f_small --output_dir=work_dirs/my5f_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my5f_LR/checkpoint.pth

    --model=mypvt3f12_3_small --output_dir=work_dirs/my3f12_3_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f12_3_LR/checkpoint.pth

    --model=mypvt3f12_2_small --output_dir=work_dirs/my3f12_2_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f12_2_LR/checkpoint.pth

    --model=mypvt3f12_1_small --output_dir=work_dirs/my3f12_1_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f12_1_LR/checkpoint.pth

   --model=mypvt4b4_small --output_dir=work_dirs/my4b4_LR \
   --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my4b4_LR/checkpoint.pth

   --model=mypvt4b3_small --output_dir=work_dirs/my4b3_LR \
   --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my4b3_LR/checkpoint.pth

   --model=mypvt4b2_small --output_dir=work_dirs/my4b2_LR \
    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my4b2_LR/checkpoint.pth

    --model=mypvt3f13_small --output_dir=work_dirs/my3f13_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f13_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f12_small --output_dir=work_dirs/my3f12_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f12_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f11_small --output_dir=work_dirs/my3f11_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f11_LR/checkpoint.pth

    --model=mypvt4b0_small --output_dir=work_dirs/my4b0_LR \
    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my4b0_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f10_small --output_dir=work_dirs/my3f10_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f10_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f9_small --output_dir=work_dirs/my3f9_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f9_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f8_small --output_dir=work_dirs/my3f8_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f8_LR/checkpoint.pth


    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f7_small --output_dir=work_dirs/my3f7_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f7_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f6_small --output_dir=work_dirs/my3f6_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f6_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3a0_small --output_dir=work_dirs/my3a0_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3a0_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f5_small --output_dir=work_dirs/my3f5 \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader --resume work_dirs/my3f5/checkpoint.pth

     python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt5c0_small --output_dir=work_dirs/my5c0_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt5b0_small --output_dir=work_dirs/my5b0_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f5_small --output_dir=work_dirs/my3f5_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f5_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3a1_small --output_dir=work_dirs/my3a1_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3a1_LR/checkpoint.pth

     python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt5c_small --output_dir=work_dirs/my5c_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my5c_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt5b_small --output_dir=work_dirs/my5b_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my5b_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3a2_small --output_dir=work_dirs/my3a2_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3_small --output_dir=work_dirs/my3_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f_small --output_dir=work_dirs/my3f_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt3f3_small --output_dir=work_dirs/my3f3_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my3f3_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt5_small --output_dir=work_dirs/my5_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my5_LR/checkpoint.pth



    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt5a_small --output_dir=work_dirs/my5a_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my5a_LR/checkpoint.pth

    python -u train.py --config configs/pvt_v2/debug.py \
    --model=mypvt5_small --output_dir=work_dirs/my5_LR \
    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader --resume work_dirs/my5_LR/checkpoint.pth

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






