#!/usr/bin/env bash
    -n6 --gpu --job-name=mesh --gpu-type 16gv100  -w SH-IDC1-10-198-6-245 \
spring.submit arun -p spring_scheduler --gres=gpu:6 --ntasks-per-node=6 --cpus-per-task=5 \
    -n6 --gpu --job-name=hrpvt --gpu-type 16gv100 \
    "
    python -u train.py --config configs/pvt_v2/debug.py \
    --batch-size 170 --data-path data/imagenet --input-size 112  --use-mcloader \
    --model=myhrpvt_32_re --output_dir=work_dirs/myhrpvt_32_re_LR --resume work_dirs/myhrpvt_32_re_LR/checkpoint.pth
    "


srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    --job-name=pvt --ntasks=4 --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 \

srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human --quotatype=auto\
    --job-name=pvt --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --config configs/pvt_v2/debug.py \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader \
    --model=myhrpvt_32 --output_dir=debug

export NCCL_LL_THRESHOLD=0
python -m torch.distributed.launch --nproc_per_node=8 --master_port=6333 --use_env \
train.py --config configs/pvt_v2/debug.py \
    --batch-size 128 --data-path data/imagenet --input-size 112  \
    --model=hrformer_32 --output_dir=work_dirs/hrformer_32_LR --resume work_dirs/hrformer_32_LR/checkpoint.pth

    --batch-size 128 --data-path data/imagenet --input-size 112  \
    --model=myhrpvt_32_sr --output_dir=work_dirs/myhrpvt_32_sr_LR --resume work_dirs/myhrpvt_32_sr_LR/checkpoint.pth

    --batch-size 128 --data-path data/imagenet --input-size 112  \
    --model=myhrpvt_32_re --output_dir=work_dirs/myhrpvt_32_re_LR --resume work_dirs/myhrpvt_32_re_LR/checkpoint.pth


    --batch-size 128 --data-path data/imagenet --input-size 112  \
    --model=myhrpvt_32 --output_dir=work_dirs/myhrpvt_32_LR --resume work_dirs/myhrpvt_32_LR/checkpoint.pth


  --batch-size 128 --data-path data/imagenet --input-size 224  \
    --model=mypvt3h2_density0_tiny --output_dir=work_dirs/my3h2_density0_tiny --resume work_dirs/my3h2_density0_tiny/checkpoint.pth

    --batch-size 128 --data-path data/imagenet --input-size 112 \
    --model=mypvt3h2_densitya0_small --output_dir=work_dirs/dena0_LR --resume work_dirs/dena0_LR/checkpoint.pth

    --model=mypvt3h7k3_small --output_dir=work_dirs/my3h7k3_LR --resume work_dirs/my3h7k3_LR/checkpoint.pth

    --model=mypvt3h10_small --output_dir=work_dirs/my3h10_LR --resume work_dirs/my3h10_LR/checkpoint.pth


srun -p pat_earth \
    --job-name=pvt --ntasks=32 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --config configs/pvt_v2/debug.py \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader \
    --model=myhrpvt_32 --output_dir=work_dirs/debug --lr=3.5e-4

srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    --job-name=pvt --ntasks=16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --config configs/pvt_v2/debug.py \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader \
    --model=myhrpvt_32 --output_dir=work_dirs/myhrpvt_32_16 --resume work_dirs/myhrpvt_32_16/checkpoint.pth

    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader \
    --model=mypvt3h2_density0f_large --output_dir=work_dirs/my3h2_density0f_large_16 --resume work_dirs/my3h2_density0f_large_16/checkpoint.pth

    --model=mypvt3h2_density0f_tiny --output_dir=work_dirs/my3h2_density0f_tiny_16 --resume work_dirs/my3h2_density0f_tiny/checkpoint.pth

    --model=mypvt3h2_density0f_tiny --output_dir=work_dirs/my3h2_density0f_tiny --resume work_dirs/my3h2_density0f_tiny/checkpoint.pth

    --model=mypvt3h2_density0f_large --output_dir=work_dirs/my3h2_density0f_large_32 --resume work_dirs/my3h2_density0f_large_32/checkpoint.pth



    --model=mypvt3h2_density0_small --output_dir=work_dirs/my3h2_density0 --resume work_dirs/my3h2_density0/checkpoint.pth


    --model=mypvt3h2_density0_large --output_dir=work_dirs/my3h2_density0_large --resume work_dirs/my3h2_density0_large/checkpoint.pth

    --model=mypvt3h2_densityc_small --output_dir=work_dirs/my3h2_densityc_8 --resume work_dirs/my3h2_densityc/checkpoint.pth

    --model=mypvt3h2_densityc_small --output_dir=work_dirs/my3h2_densityc --resume work_dirs/my3h2_densityc/checkpoint.pth

    --model=mypvt3h2_densitya0_small --output_dir=work_dirs/my3h2_densitya0 --resume work_dirs/my3h2_densitya0/checkpoint.pth


    --model=mypvt3h2_fast_small --output_dir=work_dirs/my3h2_fast --resume work_dirs/my3h2_fast/checkpoint.pth

    --model=mypvt3h2_density25_small --output_dir=work_dirs/my3h2_density25 --resume work_dirs/my3h2_density25/checkpoint.pth

    --model=mypvt3h2_density0_small --output_dir=work_dirs/my3h2_density0 --resume work_dirs/my3h2_density0/checkpoint.pth



srun -p 3dv-share  -w SH-IDC1-10-198-6-129\
srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
    --job-name=pvt --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --config configs/pvt_v2/debug.py \
    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader \
    --model=den0f_large_fine --output_dir=work_dirs/den0f_large_fine --resume work_dirs/den0f_large_fine/checkpoint.pth \
    --finetune=work_dirs/tran_pvt_v2_b4.pth


    --batch-size 64 --data-path data/imagenet --input-size 224 --use-mcloader \
    --model=myhrpvt_32 --output_dir=work_dirs/myhrpvt_32 --resume work_dirs/myhrpvt_32/checkpoint.pth


    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader \
    --model=myhrpvt_32 --output_dir=work_dirs/myhrpvt_32_LR --resume work_dirs/myhrpvt_32_LR/checkpoint.pth


    --batch-size 128 --data-path data/imagenet --input-size 224 --use-mcloader \
    --model=mypvt3h2_density25_small --output_dir=work_dirs/my3h2_density25 --resume work_dirs/my3h2_density25/checkpoint.pth

    --model=mypvt3h2_density0_small --output_dir=work_dirs/my3h2_density0 --resume work_dirs/my3h2_density0/checkpoint.pth

    --model=mypvt3h2_fast_norm_small --output_dir=work_dirs/my3h2_fast_norm --resume work_dirs/my3h2_fast_norm/checkpoint.pth


    --model=mypvt3h2_fast_norm_small --output_dir=work_dirs/my3h2_fast_norm --resume work_dirs/my3h2_fast_norm/checkpoint.pth

    --model=mypvt3h2_fast2_small --output_dir=work_dirs/my3h2_fast2 --resume work_dirs/my3h2_fast2/checkpoint.pth


    --model=mypvt3h2_fast_small --output_dir=work_dirs/my3h2_fast --resume work_dirs/my3h2_fast/checkpoint.pth



    --model=mypvt3h2_small --output_dir=work_dirs/my3h2_8 --resume work_dirs/my3h2_8/checkpoint.pth


    --batch-size 128 --data-path data/imagenet --input-size 112 --use-mcloader \
    --model=mypvt3h2a_small --output_dir=work_dirs/my3h2a_LR --resume work_dirs/my3h2a_LR/checkpoint.pth

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






