srun -p 3dv-share -w SH-IDC1-10-198-6-129 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_s_d --kill-on-bad-exit=1
dist_train.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py 8


srun -p 3dv-share -w SH-IDC1-10-198-6-129 --gres=gpu:8 -n1 --ntasks-per-node=8 --job-name=pvts_c --kill-on-bad-exit=1
--ntasks-per-node=8 --cpus-per-task=4
sh dist_train.sh mypvt_small 8 ./work_dirs/mypvt_s --data-path /mnt/lustre/zengwang/data/imagenet

srun -p 3dv-share -w SH-IDC1-10-198-6-137 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvts_c --kill-on-bad-exit=1
--ntasks-per-node=1 --cpus-per-task=4 --use-mcloader
sh dist_train.sh pvt_small 8 ./work_dirs/pvt_s --data-path data/imagenet


srun -p 3dv-share -w SH-IDC1-10-198-6-129 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=test --kill-on-bad-exit=1
sh dist_train.sh mypvt_small 2 ./work_dirs/mypvt_s --data-path /mnt/lustre/zengwang/data/imagenet



srun -p pat_earth \
    --job-name=pvt \
    --gres=gpu:8 \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python -u train.py --model pvt_small --batch-size 128 --epochs 300 --num_workers 4  --cache_mode \
    --output_dir ./work_dirs/pvt_s_test --data-path data/imagenet


srun -p 3dv-share -w SH-IDC1-10-198-6-138 \
    --job-name=pvt \
    --gres=gpu:2 \
    --ntasks=2 \
    --ntasks-per-node=2 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python -u train.py --model pvt_small --batch-size 128 --epochs 300 --num_workers 5 --cache_mode \
    --output_dir ./work_dirs/pvt_s --data-path data/imagenet






srun -p 3dv-share -w SH-IDC1-10-198-6-129 \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt_small_2 --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/mypvt_s_2 --data-path data/imagenet \
    --resume work_dirs/mypvt_s_2/checkpoint.pth



srun -p 3dv-share -w SH-IDC1-10-198-6-129 \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt2_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/mypvt2_s --data-path data/imagenet


srun -p 3dv-share -w SH-IDC1-10-198-6-130 \
    --job-name=pvt --ntasks=4 \
    --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt2_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/debug --data-path data/imagenet


srun -p pat_earth \
    --job-name=pvt --ntasks=2 \
    --gres=gpu:2 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt2_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/debug --data-path data/imagenet


srun -p pat_earth \
    --job-name=pvt --ntasks=16 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt2_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/mypvt2_s --data-path data/imagenet





srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model pvt_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/debug --data-path data/imagenet


srun -p pat_earth \
    --job-name=pvt --ntasks=16 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt2_small --batch-size 128 --epochs 500 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/mypvt2_s2 --data-path data/imagenet --resume=work_dirs/mypvt2_s/checkpoint.pth


srun -p pat_earth \
    --job-name=pvt --ntasks=16 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt2_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/mypvt2_s_again --data-path data/imagenet

srun -p pat_earth \
    --job-name=pvt --ntasks=16 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model pvtcnn_small --batch-size 128 --epochs 500 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/pvtcnn_s --data-path data/imagenet



srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py configs/retinanet_mypvt2_s_fpn_1x_coco_640.py --work-dir=work_dirs/mypvt2_s_d --launcher="slurm"

srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py configs/mypvt2_s_2.py --work-dir=work_dirs/mypvt2_s_d2 --launcher="slurm"




srun -p pat_earth \
    --job-name=pvt --ntasks=2 \
    --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt3_small --batch-size 128 --epochs 50 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/debug --data-path data/imagenet


srun -p pat_earth \
--job-name=pvt --ntasks=2 \
--gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=1 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt4_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my4_s --data-path data/imagenet

    --exclude=SH-IDC1-10-198-4-84,SH-IDC1-10-198-4-117 \

srun -p pat_earth \
    --job-name=pvt --ntasks=16 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt3_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my3_s_debug --data-path data/imagenet --warmup-epochs=5


srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt3_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my3_s --data-path data/imagenet

srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt4_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my4_s --data-path data/imagenet

srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt5_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my5_s --data-path data/imagenet

srun -p 3dv-share \
    --job-name=pvt --ntasks=16 \
    --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt6_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my6_s --data-path data/imagenet --warmup-epochs=0


srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt6_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my6_s --data-path data/imagenet --lr=5e-5


srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt7_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my7_s1 --data-path data/imagenet --resume ./work_dirs/my7_s1/checkpoint.pth

srun -p pat_earth \
    --job-name=pvt --ntasks=16 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt8_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my8_s1 --data-path data/imagenet

srun -p pat_earth \
    -x SH-IDC1-10-198-4-100,SH-IDC1-10-198-4-101,SH-IDC1-10-198-4-102,SH-IDC1-10-198-4-103,SH-IDC1-10-198-4-116,SH-IDC1-10-198-4-117,SH-IDC1-10-198-4-118,SH-IDC1-10-198-4-119 \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model pvt3_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/p3_s --data-path data/imagenet

srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt16_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my16_s --data-path data/imagenet


srun -p pat_earth \
    -x SH-IDC1-10-198-4-100,SH-IDC1-10-198-4-101,SH-IDC1-10-198-4-102,SH-IDC1-10-198-4-103,SH-IDC1-10-198-4-116,SH-IDC1-10-198-4-117,SH-IDC1-10-198-4-118,SH-IDC1-10-198-4-119 \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt18_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my18_s --data-path data/imagenet --input-size 448  --warmup-epochs 0

    --warmup-epochs 0 --lr 1e-3




spring.submit arun -p spring_scheduler -n 1 --job-name=data unzip


spring.submit arun \
    -p spring_scheduler \
    -n 16 --gpu \
    --job-name=pvt \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
    "python -u train.py --model mypvt21_small --batch-size 64 --epochs 50 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my21_fine --data-path data/imagenet \
    --finetune work_dirs/my20_s2/my20_267.pth"

    "python -u train.py --model mypvt22_small --batch-size 64 --epochs 50 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my22_s --data-path data/imagenet \
    --finetune work_dirs/my20_s2/my20_267.pth"

  "python -u train.py --model mypvt22_small --batch-size 64 --epochs 50 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my21_s --data-path data/imagenet \
    --finetune work_dirs/my20_s2/my20_267.pth"


    "python -u train.py --model pvt_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/debug --data-path data/imagenet --input-size 448 \
    --resume ./work_dirs/debug/checkpoint.pth --warmup-epochs 0"

    "python -u train.py --model mypvt18_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my18_s --data-path data/imagenet --input-size 448 --warmup-epochs 0 --lr 1e-3 "

    "python -u train.py --model mypvt16_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/debug --data-path data/imagenet --warmup-epochs 0 --lr 1e-3 "






    -x SH-IDC1-10-198-4-100,SH-IDC1-10-198-4-101,SH-IDC1-10-198-4-102,SH-IDC1-10-198-4-103,SH-IDC1-10-198-4-116,SH-IDC1-10-198-4-117,SH-IDC1-10-198-4-118,SH-IDC1-10-198-4-119 \




srun -p pat_earth \
    -x SH-IDC1-10-198-4-[100-103,116-119] \
    --job-name=pvt --ntasks=16 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model mypvt23_small --batch-size 64 --epochs 50 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my21_fine --data-path data/imagenet \
    --input-size 448

    python -u train.py --model mypvt21_small --batch-size 64 --epochs 50 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my21_fine --data-path data/imagenet \
    --finetune work_dirs/my20_s2/my20_300.pth --input-size 448

    python -u train.py --model mypvt21_small --batch-size 64 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my21_s --data-path data/imagenet \
    --input-size 448 --finetune work_dirs/my20_s2/my20_150.pth --warmup-epochs 0

    python -u train.py --model mypvt18_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my18_s --data-path data/imagenet --input-size 448 --resume work_dirs/my18_s/checkpoint.pth

    python -u train.py --model mypvt10_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my10_s --data-path data/imagenet

    python -u train.py --model pvt4_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/p4_s --data-path data/imagenet

    python -u train.py --model pvt5_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/p5_s --data-path data/imagenet

    python -u train.py --model pvt6_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/p6_s --data-path data/imagenet

    python -u train.py --model mypvt11_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my11_s --data-path data/imagenet

    python -u train.py --model pvt7_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/p7_s --data-path data/imagenet

    python -u train.py --model mypvt12_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my12_s --data-path data/imagenet

    python -u train.py --model mypvt13_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/my13_s --data-path data/imagenet

    python -u train.py --model mypvt14_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir .work_dirs/my14_s --resume .work_dirs/my14_s/checkpoint.pth --data-path data/imagenet