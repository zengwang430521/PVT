sh dist_train.sh pvt_small 1 data/models --data-path /mnt/lustre/share/zengwang/imagenet/ --resume data/models/pvt_small_le1.5.pth --eval

sh dist_train.sh pvt_small 8 logs/pvt_s --data-path /mnt/lustre/share/zengwang/imagenet/

python main_for_debug.py --model=pvt_small --batch-size=1 --output_dir=data/models/ --data-path=/path/to/imagenet --resume=data/models/pvt_small_le1.5.pth --eval


srun -p $node \
    --gres=gpu:$numGPU \
    -n$allGPU \
    --ntasks-per-node=$numGPU \
    --job-name=$expID \
    --kill-on-bad-exit=1 \
    python



srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt --kill-on-bad-exit=1
python main_for_debug.py --model=pvt_small --batch-size=1 --output_dir=data/models/ --data-path=/path/to/imagenet --resume=data/models/pvt_small_le1.5.pth --eval


srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt --kill-on-bad-exit=1
sh dist_train.sh pvt_small 8 logs/pvt_small --data-path /mnt/lustre/zengwang/data/imagenet


GPUS_PER_NODE=8 ./slurm_train.sh HA_3D pvt --model=pvt_small --output_dir=logs/pvt_small --data-path /mnt/lustre/zengwang/data/imagenet


export NCCL_LL_THRESHOLD=0
srun -p HA_3D --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt --kill-on-bad-exit=1
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 --use_env main.py
 --model pvt_small --batch-size 64 --epochs 300 --output_dir=logs/pvt_small --data-path /mnt/lustre/zengwang/data/imagenet
