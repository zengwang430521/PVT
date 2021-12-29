#!/usr/bin/env bash
cd index_process
rm -rf dist build localAttention.egg-info
python setup_dist.py install

srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human \
srun -p mm_human --quotatype=auto\
    --job-name=install --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=2 --kill-on-bad-exit=1 \
    python setup_dist.py install
