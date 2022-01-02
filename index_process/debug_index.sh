#!/usr/bin/env bash
rm -rf build dist localAttention.egg-info
srun -p mm_human \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=debug  python setup_dist.py install
