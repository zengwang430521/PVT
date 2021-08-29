srun -p 3dv-share -w SH-IDC1-10-198-6-129 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_s_d --kill-on-bad-exit=1
dist_train.sh configs/mypvt2_s_2.py 8

srun -p 3dv-share -w SH-IDC1-10-198-6-129 --gres=gpu:8 -n1 --ntasks-per-node=8 --cpus-per-task=5 --job-name=pvt_s_d --kill-on-bad-exit=1

srun -p 3dv-share -w SH-IDC1-10-198-6-130 --gres=gpu:2 -n1 --ntasks-per-node=2 --cpus-per-task=2 --job-name=pvt_s_d --kill-on-bad-exit=1 \
python -u train.py configs/debug_cfg.py --work-dir=work_dirs/debug --launcher="slurm"

srun -p pat_earth --gres=gpu:2 -n1 --ntasks-per-node=2 --cpus-per-task=2 --job-name=pvt_s_d --kill-on-bad-exit=1  \


srun -p pat_earth --gres=gpu:1 -n1 --ntasks-per-node=1 --cpus-per-task=1 --job-name=env --kill-on-bad-exit=1
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/1.3.1/torch1.6.0/cu101/mmcv_full-1.3.1%2Btorch1.6.0%2Bcu101-cp36-cp36m-manylinux1_x86_64.whl

srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_s_d --kill-on-bad-exit=1 \
dist_train.sh configs/mypvt2_s_2.py 8

srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=8 --ntasks=8 --job-name=pvt_s_d --kill-on-bad-exit=1 \
python -u train.py configs/mypvt2_s_2.py --work-dir=work_dirs/mypvt2_s_d2 --launcher="slurm"


srun -p pat_earth --gres=gpu:8 -n1 --ntasks-per-node=8 --ntasks=8 --job-name=pvt_s_d --kill-on-bad-exit=1 \
python -u train.py configs/mypvt2_s_5.py --work-dir=work_dirs/mypvt2_s_d5 --launcher="slurm"




srun -p pat_earth \
    -x SH-IDC1-10-198-4-[100-103,116-119] \
    --ntasks 1 --job-name=det \
    --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 --kill-on-bad-exit=1 \
    python -u train.py configs/my20_2.py --work-dir=work_dirs/my20_2_d1 --launcher="slurm"




    dist_train.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py 8

    dist_train.sh configs/mypvt2_s_2.py 8


    python -u train.py configs/retinanet_mypvt2_s_fpn_1x_coco_640.py --work-dir=work_dirs/mypvt2_s_d --launcher="slurm"

    python -u train.py configs/mypvt2_s_2.py --work-dir=work_dirs/mypvt2_s_d2 --launcher="slurm"

    python -u train.py configs/mypvt2_s_3.py --work-dir=work_dirs/mypvt2_s_d3 --launcher="slurm"

    python -u train.py configs/mypvt2_s_4.py --work-dir=work_dirs/mypvt2_s_d4 --launcher="slurm"

    python -u train.py configs/retinanet_pvt_s_fpn_1x_coco_640.py --work-dir=work_dirs/pvt_s_d --launcher="slurm"

    python -u train.py configs/retinanet_mypvt2_s_fpn_1x_coco_640.py --work-dir=work_dirs/debug --launcher="slurm"

