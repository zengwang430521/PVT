srun -p 3dv-share -w SH-IDC1-10-198-6-129 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=pvt_s_d --kill-on-bad-exit=1
dist_train.sh configs/retinanet_mypvt2_s_fpn_1x_coco_640.py 8



srun -p pat_earth \
    --job-name=pvt --ntasks=1 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \




    dist_train.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py 8

    dist_train.sh configs/mypvt2_s_2.py 8


    python -u train.py configs/retinanet_mypvt2_s_fpn_1x_coco_640.py --work-dir=work_dirs/mypvt2_s_d --launcher="slurm"

    python -u train.py configs/mypvt2_s_2.py --work-dir=work_dirs/mypvt2_s_d2 --launcher="slurm"

    python -u train.py configs/mypvt2_s_3.py --work-dir=work_dirs/mypvt2_s_d3 --launcher="slurm"

    python -u train.py configs/mypvt2_s_4.py --work-dir=work_dirs/mypvt2_s_d4 --launcher="slurm"

    python -u train.py configs/retinanet_pvt_s_fpn_1x_coco_640.py --work-dir=work_dirs/pvt_s_d --launcher="slurm"

    python -u train.py configs/retinanet_mypvt2_s_fpn_1x_coco_640.py --work-dir=work_dirs/debug --launcher="slurm"

