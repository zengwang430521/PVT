srun -p pat_earth \
    --job-name=pvt --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u train.py --model pvt_small_impr8_peg --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/eval --data-path data/imagenet --resume=data/pretrained/pvt_small_impr8_peg.pth --eval


    python -u train.py --model pvt_small_impr1_peg --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/eval --data-path data/imagenet --resume=data/pretrained/pvt_small_impr1_peg.pth --eval

    python -u train.py --model mypvt2d_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/eval --data-path data/imagenet --resume=work_dirs/mypvt2_s_again/checkpoint_183.pth --eval

    python -u train.py --model mypvt2_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/eval --data-path data/imagenet --resume=work_dirs/mypvt2_s_again/checkpoint_183.pth --eval

    python -u train.py --model mypvt2d_small --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/eval --data-path data/imagenet --resume=work_dirs/mypvt2_s_again/checkpoint_183.pth --eval

    python -u train.py --model pvt_small_impr1_peg --batch-size 128 --epochs 300 --num_workers 5  --cache_mode \
    --output_dir ./work_dirs/eval --data-path data/imagenet --resume=data/pretrained/pvt_small_impr1_peg.pth --eval