sh dist_train.sh pvt_small 1 data/models --data-path /mnt/lustre/share/zengwang/imagenet/ --resume data/models/pvt_small_le1.5.pth --eval

sh dist_train.sh pvt_small 8 logs/pvt_s --data-path /mnt/lustre/share/zengwang/imagenet/