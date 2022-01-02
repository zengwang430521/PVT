import torch
from mmdet.models import build_detector
import copy
from mmcv import Config
import pvt_v2
import tc_module.tcformer_partpad
import tc_module.mta_head

# cfg_file = 'configs/mask_rcnn_pvt_v2_b2_fpn_1x_coco.py'
# cfg_file = 'configs/mask_rcnn_tc_partpad2_mta_1x_coco.py'
# cfg_file = 'configs/mask_rcnn_tc_partpad_mta_1x_coco.py'
cfg_file = 'configs/mask_rcnn_tc_partpad_bimta_1x_coco.py'

device = torch.device('cuda')
img_scale = (667, 400)

cfg = Config.fromfile(cfg_file)
model = cfg.model
model = build_detector(model).to(device)
# model_dict = model.state_dict()
x = torch.zeros([1, 3, img_scale[1], img_scale[0]]).to(device)

out = model.backbone(x)
out = model.neck(out)
t = 0




# import torch
# from tc_module import tcformer_utils
#
# B, H, W, C = 2, 167, 165, 4
# N = H * W
# device = torch.device('cuda')
# x = torch.rand([B, H*W, C], device=device)
#
# idx_agg = torch.arange(N)[None, :].repeat(B, 1).to(device)
# agg_weight = x.new_ones(B, N, 1)
# loc_orig = tcformer_utils.get_grid_loc(B, H, W, device)
# input_dict = {'x': x,
#              'map_size': [H, W],
#              'loc_orig': loc_orig,
#              'idx_agg': idx_agg,
#              'agg_weight': agg_weight}
#
# Ns = round(N * 0.25)
# tcformer_utils.token_cluster_part_pad(input_dict, Ns, nh_list=[8, 4, 2], nw_list=[8, 4, 2])
