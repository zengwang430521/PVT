import torch
from typing import List


# import torch
# device = torch.device('cuda')
# from utils_mine import index_points
#
# q = torch.zeros(2, 3136, 2, 32).to(device)
# k = torch.zeros(2, 64, 49, 2, 32).to(device)
# v = torch.zeros(2, 64, 49, 2, 32).to(device)
# idx = torch.rand(2, 3136).to(device) * 64
# idx = idx.long()
#
#
# B, N, H, C = q.shape
# B, W, K, H, C = k.shape
# step = max(N // K, 1)
# begin = 0
#
#
# idx_batch = torch.arange(B, device=q.device)[:, None]
# outs = []
# while begin < N:
#     end = min(begin + step, N)
#     q_t = q[:, begin:end]
#     idx_w = idx[:, begin:end]
#     n = idx_w.shape[1]
#     k_t = k[idx_batch.expand_as(idx_w).reshape(-1), idx_w.reshape(-1)].reshape(B, n, K, H, -1)
#     v_t = v[idx_batch.expand_as(idx_w).reshape(-1), idx_w.reshape(-1)].reshape(B, n, K, H, -1)
#
#     attn = torch.einsum("bnhc,bnkhc->bnhk", [q_t, k_t])
#     attn = attn.softmax(dim=-1)
#     # attn = attn_drop(attn)
#     out = torch.einsum("bnhk,bnkhc->bnhc", [attn, v_t])
#     out = out.flatten(-2)
#     outs.append(out)
#     begin = end
#
# outs = torch.cat(outs, dim=1)
# t = outs.shape
#
# t=0












from modules.cluster_block import ClusterBlock
import torch
import matplotlib.pyplot as plt
import utils_mine


device = torch.device('cpu')
H, W = 64, 48
loc_orig = utils_mine.get_grid_loc(1, H, W, device)
B, N0, _ = loc_orig.shape
# x_orig = torch.cat([loc_orig * 0.5 + 0.5, loc_orig.new_zeros(B, N0, 1)], dim=-1)
x_orig = torch.rand(B, N0, 8).to(device)
idx_agg = torch.arange(N0)[None, :].repeat(B, 1).to(device)
agg_weight_orig = x_orig.new_ones(B, N0, 1)

loc_down = torch.rand(1, 100, 2) * 2 - 1
loc_down = loc_down.to(device)
x, loc, idx_agg, weight_t = utils_mine.merge_tokens_agg(x_orig, loc_orig, loc_down, idx_agg, weight=None, return_weight=True)

agg_weight = agg_weight_orig * weight_t
agg_weight = agg_weight / agg_weight.max(dim=1, keepdim=True)[0]

input_dict = {
    'x': x,
    'loc_orig': loc_orig,
    'idx_agg': idx_agg,
    'agg_weight': agg_weight,
    'map_size': [H, W]
}

block = ClusterBlock(8, 8, [64, 48], 2)
out_dict = block(input_dict)


# determine grid
# determin_grid(self, idx_agg, agg_weight, loc_orig, window_grid)
from utils_mine import map2token_agg_sparse_nearest, token2map_agg_sparse

B, N0 = idx_agg.shape
N = x.shape[1]

window_grid = [2, 2]
h, w = window_grid
win_map = torch.eye(h*w, device=idx_agg.device)
win_map = win_map.reshape(1, h, w, h*w).expand(B, h, w, h*w).permute(0, 3, 1, 2)
win_weight = map2token_agg_sparse_nearest(win_map, N, loc_orig, idx_agg, agg_weight)
idx_win = win_weight.argmax(dim=-1)