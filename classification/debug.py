import torch
# import classification.utils_mine as utils_mine
import utils_mine as utils_mine


# data = torch.load('../../debug_block.pth')
# x = data['x']
# x_source = data['x_source']
# loc = data['loc']
# loc_source = data['loc_source']
# conf_source = data['conf_source']
# x1, x2 = data['x1'], data['x2']
# bid = x1.isnan().nonzero()[0, 0]
#
# x = x[bid, ...].unsqueeze(0)
# loc = loc[bid, ...].unsqueeze(0)
# x_source = x_source[bid, ...].unsqueeze(0)
# loc_source = loc_source[bid, ...].unsqueeze(0)
# conf_source = conf_source[bid, ...].unsqueeze(0)
#
#
# conf_source = conf_source.cuda()
# weight = conf_source.clamp(-7, 7).exp()
# tmp = utils_mine.merge_tokens(x_source.cuda(), loc_source.cuda(), loc.cuda(), weight)
#
# x_t, loc_t = tmp
#
# t = 0


# conf = conf_source
# weight = (conf - conf.min(dim=1, keepdim=True)[0]).float().exp().half()
# loc_down = loc
# loc = loc_source
# x = x_source
#
#
# B, N, C = x.shape
# Ns = loc_down.shape[1]
#
# dists = utils_mine.square_distance(loc, loc_down)
# idx = dists.argmin(axis=2)
# idx = idx + torch.arange(B)[:, None].to(loc.device) * Ns
#
# if weight is None:
#     weight = x.new_ones(B, N, 1)
# tmp = x.new_zeros(B * Ns, C + 3)
# source = torch.cat([x * weight, loc * weight, weight], dim=-1)
# source = source.to(x.device).type(x.dtype)
# tmp.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C + 3))
# tmp = tmp.reshape(B, Ns, C + 3)
#
# x_out = tmp[..., :C]
# loc_out = tmp[..., C:C + 2]
# norm_weight = tmp[:, :, C + 2:]
# x_out = x_out / (norm_weight + 1e-4)
# loc_out = loc_out / (norm_weight + 1e-4)



#
# import classification.utils_mine as utils_mine
# conf_map, mask = utils_mine.token2map(conf_source.float(), loc_source.float(), [3, 3], 1, 1, True)
# conf_map = conf_map[bid]
# mask = mask[bid]


import matplotlib.pyplot as plt
device = torch.device('cpu')
H, W = 64, 48
loc_orig = utils_mine.get_grid_loc(1, H, W, device)
B, N0, _ = loc_orig.shape
x_orig = torch.cat([loc_orig * 0.5 + 0.5, loc_orig.new_zeros(B, N0, 1)], dim=-1)
idx_agg = torch.arange(N0)[None, :].repeat(B, 1).to(device)
agg_weight_orig = x_orig.new_ones(B, N0, 1)

loc_down = torch.rand(1, 100, 2) * 2 - 1
loc_down = loc_down.to(device)
x, loc, idx_agg, weight_t = utils_mine.merge_tokens_agg(x_orig, loc_orig, loc_down, idx_agg, weight=None, return_weight=True)

agg_weight = agg_weight_orig * weight_t
agg_weight = agg_weight / agg_weight.max(dim=1, keepdim=True)[0]


x_map, weight_map = utils_mine.token2map_agg_mat(x, loc, loc_orig, idx_agg, [H, W])
plt.subplot(1, 2, 1)
plt.imshow(x_map[0].permute(1, 2, 0).detach().cpu())

x_map_re = x_map
for i in range(100):
    x_re = utils_mine.map2token_agg_fast_nearest(x_map_re, loc.shape[1], loc_orig, idx_agg, agg_weight)
    x_map_re, weight_map_re = utils_mine.token2map_agg_mat(x_re, loc, loc_orig, idx_agg, [H//2, W//2], agg_weight)
    plt.subplot(1, 2, 2)
    plt.imshow(x_map_re[0].permute(1, 2, 0).detach().cpu())


# x_map = x
#
# x_down, pos_down, idx_agg_down, weight_t = merge_tokens_agg(x, pos, pos_down, idx_agg, None, True)
# agg_weight_down = agg_weight * weight_t
# agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]
#
# x_map1, _ = token2map_agg_sparse(x, pos, pos_orig, idx_agg, [H, W])
# x_map2, _ = token2map_agg_sparse(x, pos, pos_orig, idx_agg, [H*2, W*2])
# x_map3, _ = token2map_agg_sparse(x_down, pos_down, pos_orig, idx_agg_down, [H, W])
#
# max = x_map[0, :3].max()
# min = x_map[0, :3].min()
# import matplotlib.pyplot as plt
# r, c = 2, 2
# plt.subplot(r,c,1)
# tmp = (x_map - min) / (max - min)
# plt.imshow(tmp[0, :3, :, :].permute(1,2,0).detach().cpu())
# plt.subplot(r,c,2)
# tmp = (x_map1 - min) / (max - min)
# plt.imshow(tmp[0, :3, :, :].permute(1,2,0).detach().cpu())
# plt.subplot(r,c,3)
# tmp = (x_map2 - min) / (max - min)
# plt.imshow(tmp[0, :3, :, :].permute(1,2,0).detach().cpu())
# plt.subplot(r,c,4)
# tmp = (x_map3 - min) / (max - min)
# plt.imshow(tmp[0, :3, :, :].permute(1,2,0).detach().cpu())
# plt.show()


# x_map[0, 0] - x_map2[0, 0]
# 
# plt.subplot(1,2,1)
# tmp = x_map - x_map2
# plt.imshow(tmp[0, 0, :, :].detach().cpu())
# plt.subplot(1,2,2)
# tmp = x_map / x_map2
# plt.imshow(tmp[0, 0, :, :].detach().cpu())
# plt.show()

# B, N, C = 2, 196, 64
# Ns = 49
# x = torch.rand(B, N, C)
#
# x_down = utils_mine.farthest_point_sample(x, Ns)
# x_down = x_down



# import torch
# from torch_cluster import fps
#
# x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
# batch = torch.tensor([0, 0, 0, 0])
# index = fps(x, batch, ratio=0.5, random_start=False)
#
# x = torch.rand(3, 100, 4)
# batch = torch.arange(3)[:, None].expand(3, 100)
# index = fps(x.flatten(0, 1), batch.flatten(0, 1), ratio=0.25)
# index = index.reshape(3, 25) - torch.arange(3)[:, None]
#

#
# import matplotlib.pyplot as plt
# device = torch.device('cpu')
# B, C, H, W = 2, 16, 64, 48
#
# x_map0 = torch.rand(B, C, H, W, device=device)
# x_map0 = utils_mine.guassian_filt(x_map0, kernel_size=3, sigma=2)
#
# plt.subplot(1, 3, 1)
# plt.imshow(x_map0[0, :3].permute(1, 2, 0).detach().cpu())
#
# x = x_map0.flatten(2).transpose(1, 2)
# N = x.shape[1]
# sample_num = N // 4
#
# loc_orig = utils_mine.get_grid_loc(B, H, W, device)
# index_down = utils_mine.farthest_point_sample(x, sample_num).unsqueeze(-1)
# x_down = torch.gather(x, 1, index_down.expand([B, sample_num, C]))
# x_down, A = utils_mine.merge_tokens_agg_dist_multi(x, index_down, x_down, None, k=1)
# Agg = A
#
#
# plt.subplot(1, 3, 1)
# plt.imshow(x_map0[0, :3].permute(1, 2, 0).detach().cpu())
# x_map = utils_mine.token2map_Agg(x_down, Agg, loc_orig, [H, W], weight=None)
# plt.subplot(1, 3, 2)
# plt.imshow(x_map[0, :3].permute(1, 2, 0).detach().cpu())
# x_map_re = x_map
#
# for i in range(100):
#     x_d_re = utils_mine.map2token_Agg(x_map_re, Agg, loc_orig)
#     x_map_re = utils_mine.token2map_Agg(x_d_re, Agg, loc_orig, [H, W], weight=None)
#     # x_map_re = utils_mine.guassian_filt(x_map_re, kernel_size=3, sigma=2)
#     plt.subplot(1, 3, 3)
#     plt.imshow(x_map_re[0, : 3].permute(1, 2, 0).detach().cpu())
#

# import torch
# from torch import nn
# fc = nn.Linear(4, 1)
# nn.init.constant_(fc.weight, 0)
# nn.init.constant_(fc.bias, 0)
#
# x = torch.rand(1, 4)
# y = fc(x).sum()
# l = -y
# l.backward()
#
#
# import torch
# from torch_sparse import spmm
#
# index = torch.tensor([[0, 0, 1, 2, 2],
#                       [0, 2, 1, 0, 1]])
# value = torch.Tensor([1, 2, 4, 1, 3]).requires_grad_().half()
# matrix = torch.Tensor([[1, 4], [2, 5], [3, 6]]).half()
#
# out = spmm(index, value, 3, 3, matrix)
# print(out)
# loss = out.sum()
# loss.backward()

from tc_module import tcformer_utils
from matplotlib import pyplot as plt
device = torch.device('cuda')
loc = tcformer_utils.get_grid_loc(2, 65, 65, device)
B = 2
N = 65*65
spatial_map = torch.arange(16).reshape(1, 1, 4, 4).expand(2, -1, -1, -1).to(device).float()
idx_agg = torch.arange(N)[None, :].repeat(B, 1).to(device)
spatial_idx = tcformer_utils.map2token(spatial_map, N, loc, idx_agg)
# tcformer_utils.get_spatial_neighbor(loc, 49)
spatial_idx_map = spatial_idx.reshape(B, 65, 65)
plt.imshow(spatial_idx_map[0].detach().cpu())

for i in range(16):
    print(i)
    print((spatial_idx[0, :, 0].round().long() == i).sum())
