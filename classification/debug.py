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


# import matplotlib.pyplot as plt
# device = torch.device('cpu')
# H, W = 64, 48
# loc_orig = utils_mine.get_grid_loc(1, H, W, device)
# B, N0, _ = loc_orig.shape
# x_orig = torch.cat([loc_orig * 0.5 + 0.5, loc_orig.new_zeros(B, N0, 1)], dim=-1)
# idx_agg = torch.arange(N0)[None, :].repeat(B, 1).to(device)
# agg_weight_orig = x_orig.new_ones(B, N0, 1)
#
# loc_down = torch.rand(1, 100, 2) * 2 - 1
# loc_down = loc_down.to(device)
# x, loc, idx_agg, weight_t = utils_mine.merge_tokens_agg(x_orig, loc_orig, loc_down, idx_agg, weight=None, return_weight=True)
#
# agg_weight = agg_weight_orig * weight_t
# agg_weight = agg_weight / agg_weight.max(dim=1, keepdim=True)[0]
#
#
# x_map, weight_map = utils_mine.token2map_agg_sparse(x, loc, loc_orig, idx_agg, [H, W])
# plt.subplot(1, 2, 1)
# plt.imshow(x_map[0].permute(1, 2, 0).detach().cpu())
#
# x_map_re = x_map
# for i in range(100):
#     x_re = utils_mine.map2token_agg_mat(x_map_re, loc, loc_orig, idx_agg)
#     x_map_re, weight_map_re = utils_mine.token2map_agg_sparse(x_re, loc, loc_orig, idx_agg, [H//2, W//2], agg_weight)
#     plt.subplot(1, 2, 2)
#     plt.imshow(x_map_re[0].permute(1, 2, 0).detach().cpu())


x_map = x

x_down, pos_down, idx_agg_down, weight_t = merge_tokens_agg(x, pos, pos_down, idx_agg, None, True)
agg_weight_down = agg_weight * weight_t
agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

x_map1, _ = token2map_agg_sparse(x, pos, pos_orig, idx_agg, [H, W])
x_map2, _ = token2map_agg_sparse(x, pos, pos_orig, idx_agg, [H*2, W*2])
x_map3, _ = token2map_agg_sparse(x_down, pos_down, pos_orig, idx_agg_down, [H, W])

max = x_map[0, :3].max()
min = x_map[0, :3].min()
import matplotlib.pyplot as plt
r, c = 2, 2
plt.subplot(r,c,1)
tmp = (x_map - min) / (max - min)
plt.imshow(tmp[0, :3, :, :].permute(1,2,0).detach().cpu())
plt.subplot(r,c,2)
tmp = (x_map1 - min) / (max - min)
plt.imshow(tmp[0, :3, :, :].permute(1,2,0).detach().cpu())
plt.subplot(r,c,3)
tmp = (x_map2 - min) / (max - min)
plt.imshow(tmp[0, :3, :, :].permute(1,2,0).detach().cpu())
plt.subplot(r,c,4)
tmp = (x_map3 - min) / (max - min)
plt.imshow(tmp[0, :3, :, :].permute(1,2,0).detach().cpu())
plt.show()


# x_map[0, 0] - x_map2[0, 0]
# 
# plt.subplot(1,2,1)
# tmp = x_map - x_map2
# plt.imshow(tmp[0, 0, :, :].detach().cpu())
# plt.subplot(1,2,2)
# tmp = x_map / x_map2
# plt.imshow(tmp[0, 0, :, :].detach().cpu())
# plt.show()