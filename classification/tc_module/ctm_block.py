import torch
import torch.nn as nn
import math
from .tcformer_utils import (
    token2map, map2token,
    token_cluster_merge, token_cluster_hir, token_cluster_dpc_hir,
    )

# CTM block
class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block,
                 k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25, conf_density=False):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out

        self.block = down_block
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.gumble_sigmoid = GumbelSigmoid()
        # temperature of confidence weight
        self.register_buffer('T', torch.tensor(1.0, dtype=torch.float))
        self.T_min = 1
        self.T_decay = 0.9998
        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        # self.conv = PartialConv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k
        self.dist_assign = dist_assign
        self.ada_dc = ada_dc
        self.use_conf = use_conf
        self.conf_scale = conf_scale
        self.conf_density = conf_density

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        x_down, idx_agg_down, weight_t = token_cluster_merge(
            x, sample_num, idx_agg, weight, True, k=self.k
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down


class CTM_hir(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block,
                 k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25, conf_density=False):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out

        self.block = down_block
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.gumble_sigmoid = GumbelSigmoid()
        # temperature of confidence weight
        self.register_buffer('T', torch.tensor(1.0, dtype=torch.float))
        self.T_min = 1
        self.T_decay = 0.9998
        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        # self.conv = PartialConv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k
        self.dist_assign = dist_assign
        self.ada_dc = ada_dc
        self.use_conf = use_conf
        self.conf_scale = conf_scale
        self.conf_density = conf_density

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        x_down, idx_agg_down, weight_t = token_cluster_hir(
            x, sample_num, idx_agg, conf, weight=weight, return_weight=True
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down


class CTM_dpchir(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block,
                 k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25, conf_density=False):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out

        self.block = down_block
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.gumble_sigmoid = GumbelSigmoid()
        # temperature of confidence weight
        self.register_buffer('T', torch.tensor(1.0, dtype=torch.float))
        self.T_min = 1
        self.T_decay = 0.9998
        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        # self.conv = PartialConv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k
        self.dist_assign = dist_assign
        self.ada_dc = ada_dc
        self.use_conf = use_conf
        self.conf_scale = conf_scale
        self.conf_density = conf_density

    def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)

        x_down, idx_agg_down, weight_t = token_cluster_dpc_hir(
            x, sample_num, idx_agg, weight=weight, return_weight=True, k=self.k,
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, loc_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        return x_down, idx_agg_down, agg_weight_down
