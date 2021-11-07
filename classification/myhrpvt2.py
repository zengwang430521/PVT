# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# import math
# from .pvt_v2 import (Block, DropPath, DWConv, OverlapPatchEmbed, trunc_normal_, Attention, Mlp)

'''
low complexity in clustering
'''

# import pdb
import torch
import torch.nn as nn
from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    # constant_init,
    # kaiming_init,
    # normal_init,
)
# from mmcv.runner import load_checkpoint
# from mmcv.runner.checkpoint import load_state_dict
from mmcv.utils.parrots_wrapper import _BatchNorm

# from mmpose.models.utils.ops import resize
# from .modules.transformer_block import MlpDWBN

# from mmpose.utils import get_root_logger
# from ..builder import BACKBONES
# from .modules.bottleneck_block import Bottleneck

from utils_mine import get_merge_way, \
    index_points, \
    get_grid_loc, \
    downup

from pvt_v2_3h2_density_f import (map2token_agg_fast_nearest, token2map_agg_mat, MyAttention,
                                MyMlp, DropPath, trunc_normal_,
                                  register_model, _cfg)

from utils_mine import token_cluster_density_fixbug_sr as token_cluster_density
from utils_mine import downup_sparse as downup
from utils_mine import token2map_agg_sparse as token2map_agg_mat
from utils_mine import map2token_agg_sparse_nearest as map2token_agg_fast_nearest

import torch.nn.functional as F
import os
import logging
import torch.nn as nn
import math

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckDWP(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckDWP, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=planes,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class MyBlock(nn.Module):
    expansion=1
    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False,
                 conv_cfg=None,
                 norm_cfg=dict(type="BN", requires_grad=True)
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear, out_features=dim_out)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input_dict, src_dict=None):
        if src_dict is None:
            src_dict = input_dict

        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        x_source = src_dict['x']
        idx_agg_source = src_dict['idx_agg']
        if 'conf_source' in src_dict.keys():
            conf_source = src_dict['conf_source']
        else:
            conf_source=None

        x1 = x + self.drop_path(self.attn(self.norm1(x),
                                          loc_orig,
                                          self.norm1(x_source),
                                          idx_agg_source,
                                          H, W, conf_source))

        x2 = x1 + self.drop_path(self.mlp(self.norm2(x1),
                                          loc_orig,
                                          idx_agg,
                                          agg_weight,
                                          H, W))
        out_dict = {
            'x': x2,
            'idx_agg': idx_agg,
            'agg_weight': agg_weight,
            'x_source': x2,
            'idx_agg_source': idx_agg,
            'agg_weight_source': agg_weight,
            'loc_orig': loc_orig,
            'map_size': (H, W),
            'conf_source': None,
        }
        return out_dict


# from partialconv2d import PartialConv2d
class DownLayer(nn.Module):
    """ Down sample
    """

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

    def forward(self, input_dict):
        x=input_dict['x']
        pos_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']
        N_grid = 0


        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        if N0 == N and N == H * W:
            x_map = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            x_map, _ = token2map_agg_mat(x, None, pos_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token_agg_fast_nearest(x_map, N, pos_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        _, _, H, W = x_map.shape
        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio) - N_grid, 0)
        if sample_num < N_grid:
            sample_num = N_grid

        sr_ratio = self.block.attn.sr_ratio
        ht, wt = H * 2 // sr_ratio, W * 2 // sr_ratio


        x_down, idx_agg_down, weight_t = token_cluster_density(
            x, sample_num, idx_agg, weight, True, conf,
            k=self.k, dist_assign=self.dist_assign, ada_dc=self.ada_dc,
            use_conf=self.use_conf, conf_scale=self.conf_scale, conf_density=self.conf_density,
            map_size=[ht, wt], agg_weight=agg_weight,
            loc_orig=pos_orig if sr_ratio > 1 else None
        )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]


        tmp_dict = input_dict.copy()
        tmp_dict['x'] = x
        tmp_dict['conf_source'] = conf

        down_dict = {
            'x': x_down,
            'idx_agg': idx_agg_down,
            'agg_weight': agg_weight_down,
            'map_size': [H, W],
            'loc_orig': input_dict['loc_orig'],
            'conf_source': None
        }

        down_dict = self.block(down_dict, tmp_dict)
        return down_dict




# class MyBlock(BaseBlock):
#
#     def forward(self, input_dict, src_dict=None):
#         if src_dict is None:
#             src_dict = input_dict
#
#         x = input_dict['x']
#         loc_orig = input_dict['loc_orig']
#         idx_agg = input_dict['idx_agg']
#         agg_weight = input_dict['agg_weight']
#         H, W = input_dict['map_size']
#
#         x_source = src_dict['x']
#         idx_agg_source = src_dict['idx_agg']
#         if 'conf_source' in src_dict.keys():
#             conf_source = src_dict['conf_source']
#         else:
#             conf_source=None
#
#         x1 = x + self.drop_path(self.attn(self.norm1(x),
#                                           loc_orig,
#                                           self.norm1(x_source),
#                                           idx_agg_source,
#                                           H, W, conf_source))
#
#         x2 = x1 + self.drop_path(self.mlp(self.norm2(x1),
#                                           loc_orig,
#                                           idx_agg,
#                                           agg_weight,
#                                           H, W))
#         out_dict = {
#             'x': x2,
#             'idx_agg': idx_agg,
#             'agg_weight': agg_weight,
#             'x_source': x2,
#             'idx_agg_source': idx_agg,
#             'agg_weight_source': agg_weight,
#             'loc_orig': loc_orig,
#             'map_size': (H, W),
#             'conf_source': None,
#         }
#         return out_dict

# class MyMlpBN(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False,
#                  norm_cfg=dict(type="BN", requires_grad=True),
#                  dw_act_layer=nn.GELU
#                  ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.norm1 = TokenNorm(build_norm_layer(norm_cfg, hidden_features))
#         self.act1 = act_layer()
#
#         self.dwconv = MyDWConv(hidden_features)
#         self.act2 = dw_act_layer()
#         self.norm2 = TokenNorm(build_norm_layer(norm_cfg, hidden_features))
#
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.act3 = act_layer()
#         self.norm3 = TokenNorm(build_norm_layer(norm_cfg, out_features))
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, loc_orig, idx_agg, agg_weight, H, W):
#         x = self.fc1(x)
#         x = self.norm1(x)
#         x = self.act1(x)
#         x = self.dwconv(x, loc_orig, idx_agg, agg_weight, H, W)
#         x = self.norm2(x)
#         x = self.act2(x)
#         x = self.fc2(x)
#         x = self.norm3(x)
#         x = self.act3(x)
#         return x

# class MyBlockBN(nn.Module):
#     expansion = 1
#
#     def __init__(
#             self,
#             inplanes,
#             planes,
#             num_heads,
#             mlp_ratio=4.,
#             qkv_bias=False,
#             qk_scale=None,
#             drop=0.,
#             attn_drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             sr_ratio=1,
#             linear=False,
#             conv_cfg=None,
#             norm_cfg=dict(type="BN", requires_grad=True)
#     ):
#         super().__init__()
#         self.dim = inplanes
#         self.out_dim = planes
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#
#         self.norm1 = norm_layer(self.dim)
#         self.attn = MyAttention(
#             self.dim,
#             num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
#
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(self.dim)
#         mlp_hidden_dim = int(self.dim * mlp_ratio)
#         self.mlp = MyMlpBN(
#             in_features=self.dim,
#             out_features=self.out_dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#             linear=linear,
#             norm_cfg=norm_cfg
#         )
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, input_dict, src_dict=None):
#         if src_dict is None:
#             src_dict = input_dict
#
#         x = input_dict['x']
#         loc_orig = input_dict['loc_orig']
#         idx_agg = input_dict['idx_agg']
#         agg_weight = input_dict['agg_weight']
#         H, W = input_dict['map_size']
#
#         x_source = src_dict['x']
#         idx_agg_source = src_dict['idx_agg']
#         if 'conf_source' in src_dict.keys():
#             conf_source = src_dict['conf_source']
#         else:
#             conf_source=None
#
#         x1 = x + self.drop_path(self.attn(self.norm1(x),
#                                           loc_orig,
#                                           self.norm1(x_source),
#                                           idx_agg_source,
#                                           H, W, conf_source))
#
#         x2 = x1 + self.drop_path(self.mlp(self.norm2(x1),
#                                           loc_orig,
#                                           idx_agg,
#                                           agg_weight,
#                                           H, W))
#         out_dict = {
#             'x': x2,
#             'idx_agg': idx_agg,
#             'agg_weight': agg_weight,
#             'x_source': x2,
#             'idx_agg_source': idx_agg,
#             'agg_weight_source': agg_weight,
#             'loc_orig': loc_orig,
#             'map_size': (H, W),
#             'conf_source': None,
#         }
#         return out_dict



class TokenConv(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)

    def forward(self, input_dict):
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        x_map, _ = token2map_agg_mat(x, None, loc_orig, idx_agg, [H, W])
        x_map = super().forward(x_map)
        x = map2token_agg_fast_nearest(x_map, x.shape[1], loc_orig, idx_agg, agg_weight) + \
            self.skip(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class FuseLink(TokenConv):
    def __init__(self, kwargs):
        norm_cfg = kwargs.pop('norm_cfg')
        super().__init__(**kwargs)
        self.norm = build_norm_layer(norm_cfg, kwargs['out_channels'])[1]

    def forward(self, tar_dict, src_dict):
        x = super().forward(src_dict)
        x = self.norm(x.permute(0, 2, 1).unsqueeze(-1)).flatten(2).permute(0, 2, 1)
        tmp_dict = {
            'x': x,
            'idx_agg': src_dict['idx_agg']
        }
        x = downup(tar_dict, tmp_dict)
        return x


class TokenNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.name = norm[0]
        self.norm = norm[1]

    def forward(self, x):
        if 'ln' in self.name:
            x = self.norm(x)
        else:
            x = self.norm(x.permute(0, 2, 1).unsqueeze(-1)).flatten(2).permute(0, 2, 1)
        return x


# one step for multi-level sampling
class FuseLayer_Straight(nn.Module):
    def __init__(
            self,
            num_branches,
            in_channels,
            multiscale_output,
            norm_cfg,
            remerge=False,
    ):
        super().__init__()
        self.remerge = remerge
        self.norm_cfg = norm_cfg
        self.num_branches = num_branches
        self.num_out_branches = num_branches if multiscale_output else 1
        fuse_layers = []
        for i in range(self.num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j < i:
                    # down sample
                    fuse_link = nn.Sequential(
                        TokenConv(
                            in_channels=in_channels[j],
                            out_channels=in_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            groups=in_channels[j],
                            bias=False),
                        TokenNorm(build_norm_layer(self.norm_cfg, in_channels[i]))
                    )

                elif j == i:
                    # sample stage
                    fuse_link = None
                else:
                    # upsample
                    fuse_link = DictLayer(
                        nn.Sequential(
                            nn.Linear(in_channels[j], in_channels[i], bias=False),
                            TokenNorm(build_norm_layer(self.norm_cfg, in_channels[i]))),
                        input_decap=True,
                        output_cap=False,
                    )
                fuse_layer.append(fuse_link)
            fuse_layer = nn.ModuleList(fuse_layer)
            fuse_layers.append(fuse_layer)
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_lists):
        assert len(input_lists) == self.num_branches
        out_lists = []
        for i in range(self.num_out_branches):
            tar_dict = input_lists[i]
            x = tar_dict['x']
            idx_agg = tar_dict['idx_agg']
            agg_weight = tar_dict['agg_weight']

            if self.remerge and i > 0:
                # merge again
                with torch.no_grad():
                    idx_agg, agg_weight = get_merge_way(out_lists[i-1], tar_dict['x'].shape[1])
                x = downup({'x': x, 'idx_agg': idx_agg, 'agg_weight': agg_weight}, tar_dict)
                pass


            out_dict = {
                'x': x,
                'idx_agg': idx_agg,
                'agg_weight': agg_weight,
                'map_size': tar_dict['map_size'],
                'loc_orig': tar_dict['loc_orig']
            }

            for j in range(self.num_branches):
                if j != i:
                    src_dict = input_lists[j]
                    x_t = self.fuse_layers[i][j](src_dict)
                    x_t = downup(out_dict, {'x': x_t, 'idx_agg': src_dict['idx_agg']})
                    x = x + x_t

            out_dict['x'] = self.relu(x)
            out_lists.append(out_dict)
        return out_lists


# class DownLayer(BaseDown):
#     """ Down sample
#     """
#     def forward(self, input_dict):
#         H, W = input_dict['map_size']
#         idx_agg = input_dict['idx_agg']
#         agg_weight = input_dict['agg_weight']
#         x = input_dict['x']
#         loc_orig = input_dict['loc_orig']
#
#         x_down, idx_agg_down, agg_weight_down = super().forward(x, loc_orig, idx_agg, agg_weight, H, W, N_grid=0)
#
#         down_dict = {
#             'x': x_down,
#             'idx_agg': idx_agg_down,
#             'agg_weight': agg_weight_down,
#             'map_size': [H, W],
#             'loc_orig': input_dict['loc_orig'],
#             'conf_source': None
#         }
#         return down_dict


class MyModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        in_channels,
        num_channels,
        multiscale_output,
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        num_heads=None,
        sr_ratios=None,
        num_mlp_ratios=None,
        drop_paths=0.0,
        remerge=False,
    ):
        super().__init__()
        self._check_branches(num_branches, num_blocks, in_channels, num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches
        self.remerge = remerge

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_heads,
            sr_ratios,
            num_mlp_ratios,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()

        # MHSA parameters
        self.num_heads = num_heads
        self.num_mlp_ratios = num_mlp_ratios

    def _check_branches(self, num_branches, num_blocks, in_channels, num_channels):
        # logger = get_root_logger()
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = "NUM_BRANCHES({}) <> IN_CHANNELS({})".format(
                num_branches, len(in_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_heads,
        sr_ratios,
        num_mlp_ratios,
        drop_paths,
        stride=1,
    ):
        """Make one branch."""
        downsample = None
        if (
            stride != 1
            or self.in_channels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(
                    self.norm_cfg, num_channels[branch_index] * block.expansion
                )[1],
            )

        layers = []

        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                num_heads=num_heads[branch_index],
                sr_ratio=sr_ratios[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                drop_path=drop_paths[0],
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
            )
        )
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    sr_ratio=sr_ratios[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
    ):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None
        return FuseLayer_Straight(
            self.num_branches,
            self.in_channels,
            self.multiscale_output,
            remerge=self.remerge,
            norm_cfg=self.norm_cfg
        )

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = self.fuse_layers(x)
        return x_fuse


class DictLayer(nn.Module):
    def __init__(self, layer, input_decap=False, output_cap=True):
        super().__init__()
        self.layer = layer
        self.input_decap = input_decap
        self.output_cap = output_cap

    def forward(self, input_dict):
        if self.input_decap:
            x = self.layer(input_dict['x'])
        else:
            x = self.layer(input_dict)

        if self.output_cap:
            out_dict = input_dict.copy()
            out_dict['x'] = x
            return out_dict
        else:
            return x


class MyHRPVT(nn.Module):

    blocks_dict = {
        "BOTTLENECK": Bottleneck,
        "MYBLOCK": MyBlock,
        # "MYBLOCKBN": MyBlockBN,

    }

    def __init__(
        self,
        extra,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
        return_map=False,
        num_classes=1000
    ):
        super().__init__()
        self.return_map = return_map
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            self.conv_cfg, 64, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # generat drop path rate list
        depth_s2 = (
            self.extra["stage2"]["num_blocks"][0] * self.extra["stage2"]["num_modules"]
        )
        depth_s3 = (
            self.extra["stage3"]["num_blocks"][0] * self.extra["stage3"]["num_modules"]
        )
        depth_s4 = (
            self.extra["stage4"]["num_blocks"][0] * self.extra["stage4"]["num_modules"]
        )
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = self.extra["drop_path_rate"]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.dpr = dpr
        self.depth = depths

        # logger = get_root_logger()
        # logger.info(dpr)

        # stage 1
        self.stage1_cfg = self.extra["stage1"]
        num_channels = self.stage1_cfg["num_channels"][0]
        block_type = self.stage1_cfg["block"]
        num_blocks = self.stage1_cfg["num_blocks"][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra["stage2"]
        num_channels = self.stage2_cfg["num_channels"]
        block_type = self.stage2_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channels], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_paths=dpr[0:depth_s2]
        )

        # stage 3
        self.stage3_cfg = self.extra["stage3"]
        num_channels = self.stage3_cfg["num_channels"]
        block_type = self.stage3_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            drop_paths=dpr[depth_s2 : depth_s2 + depth_s3],
        )

        # stage 4
        self.stage4_cfg = self.extra["stage4"]
        num_channels = self.stage4_cfg["num_channels"]
        block_type = self.stage4_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=self.stage4_cfg.get("multiscale_output", False),
            drop_paths=dpr[depth_s2 + depth_s3 :],
        )

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(
            pre_stage_channels
        )

        self.classifier = nn.Linear(2048, num_classes)


    def _make_head(self, pre_stage_channels):
        head_block = BottleneckDWP
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                ),
                nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer


    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_block(self, branch_index):
        cfg = getattr(self, f'stage{branch_index+1}_cfg')
        block = self.blocks_dict[cfg['block']]
        depth = 0
        for i in range(branch_index-1):
            depth += self.depth[i]

        block = block(
            cfg['num_channels'][branch_index],
            cfg['num_channels'][branch_index],
            num_heads=cfg['num_heads'][branch_index],
            sr_ratio=cfg['sr_ratios'][branch_index],
            mlp_ratio=cfg['num_mlp_ratios'][branch_index],
            drop_path=self.dpr[depth],
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
        )
        return block

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # only change channels
                    transition_layers.append(
                        DictLayer(
                            nn.Sequential(
                                TokenConv(
                                    in_channels=num_channels_pre_layer[i],
                                    out_channels=num_channels_cur_layer[i],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False),
                                TokenNorm(build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])),
                                nn.ReLU(inplace=True),
                            )
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                # down layers
                down_layers = DownLayer(
                    embed_dim=num_channels_pre_layer[-1],
                    dim_out=num_channels_cur_layer[i],
                    drop_rate=0,
                    sample_ratio=0.25,
                    down_block=self._make_block(i)
                )
                transition_layers.append(down_layers)

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        stride=1,
        num_heads=1,
        sr_ratio=1,
        mlp_ratio=4.0,
    ):
        """Make each layer."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1],
            )

        layers = []

        if block == Bottleneck or block == BottleneckDWP:
            layers.append(
                block(
                    inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    # with_cp=self.with_cp,
                    # norm_cfg=self.norm_cfg,
                    # conv_cfg=self.conv_cfg,
                )
            )
        else:
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads=num_heads,
                    sr_ratio=sr_ratio,
                    mlp_ratio=mlp_ratio,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    # with_cp=self.with_cp,
                    # norm_cfg=self.norm_cfg,
                    # conv_cfg=self.conv_cfg,
                )
            )

        return nn.Sequential(*layers)

    def _make_stage(
        self, layer_config, in_channels, multiscale_output=True, drop_paths=0.0
    ):
        """Make each stage."""
        remerge = layer_config["remerge"]
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = self.blocks_dict[layer_config["block"]]

        num_heads = layer_config["num_heads"]
        sr_ratios = layer_config["sr_ratios"]

        num_mlp_ratios = layer_config["num_mlp_ratios"]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                MyModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    num_heads=num_heads,
                    sr_ratios=sr_ratios,
                    num_mlp_ratios=num_mlp_ratios,
                    drop_paths=drop_paths[num_blocks[0] * i : num_blocks[0] * (i + 1)],
                    remerge=remerge[i]
                )
            )

        return nn.Sequential(*hr_modules), in_channels

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.
    #
    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #         Defaults to None.
    #     """
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         ckpt = load_checkpoint(self, pretrained, strict=False)
    #         if "model" in ckpt:
    #             msg = self.load_state_dict(ckpt["model"], strict=False)
    #             logger.info(msg)
    #     elif pretrained is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 """mmseg: kaiming_init(m)"""
    #                 normal_init(m, std=0.001)
    #             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
    #                 constant_init(m, 1)
    #
    #         if self.zero_init_residual:
    #             for m in self.modules():
    #                 if isinstance(m, Bottleneck):
    #                     constant_init(m.norm3, 0)
    #                 # elif isinstance(m, BasicBlock):
    #                 #     constant_init(m.norm2, 0)
    #     else:
    #         raise TypeError("pretrained must be a str or None")

    def init_weights(
        self,
        pretrained="",
    ):
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info("=> loading pretrained model {}".format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }
            for k, _ in pretrained_dict.items():
                logger.info("=> loading {} pretrained model {}".format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def init_dict(self, x):
        B, C, H, W = x.shape
        device = x.device
        x = x.flatten(2).permute(0, 2, 1)
        loc_orig = get_grid_loc(B, H, W, device)
        B, N, _ = x.shape
        idx_agg = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        out_dict = {
            'x': x,
            'idx_agg': idx_agg,
            'agg_weight': agg_weight,
            'loc_orig': loc_orig,
            'map_size': [H, W]
        }
        return out_dict

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.init_dict(x)

        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        if self.return_map:
            y_list = self.tran2map(y_list)

        # return y_list
        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)

        return y


    def tran2map(self, input_list):
        for i in range(len(input_list)):
            input_dict = input_list[i]
            if i == 0:
                x = input_dict['x']
                B, N, C = x.shape
                H, W = input_dict['map_size']
                x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                input_list[i] = x
            else:
                x = input_dict['x']
                B, N, C = x.shape
                H, W = input_dict['map_size']
                idx_agg = input_dict['idx_agg']
                loc_orig = input_dict['loc_orig']
                x, _ = token2map_agg_mat(x, None, loc_orig, idx_agg, [H, W])
                input_list[i] = x
        return input_list

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()





@register_model
def myhrpvt_32_sr(pretrained=False, **kwargs):
    norm_cfg = dict(type='BN', requires_grad=True)
    model = MyHRPVT(
        in_channels=3,
        norm_cfg=norm_cfg,
        return_map=True,
        extra=dict(
            drop_path_rate=0.1,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2,),
                num_channels=(64,),
                num_heads=[2],
                num_mlp_ratios=[4]),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                remerge=(False, False),
                block='MYBLOCK',
                num_blocks=(2, 2),
                num_channels=(32, 64),
                num_heads=[1, 2],
                num_mlp_ratios=[4, 4],
                sr_ratios=[8, 4]),
            stage3=dict(
                num_modules=4,
                remerge=(False, False, False, False),
                num_branches=3,
                block='MYBLOCK',
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128),
                num_heads=[1, 2, 4],
                num_mlp_ratios=[4, 4, 4],
                sr_ratios=[8, 4, 2]),
            stage4=dict(
                num_modules=2,
                remerge=(False, False),
                num_branches=4,
                block='MYBLOCK',
                num_blocks=(2, 2, 2, 2),
                num_channels=(32, 64, 128, 256),
                num_heads=[1, 2, 4, 8],
                num_mlp_ratios=[4, 4, 4, 4],
                sr_ratios=[8, 4, 2, 1],
                multiscale_output=True),
        )
    )

    model.default_cfg = _cfg()

    return model



# @register_model
# def myhrpvt_32_re(pretrained=False, **kwargs):
#     norm_cfg = dict(type='BN', requires_grad=True)
#     model = MyHRPVT(
#         in_channels=3,
#         norm_cfg=norm_cfg,
#         return_map=True,
#         extra=dict(
#             drop_path_rate=0.1,
#             stage1=dict(
#                 num_modules=1,
#                 num_branches=1,
#                 block='BOTTLENECK',
#                 num_blocks=(2,),
#                 num_channels=(64,),
#                 num_heads=[2],
#                 num_mlp_ratios=[4]),
#             stage2=dict(
#                 num_modules=1,
#                 num_branches=2,
#                 remerge=(True, False),
#                 block='MYBLOCK',
#                 num_blocks=(2, 2),
#                 num_channels=(32, 64),
#                 num_heads=[1, 2],
#                 num_mlp_ratios=[4, 4],
#                 sr_ratios=[8, 4]),
#             stage3=dict(
#                 num_modules=4,
#                 remerge=(True, False, False, False),
#                 num_branches=3,
#                 block='MYBLOCK',
#                 num_blocks=(2, 2, 2),
#                 num_channels=(32, 64, 128),
#                 num_heads=[1, 2, 4],
#                 num_mlp_ratios=[4, 4, 4],
#                 sr_ratios=[8, 4, 2]),
#             stage4=dict(
#                 num_modules=2,
#                 remerge=(True, False),
#                 num_branches=4,
#                 block='MYBLOCK',
#                 num_blocks=(2, 2, 2, 2),
#                 num_channels=(32, 64, 128, 256),
#                 num_heads=[1, 2, 4, 8],
#                 num_mlp_ratios=[4, 4, 4, 4],
#                 sr_ratios=[8, 4, 2, 1],
#                 multiscale_output=True),
#         )
#     )
#
#     model.default_cfg = _cfg()
#
#     return model
