# --------------------------------------------------------
# High Resolution Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Rao Fu, RainbowSecret
# --------------------------------------------------------

import os
import pdb
import math
import logging
import torch
import torch.nn as nn
from functools import partial

from .multihead_isa_attention import PadBlock, LocalPermuteModule, MultiheadAttentionRPE
from .ffn_block import MlpDWBN
from timm.models.layers import DropPath
from utils_mine import map2token_agg_sparse_nearest, token2map_agg_sparse
from einops import rearrange
import torch.nn.functional as F

import copy
import warnings
from typing import Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch._jit_internal import List, Optional, Tuple
from torch.nn.functional import linear, pad, softmax, dropout
from torch._overrides import has_torch_function, handle_torch_function
from timm.models.layers import trunc_normal_





# def determin_grid(x, idx_agg, agg_weight, loc_orig, window_grid):
#     B, N0 = idx_agg.shape
#     N = x.shape[1]
#
#     h, w = window_grid
#     win_map = torch.eye(h * w, device=idx_agg.device)
#     win_map = win_map.reshape(1, h, w, h * w).expand(B, h, w, h * w).permute(0, 3, 1, 2)
#     win_weight = map2token_agg_sparse_nearest(win_map, N, loc_orig, idx_agg, agg_weight)
#     idx_win = win_weight.argmax(dim=-1)
#     return idx_win


class ClusterISAAttention(nn.Module):
    r"""interlaced sparse multi-head self attention (ISA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        window_size=7,
        attn_type="isa_local",
        rpe=False,
        **kwargs,
    ):
        super().__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.with_rpe = rpe

        self.attn = ClusterAttention(
            embed_dim, num_heads, rpe=rpe, window_size=window_size, **kwargs
        )
        self.pad_helper = PadBlock(window_size)
        assert attn_type in ["isa_local"]
        if attn_type == "isa_local":
            self.permute_helper = LocalPermuteModule(window_size)
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")


    def forward(self, tar_dict, src_dict=None):
        if src_dict is None:
            src_dict = tar_dict
        x = tar_dict['x']
        B, N, C = x.shape
        x_source = src_dict['x']
        x_map, _ = token2map_agg_sparse(x_source,
                                     None,
                                     src_dict['loc_orig'],
                                     src_dict['idx_agg'],
                                     src_dict['map_size'])
        x_map = x_map.permute(0, 2, 3, 1)

        x_map_pad = self.pad_helper.pad_if_needed(x_map, x_map.size())


        # (B, C, k_h*k_w, n_h * n_w)
        b, h, w, c = x_map_pad.shape
        k_h, k_w = self.permute_helper.lgs
        n_h, n_w = h // k_h, w // k_w
        x_permute = rearrange(
            x_map_pad,
            "b (n_h  k_h) (n_w k_w) c -> b (n_h n_w) (k_h k_w) c ",
            b=b, n_h=n_h, k_h=k_h, n_w=n_w, k_w=k_w, c=c,
        )


        # determine win idx
        win_map = torch.eye(n_h * n_w, device=x.device, dtype=x.dtype).reshape(1, n_h, n_w, n_h * n_w).permute(0, 3, 1, 2)
        win_map = F.interpolate(win_map, [h, w], mode='nearest')
        win_map = self.pad_helper.depad_if_needed(win_map, x_map_pad.size())
        win_map = win_map.repeat([B, 1, 1, 1])
        win_weight = map2token_agg_sparse_nearest(win_map, N, tar_dict['loc_orig'], tar_dict['idx_agg'], tar_dict['agg_weight'])
        idx_win = win_weight.argmax(dim=-1)


        # attention
        out, _, _ = self.attn(x, x_permute, x_permute, idx_win, rpe=self.with_rpe)
        return out.reshape(B, N, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


def win_att1(q, k, v, idx, p, training):
    B, N, H, C = q.shape
    B, W, K, H, C = k.shape
    step = max(N // K, 1)
    begin = 0

    idx_batch = torch.arange(B, device=q.device)[:, None]
    outs = []
    while begin < N:
        end = min(begin + step, N)
        q_t = q[:, begin:end]
        idx_w = idx[:, begin:end]
        n = idx_w.shape[1]
        k_t = k[idx_batch.expand_as(idx_w).reshape(-1), idx_w.reshape(-1)].reshape(B, n, K, H, -1)
        v_t = v[idx_batch.expand_as(idx_w).reshape(-1), idx_w.reshape(-1)].reshape(B, n, K, H, -1)

        attn = torch.einsum("bnhc,bnkhc->bnhk", [q_t, k_t])
        attn = attn.softmax(dim=-1)
        attn = dropout(attn, p=p, training=training)
        out = torch.einsum("bnhk,bnkhc->bnhc", [attn, v_t])
        out = out.flatten(-2)
        outs.append(out)
        begin = end

    outs = torch.cat(outs, dim=1)
    return outs


def win_att(q, k, v, idx, p, training):
    B, N, H, C = q.shape
    B, W, K, H, C = k.shape

    idx_batch = torch.arange(B, device=q.device)[:, None]
    attn = torch.einsum("bnhc,bnkhc->bnhk", [q, k[idx_batch.expand_as(idx).reshape(-1), idx.reshape(-1)].reshape(B, N, K, H, -1)])
    attn = attn.softmax(dim=-1)
    attn = dropout(attn, p=p, training=training)
    out = torch.einsum("bnhk,bnkhc->bnhc", [attn, v[idx_batch.expand_as(idx).reshape(-1), idx.reshape(-1)].reshape(B, N, K, H, -1)])
    out = out.flatten(-2)
    return out


class ClusterAttention(MultiheadAttentionRPE):
    def forward(
            self,
            query,
            key,
            value,
            idx_win,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            do_qkv_proj=True,
            do_out_proj=True,
            rpe=False,
    ):
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                idx_win,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                idx_win,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )


    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        idx_win: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[Tensor] = None,
        do_qkv_proj: bool = True,
        do_out_proj: bool = True,
        rpe=True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not torch.jit.is_scripting():
            tens_ops = (
                query,
                key,
                value,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                out_proj_weight,
                out_proj_bias,
            )
            if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
                tens_ops
            ):
                return handle_torch_function(
                    multi_head_attention_forward,
                    tens_ops,
                    query,
                    key,
                    value,
                    embed_dim_to_check,
                    num_heads,
                    in_proj_weight,
                    in_proj_bias,
                    bias_k,
                    bias_v,
                    add_zero_attn,
                    dropout_p,
                    out_proj_weight,
                    out_proj_bias,
                    training=training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=use_separate_proj_weight,
                    q_proj_weight=q_proj_weight,
                    k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight,
                    static_k=static_k,
                    static_v=static_v,
                )

        if attn_mask is not None:
            print('error, att_mask is not supported!')

        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            print('error, key_padding_mask is not supported!')

        if self.rpe and rpe:
            print('error, RPE is not supported!')

        if add_zero_attn:
            print('error, add zero attn is not supported!')


        # tgt_len, bsz, embed_dim = query.size()
        B, N, C = query.shape
        embed_dim = C

        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check

        # allow MHA to have different sizes for the feature dimension
        # assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        # whether or not use the original query/key/value
        B, N, C = query.shape
        B, W, K, _ = key.shape

        q = self.q_proj(query) * scaling if do_qkv_proj else query
        k = self.k_proj(key.flatten(1, 2)).reshape(B, W, K, -1) if do_qkv_proj else key
        v = self.v_proj(value.flatten(1, 2)).reshape(B, W, K, -1) if do_qkv_proj else value

        q = q.reshape(B, N, num_heads, head_dim)
        k = k.reshape(B, W, K, num_heads, head_dim)
        v = v.reshape(B, W, K, num_heads, v_head_dim)

        # get attention
        attn_output = win_att(q, k, v, idx_win, p=dropout_p, training=training)


        if do_out_proj:
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            print('error: nead weiights')
        else:
            return attn_output, q, k  # additionaly return the query and key




class ClusterBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        # input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_type="isa_local",
        ffn_type="conv_mlp",
        sr_ratio=None,
        norm_cfg=None,
        conv_cfg=None,

    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        # self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.mlp_ratio = mlp_ratio

        self.attn = ClusterISAAttention(
            self.dim,
            num_heads=num_heads,
            window_size=window_size,
            attn_type=attn_type,
            rpe=False,
            dropout=attn_drop,
        )
        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.out_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(self.dim * mlp_ratio)

        if self.ffn_type in ["conv_mlp"]:
            self.mlp = ClusterMlpDWBN(
                in_features=self.dim,
                hidden_features=mlp_hidden_dim,
                out_features=self.out_dim,
                act_layer=act_layer,
                drop=drop,
            )
        elif self.ffn_type in ["identity"]:
            self.mlp = nn.Identity()
        else:
            raise RuntimeError("Unsupported ffn type: {}".format(self.ffn_type))


    def forward(self, tar_dict, src_dict=None):
        if self.attn_type in ["isa_local"]:
            x = tar_dict['x']
            # Attention
            tar_dict['x'] = self.norm1(x)
            x = x + self.drop_path(self.attn(tar_dict, src_dict))
            # FFN
            x = self.norm2(x)
            tar_dict['x'] = x
            x = x + self.drop_path(self.mlp(tar_dict))
            # reshape
            # x = x.permute(0, 2, 1).view(B, C, H, W)

            out_dict = {
                'x': x,
                'idx_agg': tar_dict['idx_agg'],
                'agg_weight': tar_dict['agg_weight'],
                'loc_orig': tar_dict['loc_orig'],
                'map_size': tar_dict['map_size'],
            }
            return out_dict
        else:
            print('error in attention type!')



class ClusterMlpDWBN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        dw_act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = act_layer()
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1,
        )
        self.act2 = dw_act_layer()
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = act_layer()
        self.norm3 = nn.BatchNorm2d(out_features)
        self.drop = nn.Dropout(drop)

        self.skip = nn.Conv2d(in_channels=hidden_features,
                              out_channels=hidden_features,
                              kernel_size=1, bias=False,
                              groups=hidden_features)

    def forward(self, input_dict):
        x = input_dict['x']
        B, N, C = x.shape

        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x_map, _ = token2map_agg_sparse(
            x.squeeze(-1).permute(0, 2, 1),
            None,
            input_dict['loc_orig'],
            input_dict['idx_agg'],
            input_dict['map_size']
        )
        x_map = self.dw3x3(x_map)
        x_map = map2token_agg_sparse_nearest(
            x_map, N,
            input_dict['loc_orig'],
            input_dict['idx_agg'],
            input_dict['agg_weight']).permute(0, 2, 1).unsqueeze(-1)
        x = x_map + self.skip(x)

        x = self.norm2(x)
        x = self.act2(x)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.drop(x)

        x = x.squeeze(-1).permute(0, 2, 1)
        return x
