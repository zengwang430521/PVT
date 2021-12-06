import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from pvt_v2 import (Block, DropPath, DWConv, OverlapPatchEmbed,
                    to_2tuple, trunc_normal_, register_model, _cfg)
from utils_mine import (
    get_grid_loc,
    gumble_top_k,
    show_tokens_merge_multi, show_conf_merge_multi,
    merge_tokens, merge_tokens_agg_dist_multi, token2map_Agg, map2token_Agg,
    farthest_point_sample

)
vis = False


'''
do not select tokens, merge tokens. weight NOT clamp, conf do not clamp
merge feature, but not merge locs, reserve all locs.
inherit weights when map2token, which can regarded as tokens merge
farthest_point_sample DOWN, N_grid = 0, feature distance merge
token2map nearest + skip token conv (this must be used together.)

merge to multiple tokens

'''

class MyMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MyDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
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

    def forward(self, x, Agg, loc_orig, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, Agg, loc_orig, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MyDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv_skip = nn.Conv1d(dim, dim, 1, bias=False, groups=dim)

    def forward(self, x, Agg, loc_orig, H, W):
        x_map = token2map_Agg(x, Agg, loc_orig, [H, W], weight=None)
        x_map = self.dwconv(x_map)
        x = map2token_Agg(x_map, Agg, loc_orig) + \
            self.dwconv_skip(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class MyAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
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

    def forward(self, x, Agg, loc_orig, x_source, Agg_source, H, W, conf_source=None):
        B, N, C = x.shape
        Ns = x_source.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                if conf_source is None:
                    conf_source = x_source.new_zeros(B, Ns, 1)
                tmp = torch.cat([x_source, conf_source], dim=-1)
                tmp = token2map_Agg(tmp, Agg_source, loc_orig, [H, W], weight=None)
                x_source = tmp[:, :C]
                conf_source = tmp[:, C:]

                x_source = self.sr(x_source)
                _, _, h, w = x_source.shape
                x_source = x_source.reshape(B, C, -1).permute(0, 2, 1)
                x_source = self.norm(x_source)
                conf_source = F.avg_pool2d(conf_source, kernel_size=self.sr_ratio, stride=self.sr_ratio)
                conf_source = conf_source.reshape(B, 1, -1).permute(0, 2, 1)

        else:
            print('error!')

        kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if conf_source is not None:
            conf_source = conf_source.squeeze(-1)[:, None, None, :]
            attn = attn + conf_source
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MyBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
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
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

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

    def forward(self, x, Agg, loc_orig, x_source, Agg_source, H, W, conf_source=None):
        x1 = x + self.drop_path(self.attn(self.norm1(x),
                                          Agg,
                                          loc_orig,
                                          self.norm1(x_source),
                                          Agg_source,
                                          H, W, conf_source))

        x2 = x1 + self.drop_path(self.mlp(self.norm2(x1),
                                          Agg,
                                          loc_orig,
                                          H, W))
        if torch.isnan(x2).any():
            save_dict = {
                'x':x.detach().cpu(),
                'x_source': x_source,
                'x1': x1,
                'x2': x2
            }
            if conf_source is not None:
                save_dict['conf_source'] = conf_source
            for key in save_dict.keys():
                save_dict[key] = save_dict[key].detach().cpu()
            torch.save(save_dict, 'debug_block.pth')
            exit(1)
            assert torch.isnan(x2).any() is False
        return x2



# from partialconv2d import PartialConv2d
class DownLayer(nn.Module):
    """ Down sample
    """
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block, k=1):
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
        self.k = k

    def forward(self, x, Agg, loc_orig, H, W, N_grid):

        x_map = token2map_Agg(x, Agg, loc_orig, [H, W])
        x_map = self.conv(x_map)
        _, _, H, W = x_map.shape
        x = map2token_Agg(x_map, Agg, loc_orig) +\
            self.conv_skip(x)
        B, N, C = x.shape

        # sample_num = max(math.ceil(N * self.sample_ratio) - N_grid, 0)
        # sample_num = max(math.ceil(N * self.sample_ratio), 0)
        sample_num = max(math.ceil(N * self.sample_ratio) - N_grid, 0)
        if sample_num < N_grid:
            sample_num = N_grid

        conf = self.conf(self.norm(x))

        # # _, index_down = torch.topk(conf_ada, self.sample_num, 1)
        # index_down = gumble_top_k(conf_ada, sample_num, 1, T=1)
        # pos_down = torch.gather(pos_ada, 1, index_down.expand([B, sample_num, 2]))
        # pos_down = torch.cat([pos_grid, pos_down], 1)

        x_grid = x[:, :N_grid]
        x_ada = x[:, N_grid:]
        with torch.no_grad():
            index_down = farthest_point_sample(x_ada, sample_num).unsqueeze(-1)
        x_down = torch.gather(x_ada, 1, index_down.expand([B, sample_num, C]))
        x_down = torch.cat([x_grid, x_down], 1)

        # pos_down = get_grid_loc(B, H, W, x.device)

        # conf = conf.clamp(-7, 7)
        # weight = conf.clamp(-7, 7).exp()
        weight = conf.exp()
        x_down, A = merge_tokens_agg_dist_multi(x, index_down, x_down, weight, k=self.k)
        Agg_down = A @ Agg
        Agg_down = Agg_down / Agg_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, Agg_down, loc_orig, x, Agg, H, W, conf_source=conf)

        if vis:
            show_conf_merge_multi(conf, Agg, loc_orig)
        return x_down, Agg_down, A



class MyPVT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, k=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(1):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        for i in range(1, num_stages):
            down_layers = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[i-1], dim_out=embed_dims[i],
                                          drop_rate=drop_rate, k=k,
                                          down_block=MyBlock(
                                              dim=embed_dims[i], num_heads=num_heads[i],
                                              mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                              norm_layer=norm_layer, sr_ratio=sr_ratios[i], linear=linear)
                                    )
            block = nn.ModuleList([MyBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(1, depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"down_layers{i}", down_layers)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

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

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        self.down_layers1.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[1] - 1):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1] - 1
        self.down_layers2.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[2] - 1):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2] - 1
        self.down_layers3.block.drop_path.drop_prob = dpr[cur]
        cur += 1
        for i in range(self.depths[3] - 1):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if vis:
            outs = []
            img = x

        B = x.shape[0]
        # stage 1
        i = 0
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)
        loc = get_grid_loc(B, H, W, x.device)
        N_grid = 0

        B, N, _ = x.shape
        loc_orig = loc
        Agg = torch.eye(N, device=x.device, dtype=x.dtype)[None, :, :].repeat([B, 1, 1])

        if vis: outs.append((x, [H, W], Agg, loc_orig, Agg))

        for i in range(1, self.num_stages):
            down_layers = getattr(self, f"down_layers{i}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, Agg, A = down_layers(x, Agg, loc_orig, H, W, N_grid)  # down sample
            H, W = H // 2, W // 2

            for j, blk in enumerate(block):
                x = blk(x, Agg, loc_orig, x, Agg, H, W, conf_source=None)
            x = norm(x)

            if vis: outs.append((x, [H, W], Agg, loc_orig, A))

        if vis:
            show_tokens_merge_multi(img, outs, N_grid)

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def mypvt3h7_small(pretrained=False, **kwargs):
    model = MyPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], k=1, **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def mypvt3h7k3_small(pretrained=False, **kwargs):
    model = MyPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], k=3, **kwargs)
    model.default_cfg = _cfg()

    return model

# For test
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = mypvt3h7_small(drop_path_rate=0.).to(device)
    # pre_dict = torch.load('work_dirs/my20_s2/my20_300.pth')['model']
    # model.load_state_dict(pre_dict)
    for i in range(10):
        x = torch.rand([2, 3, 112, 112]).to(device)
        tmp = model(x)
    print('Finish')


