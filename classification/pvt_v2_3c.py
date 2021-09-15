import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from pvt_v2 import (Block, DropPath, DWConv, OverlapPatchEmbed,
                    to_2tuple, trunc_normal_, register_model, _cfg)
from utils_mine import (
    get_grid_loc, get_loc, extract_local_feature, extract_neighbor_feature,
    gumble_top_k, guassian_filt, reconstruct_feature, token2map, map2token,
    show_tokens, show_conf
)

vis = False

'''
modify down block
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

    def forward(self, x, loc, H, W, kernel_size, sigma):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, loc, H, W, kernel_size, sigma)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MyDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, loc, H, W, kernel_size, sigma):
        B, N, C = x.shape
        x = token2map(x, loc, [H, W], kernel_size=kernel_size, sigma=sigma)
        # x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        # x = x.flatten(2).transpose(1, 2)
        x = map2token(x, loc)
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

    def forward(self, x, x_source, loc_source, H, W, conf_source=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                kernel = self.sr_ratio + 1
                x_source = token2map(x_source, loc_source, [H, W], kernel_size=kernel, sigma=2)
                x_source = self.sr(x_source).reshape(B, C, -1).permute(0, 2, 1)
                x_source = self.norm(x_source)
                if conf_source is not None:
                    h, w = H // self.sr_ratio, W // self.sr_ratio
                    conf_source = token2map(conf_source, loc_source, [h, w], 1, 1)
                    conf_source = conf_source.reshape(B, 1, -1).permute(0, 2, 1)
        else:
            h, w = H // self.sr_ratio, W // self.sr_ratio
            x_source = token2map(x_source, loc_source, [h, w], 1, 1)
            x_source = self.sr(x_source)
            x_source = x_source.reshape(B, C, -1).permute(0, 2, 1)
            x_source = self.norm(x_source)
            x_source = self.act(x_source)
            if conf_source is not None:
                conf_source = token2map(conf_source, loc_source, [h, w], 1, 1)
                conf_source = conf_source.reshape(B, 1, -1).permute(0, 2, 1)

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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False,
                 sample_ratio=1, dim_in=None
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
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.sample_ratio = sample_ratio
        self.dim_in = dim_in
        if self.sample_ratio < 1:
            self.pre_conv = nn.Conv2d(dim_in, dim, kernel_size=3, stride=2, padding=1)
            self.conf_norm = nn.LayerNorm(dim)
            self.conf = nn.Linear(dim, 1)

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

    def forward(self, x, loc, H, W, N_grid=0):
        if self.sample_ratio == 1:
            # x_source, loc_source, conf_source = x, loc, None
            # x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), loc_source, H, W, conf_source))
            # kernel_size = self.attn.sr_ratio + 1
            # x = x + self.drop_path(self.mlp(self.norm2(x), loc, H, W, kernel_size, 2))

            x_norm = self.norm1(x)
            x_dst = x_norm
            x_source, loc_source, conf_source = x_norm, loc, None

        else:
            # pre conv
            kernel_size = (self.attn.sr_ratio * 2) + 1
            x = token2map(x, loc, [H, W], kernel_size, 2)
            x = self.pre_conv(x)
            H, W = H // 2, W // 2
            x = map2token(x, loc)
            B, N, C = x.shape

            # confidence
            conf = self.conf(self.conf_norm(x))

            # get down sample tokens idx
            sample_num = max(math.ceil(N * self.sample_ratio) - N_grid, 0)
            if sample_num == 0:
                sample_num = max(math.ceil(N * self.sample_ratio), 0)

            conf_ada = conf[:, N_grid:]
            if vis:
                show_conf(conf, loc)
            index_down = gumble_top_k(conf_ada, sample_num, 1, T=1)

            # norm in block
            x_norm = self.norm1(x)

            # down sample
            x_grid, x_ada = x[:, :N_grid], x[:, N_grid:]
            x_grid_norm, x_ada_norm = x_norm[:, :N_grid], x_norm[:, N_grid:]
            loc_grid, loc_ada = loc[:, :N_grid], loc[:, N_grid:]

            x_down = torch.gather(x_ada, 1, index_down.expand([B, sample_num, C]))
            x_down_norm = torch.gather(x_ada_norm, 1, index_down.expand([B, sample_num, C]))
            loc_down = torch.gather(loc_ada, 1, index_down.expand([B, sample_num, 2]))

            x_down = torch.cat([x_grid, x_down], 1)
            x_down_norm = torch.cat([x_grid_norm, x_down_norm], 1)
            loc_down = torch.cat([loc_grid, loc_down], 1)

            x_source, loc_source, conf_source = x_norm, loc, conf
            x, loc = x_down, loc_down
            x_dst = x_down_norm

        x = x + self.drop_path(self.attn(x_dst, x_source, loc_source, H, W, conf_source))
        kernel_size = self.attn.sr_ratio + 1
        x = x + self.drop_path(self.mlp(self.norm2(x), loc, H, W, kernel_size, 2))
        return x, loc


class MyPVT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
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
            block = nn.ModuleList([MyBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear,
                sample_ratio=0.25 if j == 0 else 1, dim_in=embed_dims[i-1] if j == 0 else None)
                for j in range(0, depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

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

        # stage 1
        i = 0
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)
        x, loc, N_grid = get_loc(x, H, W, self.grid_stride)
        if vis: outs.append((x, loc, [H, W]))

        for i in range(1, self.num_stages):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            for j, blk in enumerate(block):
                x, loc = blk(x, loc, H, W, N_grid)
                if j == 0:
                    H, W = H // 2, W // 2
            x = norm(x)
            if vis: outs.append((x, loc, [H, W]))

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def mypvt3c_small(pretrained=False, **kwargs):
    model = MyPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],  **kwargs)
    model.default_cfg = _cfg()

    return model


# For test
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = mypvt3c_small(drop_path_rate=0.).to(device)
    # model.reset_drop_path(0.)
    # pre_dict = torch.load('work_dirs/my20_s2/my20_300.pth')['model']
    # model.load_state_dict(pre_dict)
    x = torch.zeros([1, 3, 112, 112]).to(device)
    # x = F.avg_pool2d(x, kernel_size=2)
    tmp = model.forward(x)
    print('Finish')

