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




class MyMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MyDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
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

        self.sr_ratio = sr_ratio
        #if sr_ratio > 1:
            # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
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

        h, w = H // self.sr_ratio, W // self.sr_ratio
        x_source = token2map(x_source, loc_source, [h, w], 1, 1)
        x_source = self.sr(x_source)
        x_source = x_source.reshape(B, C, -1).permute(0, 2, 1)
        x_source = self.norm(x_source)
        x_source = self.act(x_source)
        if conf_source is not None:
            conf_source = token2map(conf_source, loc_source, [h, w], 1, 1)
            conf_source = conf_source.reshape(B, 1, -1).permute(0, 2, 1)

        _, Ns, _ = x_source.shape
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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, x_source, loc, loc_source, H, W, conf_source=None):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), loc_source, H, W, conf_source))
        kernel_size = self.attn.sr_ratio + 1
        x = x + self.drop_path(self.mlp(self.norm2(x), loc, H, W, kernel_size, 2))
        return x



# from partialconv2d import PartialConv2d
class DownLayer(nn.Module):
    """ Down sample
    """
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block):
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
        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=1, padding=1)
        # self.conv = PartialConv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

    def forward(self, x, pos, H, W, N_grid):
        # x, mask = token2map(x, pos, [H, W], 1, 2, return_mask=True)
        # x = self.conv(x, mask)

        kernel_size = self.block.attn.sr_ratio + 1
        x = token2map(x, pos, [H, W], kernel_size, 2)
        x = self.conv(x)
        x = map2token(x, pos)
        B, N, C = x.shape
        # sample_num = max(math.ceil(N * self.sample_ratio) - N_grid, 0)
        sample_num = max(math.ceil(N * self.sample_ratio), 0)

        x_grid = x[:, :N_grid]
        x_ada = x[:, N_grid:]
        pos_grid = pos[:, :N_grid]
        pos_ada = pos[:, N_grid:]

        conf = self.conf(self.norm(x))
        conf_ada = conf[:, N_grid:]
        if vis:
            show_conf(conf, pos)
        # temperature
        # T = self.T if self.training else self.T_min
        T = self.T
        self.T = (self.T * self.T_decay).clamp(self.T_min, 1.0)

        # _, index_down = torch.topk(conf_ada, self.sample_num, 1)
        index_down = gumble_top_k(conf_ada, sample_num, 1, T=T)

        # conf = F.softmax(conf, dim=1) * N
        # conf = F.sigmoid(conf)
        # conf = self.gumble_sigmoid(conf)

        x_down = torch.gather(x_ada, 1, index_down.expand([B, sample_num, C]))
        pos_down = torch.gather(pos_ada, 1, index_down.expand([B, sample_num, 2]))

        x_down = torch.cat([x_grid, x_down], 1)
        pos_down = torch.cat([pos_grid, pos_down], 1)

        x_down = self.block(x_down, x, pos_down, pos, H, W, conf)
        return x_down, pos_down


class MyPVT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=False, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.grid_stride = sr_ratios[0]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        # self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1])
        # self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2])
        # self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
        #                                       embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], linear=linear)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.down_layers1 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[0], dim_out=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                                      )
        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1, depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.down_layers2 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[1], dim_out=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1]))
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1, depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.down_layers3 = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[2], dim_out=embed_dims[3], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2]))
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(1, depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

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
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, loc, N_grid = get_loc(x, H, W, self.grid_stride)
        if vis: outs.append((x, loc, [H, W]))

        # stage 2
        x, loc = self.down_layers1(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block2:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm2(x)
        if vis: outs.append((x, loc, [H, W]))

        # stage 3
        x, loc = self.down_layers2(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block3:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm3(x)
        if vis: outs.append((x, loc, [H, W]))

        # stage 4
        x, loc = self.down_layers3(x, loc, H, W, N_grid)     # down sample
        H, W = H // 2, W // 2
        for blk in self.block4:
            x = blk(x, x, loc, loc, H, W)
        x = self.norm4(x)
        if vis:
            outs.append((x, loc, [H, W]))
            show_tokens(img, outs, N_grid)

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


'''Actually, this is modified from pvt_v2_b2_li '''
@register_model
def mypvt20_2_small(pretrained=False, **kwargs):
    model = MyPVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True,  **kwargs)
    model.default_cfg = _cfg()

    return model


# For test
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = mypvt20_2_small(drop_path_rate=0.).to(device)
    model.reset_drop_path(0.)
    # pre_dict = torch.load('work_dirs/my20_s2/my20_300.pth')['model']
    # model.load_state_dict(pre_dict)
    x = torch.zeros([1, 3, 448, 448]).to(device)
    x = F.avg_pool2d(x, kernel_size=2)
    tmp = model.forward(x)
    print('Finish')

