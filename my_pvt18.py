import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from pvt import ( Mlp, Attention, PatchEmbed, Block, DropPath, to_2tuple, trunc_normal_,register_model, _cfg)
import math
import matplotlib.pyplot as plt
# from partialconv2d import PartialConv2d
# from torchvision.ops import roi_align

vis = False


def gumble_top_k(x, k, dim, T=1, p_value=1e-6):
    # Noise
    noise = torch.rand_like(x)
    noise = -1 * (noise + p_value).log()
    noise = -1 * (noise + p_value).log()
    # add
    x = x / T + noise
    _, index_k = torch.topk(x, k, dim)
    return index_k


def guassian_filt(x, kernel_size=3, sigma=2):
    channels = x.shape[1]

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size, device=x.device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    paddding = int((kernel_size - 1) // 2)

    y = F.conv2d(
        input=x,
        weight=gaussian_kernel,
        stride=1,
        padding=paddding,
        dilation=1,
        groups=channels
    )
    return y


def reconstruct_feature(feature, mask, kernel_size, sigma):
    if kernel_size <= 1:
        return feature
    feature = feature * mask
    out = guassian_filt(torch.cat([feature, mask], dim=1),
                        kernel_size=kernel_size, sigma=sigma)
    feature_inter = out[:, :-1]
    mask_inter = out[:, [-1]]
    tmp = mask_inter.min()
    feature_inter = feature_inter / (mask_inter + 1e-6)
    mask_inter = (mask_inter > 0).float()
    feature_inter = feature_inter * mask_inter
    out = feature + (1 - mask) * feature_inter
    return out


def token2map(x, loc, map_size, kernel_size, sigma, return_mask=False):
    H, W = map_size
    B, N, C = x.shape
    loc = loc.clamp(0, 1)
    loc = loc * torch.FloatTensor([W-1, H-1]).to(loc.device)[None, None, :]
    loc = loc.round().long()
    idx = loc[..., 0] + loc[..., 1] * W
    idx = idx + torch.arange(B)[:, None].to(loc.device) * H*W

    out = x.new_zeros(B*H*W, C+1)
    out.index_add_(dim=0, index=idx.reshape(B*N),
                   source=torch.cat([x, x.new_ones(B, N, 1)], dim=-1).reshape(B*N, C+1))
    out = out.reshape(B, H, W, C+1).permute(0, 3, 1, 2)
    feature, mask = out[:, :-1], out[:, [-1]]

    feature = feature / (mask + 1e-6)
    mask = (mask > 0).float()
    feature = feature * mask
    feature = reconstruct_feature(feature, mask, kernel_size, sigma)
    if return_mask:
        return feature, mask
    return feature


def map2token(feature_map, loc_xy, mode='bilinear', align_corners=False):
    B, N, _ = loc_xy.shape
    # B, C, H, W = feature_map.shape
    loc_xy = loc_xy.type(feature_map.dtype) * 2 - 1
    loc_xy = loc_xy.unsqueeze(1)
    tokens = F.grid_sample(feature_map, loc_xy, mode=mode, align_corners=align_corners)
    tokens = tokens.permute(0, 2, 3, 1).squeeze(1)
    return tokens


class MyAttention(nn.Module):
    def __init__(self, dim, dim_out=-1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        if dim_out < 0:
            dim_out = dim
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert dim_out % num_heads == 0, f"dim_out {dim_out} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, x_source, loc_source, H, W, conf_source=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)

        h, w = H // self.sr_ratio, W // self.sr_ratio
        x_source = token2map(x_source, loc_source, [h, w], 1, 1)
        x_source = self.sr(x_source)
        x_source = x_source.reshape(B, C, -1).permute(0, 2, 1)
        x_source = self.norm(x_source)
        if conf_source is not None:
            conf_source = token2map(conf_source, loc_source, [h, w], 1, 2)
            conf_source = conf_source.reshape(B, 1, -1).permute(0, 2, 1)

        _, Ns, _ = x_source.shape
        k = self.k(x_source).reshape(B, Ns, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_source).reshape(B, Ns, self.num_heads, self.dim_out // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if conf_source is not None:
            conf_source = conf_source.squeeze(-1)[:, None, None, :]
            attn = attn + conf_source
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim_out)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MyBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dim_out=None, sr_ratio=1, alpha=1):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(
            dim,
            dim_out=dim_out,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_fc = False
        if dim_out != dim:
            self.use_fc = True
            self.fc = nn.Linear(dim, dim_out)
        self.alpha = alpha
        self.dim = dim
        self.dim_out = dim_out

    def forward(self, x, x_source, loc, H, W, conf_source=None):
        if self.use_fc:
            x = self.fc(x) + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), loc, H, W, conf_source)) * self.alpha
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x_source), loc, H, W, conf_source)) * self.alpha

        x = x + self.drop_path(self.mlp(self.norm2(x))) * self.alpha

        if torch.isnan(x).any():
            print('NAN')

        return x


def get_pos_embed(pos_embed, loc_xy, pos_size=None):
    _, H, W, C = pos_embed.shape
    B, N, _ = loc_xy.shape
    pos_embed = pos_embed.permute(0, 3, 1, 2).expand([B, C, H, W])
    loc_xy = loc_xy * 2 - 1
    loc_xy = loc_xy.unsqueeze(1)
    pos_feature = F.grid_sample(pos_embed, loc_xy)
    pos_feature = pos_feature.permute(0, 2, 3, 1).squeeze(1)
    # print('use interpolate pos embed.')
    return pos_feature


class DownLayer(nn.Module):
    """ Down sample
    """
    def __init__(self, sample_num, embed_dim, drop_rate, down_block):
        super().__init__()
        self.sample_num = sample_num
        self.block = down_block
        self.pos_drop = nn.Dropout(p=drop_rate)
        # self.gumble_sigmoid = GumbelSigmoid()
        # temperature of confidence weight
        self.register_buffer('T', torch.tensor(1.0, dtype=torch.float))
        self.T_min = 1
        self.T_decay = 0.9998
        self.conv = nn.Conv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        # self.conv = PartialConv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(self.block.dim_out)
        self.conf = nn.Linear(self.block.dim_out, 1)

    def forward(self, x, pos, pos_embed, H, W, pos_size, N_grid):
        # x, mask = token2map(x, pos, [H, W], 1, 2, return_mask=True)
        # x = self.conv(x, mask)
        x = token2map(x, pos, [H, W], self.block.attn.sr_ratio + 1, 2)
        x = self.conv(x)
        x = map2token(x, pos)
        x = self.norm(x)

        B, N, C = x.shape
        assert self.sample_num <= N

        x_grid = x[:, :N_grid]
        x_ada = x[:, N_grid:]
        pos_grid = pos[:, :N_grid]
        pos_ada = pos[:, N_grid:]
        conf = self.conf(x)
        if vis:
            if H == 56:
                show_conf(conf, pos)

        conf_ada = conf[:, N_grid:]

        # temperature
        # T = self.T if self.training else self.T_min
        T = self.T
        self.T = (self.T * self.T_decay).clamp(self.T_min, 1.0)

        # _, index_down = torch.topk(conf_ada, self.sample_num, 1)
        index_down = gumble_top_k(conf_ada, self.sample_num, 1, T=T)
        # conf = F.softmax(conf, dim=1) * N
        # conf = F.sigmoid(conf)
        # conf = self.gumble_sigmoid(conf)

        x_down = torch.gather(x_ada, 1, index_down.expand([B, self.sample_num, C]))
        pos_down = torch.gather(pos_ada, 1, index_down.expand([B, self.sample_num, 2]))

        x_down = torch.cat([x_grid, x_down], 1)
        pos_down = torch.cat([pos_grid, pos_down], 1)

        # x = x * conf
        x_down = self.block(x_down, x, pos, H, W, conf)
        # if vis and conf.shape[1] == H*W:
        #     conf_t = F.sigmoid(conf.float()).reshape(-1, H, W).detach().cpu().numpy()
        #     import matplotlib.pyplot as plt
        #     for i in range(len(conf_t)):
        #         ax = plt.subplot(1, len(conf_t), i+1)
        #         tmp = conf_t[i]
        #         # tmp = tmp / tmp.max()
        #         ax.imshow(tmp)
        #     tmp = 0

        if pos_embed is not None:
            pos_feature = get_pos_embed(pos_embed, pos_down, pos_size)
            x_down += pos_feature
            x_down = self.pos_drop(x_down)
        return x_down, pos_down


def extract_local_feature(src, loc, kernel_size=(3, 3)):
    B, C, H, W = src.shape
    B, N, _ = loc.shape

    h, w = kernel_size
    x = torch.arange(w, device=loc.device, dtype=loc.dtype)
    x = (x - (w-1) / 2.0) / (W-1)
    y = torch.arange(h, device=loc.device, dtype=loc.dtype)
    y = (y - (h - 1) / 2.0) / (H-1)
    y, x = torch.meshgrid(y, x)
    grid = torch.stack([x, y], dim=-1)
    grid = loc[:, :, None, None, :] + grid[None, None, ...]     # (B, N, h, w, 2)

    grid = grid * 2 - 1
    loc_feature = F.grid_sample(src, grid.flatten(2, 3))        # (B, C, N, h * w)
    loc_feature = loc_feature.reshape(B, C, N, h, w)            # (B, C, N, h, w)
    loc_feature = loc_feature.permute(0, 2, 1, 3, 4)            # (B, N, C, h, w)
    return loc_feature.flatten(0, 1)                            # (B * N, C, h, w)


class ExtraSampleLayer(nn.Module):
    def __init__(self, embed_dim, src_dim=3, kernel_size=(4, 4), stride=4, delta_factor=0.01, mlp_ratio=4, local_dim=64):
        super().__init__()
        self.local_dim = local_dim
        self.delta_layer = nn.Linear(embed_dim, 2)
        self.delta_factor = delta_factor
        self.local_conv = nn.Conv2d(src_dim, local_dim, kernel_size, stride)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(local_dim)
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim+local_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim)

    def forward(self, x, loc, src, pos_embed, H, W, kernel_size):
        B, N, _ = loc.shape
        delta = self.delta_layer(self.norm1(x)) * self.delta_factor
        # delta = delta * 0.0 + delta.detach() * 1
        loc_extra = loc + delta
        loc_extra = loc_extra.clamp(0, 1)
        extra = extract_local_feature(src, loc_extra, self.kernel_size)
        extra = self.local_conv(extra).squeeze(-1).squeeze(-1)
        extra = extra.reshape(B, N, self.local_dim)
        extra = self.norm2(extra)
        extra_inter = token2map(x, loc, [H, W], kernel_size=kernel_size, sigma=2)
        extra_inter = map2token(extra_inter, loc_extra)
        extra = torch.cat([extra_inter, extra], dim=-1)
        extra = self.mlp(extra)
        x, loc = torch.cat([x, extra], dim=1), torch.cat([loc, loc_extra], dim=1)
        return x, loc

        # x_local = token2map(extra, loc_extra, [H, W], kernel_size=kernel_size, sigma=2)
        # x_local = map2token(x_local, loc)
        # extra = torch.cat([extra_inter, extra], dim=-1)
        # x = torch.cat([x, x_local], dim=-1)
        # x, loc = torch.cat([x, extra], dim=1), torch.cat([loc, loc_extra], dim=1)
        # x = self.mlp(x)
        # x = self.norm2(x)
        # return x, loc


def show_tokens(x, out, N_grid=7*7):
    import matplotlib.pyplot as plt
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    x = x * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
    # for i in range(x.shape[0]):
    for i in range(1):
        img = x[i].permute(1, 2, 0).detach().cpu()
        ax = plt.subplot(1, len(out)+2, 1)
        ax.clear()
        ax.imshow(img)
        for lv in range(len(out)):
            ax = plt.subplot(1, len(out)+2, lv+2+(lv > 0))
            ax.clear()
            ax.imshow(img, extent=[0, 1, 0, 1])
            loc = out[lv][1][i].detach().cpu().numpy()

            loc_grid = loc[:N_grid]
            ax.scatter(loc_grid[:, 0], 1 - loc_grid[:, 1], c='blue', s=0.4 + lv * 0.1)
            if lv > 0:
                N = loc.shape[0]
                loc_ada = loc[N_grid:N//2]
                ax.scatter(loc_ada[:, 0], 1 - loc_ada[:, 1], c='red', s=0.4+lv*0.1)
                loc_extra = loc[N//2:]
                ax.scatter(loc_extra[:, 0], 1 - loc_extra[:, 1], c='yellow', s=0.4+lv*0.1)
            else:
                loc_ada = loc[N_grid:]
                ax.scatter(loc_ada[:, 0], 1 - loc_ada[:, 1], c='red', s=0.4+lv*0.1)
    return


def show_conf(conf, loc):
    H = int(conf.shape[1]**0.5)
    conf = F.softmax(conf, dim=1)
    conf_map = token2map(conf,  map_size=[H, H], loc=loc, kernel_size=3, sigma=2)
    lv = 3
    ax = plt.subplot(1, 6, lv)
    ax.clear()
    ax.imshow(conf_map[0, 0].detach().cpu())


class MyPVT18(nn.Module):
    def __init__(self, img_size=448, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], alpha=1):
        super().__init__()
        self.ave_layer = nn.AvgPool2d(2)
        img_size = img_size // 2

        self.num_classes = num_classes
        self.depths = depths
        self.alpha = alpha
        self.grid_stride = sr_ratios[0]
        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        # pos_embed
        H = W = img_size // patch_size
        self.pos_embed1 = nn.Parameter(torch.zeros(1, H, W, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, H // 2, W // 2, embed_dims[1]))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, H // 4, W // 4, embed_dims[2]))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, H // 8, W // 8, embed_dims[3]))
        self.pos_size = [img_size // patch_size, img_size // patch_size]

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        sample_num = self.patch_embed1.num_patches
        cur = 0
        # N_grid = sample_num // self.grid_stride // self.grid_stride
        N_grid = (H // self.grid_stride) * (W // self.grid_stride)

        # stage 1
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])  # , alpha=alpha)
            for i in range(depths[0])])
        cur += depths[0]

        # stage 2
        sample_num = sample_num // 4
        self.down_layers1 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[0], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], dim_out=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0], alpha=alpha))

        self.extra_layer1 = ExtraSampleLayer(embed_dim=embed_dims[1], mlp_ratio=mlp_ratios[1])
        self.extra_block1 = MyBlock(
                dim=embed_dims[1], dim_out=embed_dims[1], num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + 1],
                norm_layer=norm_layer, sr_ratio=sr_ratios[0], alpha=alpha)
        self.extra_down1 = DownLayer(sample_num=sample_num - N_grid, embed_dim=embed_dims[1], drop_rate=drop_rate,
                      down_block=MyBlock(
                          dim=embed_dims[1], dim_out=embed_dims[1], num_heads=num_heads[1],
                          mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+2],
                          norm_layer=norm_layer, sr_ratio=sr_ratios[0], alpha=alpha))

        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], alpha=alpha)
            for i in range(3, depths[1])])
        cur += depths[1]

        # stage 3
        sample_num = sample_num // 4
        self.down_layers2 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], dim_out=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1], alpha=alpha))
        self.extra_layer2 = ExtraSampleLayer(embed_dim=embed_dims[2], mlp_ratio=mlp_ratios[2])
        self.extra_block2 = MyBlock(
                dim=embed_dims[2], dim_out=embed_dims[2], num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+1],
                norm_layer=norm_layer, sr_ratio=sr_ratios[1], alpha=alpha)
        self.extra_down2 = DownLayer(sample_num=sample_num - N_grid, embed_dim=embed_dims[2], drop_rate=drop_rate,
                      down_block=MyBlock(
                          dim=embed_dims[2], dim_out=embed_dims[2], num_heads=num_heads[2],
                          mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+2],
                          norm_layer=norm_layer, sr_ratio=sr_ratios[1], alpha=alpha))
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], alpha=alpha)
            for i in range(3, depths[2])])
        cur += depths[2]

        # stage 4
        sample_num = sample_num // 4
        self.down_layers3 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[3], dim_out=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2], alpha=alpha))
        self.extra_layer3 = ExtraSampleLayer(embed_dim=embed_dims[3], mlp_ratio=mlp_ratios[3])
        self.extra_block3 = MyBlock(
                dim=embed_dims[3], dim_out=embed_dims[3], num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+1],
                norm_layer=norm_layer, sr_ratio=sr_ratios[2], alpha=alpha)
        self.extra_down3 = DownLayer(sample_num=sample_num - N_grid, embed_dim=embed_dims[3], drop_rate=drop_rate,
                      down_block=MyBlock(
                          dim=embed_dims[3], dim_out=embed_dims[3], num_heads=num_heads[3],
                          mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur+2],
                          norm_layer=norm_layer, sr_ratio=sr_ratios[2], alpha=alpha))
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], alpha=alpha)
            for i in range(3, depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # cls_token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.num = 0

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        self.down_layers1.block.drop_path.drop_prob = dpr[cur]
        self.extra_block1.drop_path.drop_prob = dpr[cur+1]
        self.extra_down1.block.drop_path.drop_prob = dpr[cur+2]
        cur += 3
        for i in range(self.depths[1] - 3):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1] - 3
        self.down_layers2.block.drop_path.drop_prob = dpr[cur]
        self.extra_block2.drop_path.drop_prob = dpr[cur+1]
        self.extra_down2.block.drop_path.drop_prob = dpr[cur+2]
        cur += 3
        for i in range(self.depths[2] - 3):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2] - 3
        self.down_layers3.block.drop_path.drop_prob = dpr[cur]
        self.extra_block3.drop_path.drop_prob = dpr[cur+1]
        self.extra_down3.block.drop_path.drop_prob = dpr[cur+2]
        cur += 3
        for i in range(self.depths[3] - 3):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]
        device = x.device
        outs = []
        img = x
        x = self.ave_layer(x)

        # stage 1 Unchanged
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1.flatten(start_dim=1, end_dim=2)
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)

        # stage 2
        y_map, x_map = torch.meshgrid(torch.arange(H, device=device).float() / (H - 1),
                                      torch.arange(W, device=device).float() / (W - 1))
        xy_map = torch.stack((x_map, y_map), dim=-1)
        loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])

        # split into grid and adaptive tokens
        pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        tmp = pos.reshape([H, W])
        grid_stride = self.grid_stride
        pos_grid = tmp[grid_stride // 2:H:grid_stride, grid_stride // 2:W:grid_stride]
        pos_grid = pos_grid.reshape([-1])
        mask = torch.ones(pos.shape, dtype=torch.bool, device=pos.device)
        mask[pos_grid] = 0
        pos_ada = torch.masked_select(pos, mask)

        x_grid = torch.index_select(x, 1, pos_grid)
        x_ada = torch.index_select(x, 1, pos_ada)
        loc_grid = torch.index_select(loc, 1, pos_grid)
        loc_ada = torch.index_select(loc, 1,  pos_ada)

        x = torch.cat([x_grid, x_ada], 1)
        loc = torch.cat([loc_grid, loc_ada], 1)
        N_grid = x_grid.shape[1]

        if vis:
            outs.append((x, loc, [H, W]))

        # stage 2
        x, loc = self.down_layers1(x, loc, self.pos_embed2, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x, loc = self.extra_layer1(x, loc, img, self.pos_embed2, H, W, kernel_size=5)
        if vis:
            outs.append((x, loc, [H, W]))
        x = self.extra_block1(x, x, loc, H, W)
        x, loc = self.extra_down1(x, loc, None, H, W, self.pos_size, N_grid)
        for blk in self.block2:
            x = blk(x, x, loc, H, W)


        # stage 3
        x, loc = self.down_layers2(x, loc, self.pos_embed3, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x, loc = self.extra_layer2(x, loc, img, self.pos_embed3, H, W, kernel_size=3)
        if vis:
            outs.append((x, loc, [H, W]))
        x = self.extra_block2(x, x, loc, H, W)
        x, loc = self.extra_down2(x, loc, None, H, W, self.pos_size, N_grid)
        for blk in self.block3:
            x = blk(x, x, loc, H, W)

        # stage 4
        x, loc = self.down_layers3(x, loc, self.pos_embed4, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x, loc = self.extra_layer3(x, loc, img, self.pos_embed4, H, W, kernel_size=1)
        if vis:
            outs.append((x, loc, [H, W]))
            # show_tokens(img, outs, N_grid)
            if self.num % 1 == 0:
                show_tokens(img, outs, N_grid)
            self.num = self.num + 1

        x = self.extra_block3(x, x, loc, H, W)
        x, loc = self.extra_down3(x, loc, None, H, W, self.pos_size, N_grid)
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.block4:
            x = blk(x, x, loc, H, W)

        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def mypvt18_small(pretrained=False, **kwargs):
    model = MyPVT18(
        img_size=448,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def mypvt18_small_2(pretrained=False, **kwargs):
    model = MyPVT18(
        img_size=224,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model




class MyPVT17(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], alpha=1):
        super().__init__()
        self.ave_layer = nn.AvgPool2d(2)
        img_size = img_size // 2

        self.num_classes = num_classes
        self.depths = depths
        self.alpha = alpha
        self.grid_stride = sr_ratios[0]
        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        # pos_embed
        H = W = img_size // patch_size
        self.pos_embed1 = nn.Parameter(torch.zeros(1, H, W, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, H // 2, W // 2, embed_dims[1]))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, H // 4, W // 4, embed_dims[2]))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, H // 8, W // 8, embed_dims[3]))
        self.pos_size = [img_size // patch_size, img_size // patch_size]

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        sample_num = self.patch_embed1.num_patches
        cur = 0
        # N_grid = sample_num // self.grid_stride // self.grid_stride
        N_grid = (H // self.grid_stride) * (W // self.grid_stride)

        # stage 1
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])  # , alpha=alpha)
            for i in range(depths[0])])
        cur += depths[0]

        # stage 2
        sample_num = sample_num // 4
        self.down_layers1 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[0], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], dim_out=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0], alpha=alpha))
        self.extra_layer1 = ExtraSampleLayer(embed_dim=embed_dims[1])

        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], alpha=alpha)
            for i in range(1, depths[1])])
        cur += depths[1]

        # stage 3
        sample_num = sample_num // 4
        self.down_layers2 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], dim_out=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1], alpha=alpha))
        self.extra_layer2 = ExtraSampleLayer(embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], alpha=alpha)
            for i in range(1, depths[2])])
        cur += depths[2]

        # stage 4
        sample_num = sample_num // 4
        self.down_layers3 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[3], dim_out=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2], alpha=alpha))
        self.extra_layer3 = ExtraSampleLayer(embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], alpha=alpha)
            for i in range(1, depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # cls_token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.num = 0

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]
        device = x.device
        outs = []
        img = x
        x = self.ave_layer(x)

        # stage 1 Unchanged
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1.flatten(start_dim=1, end_dim=2)
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)

        # stage 2
        y_map, x_map = torch.meshgrid(torch.arange(H, device=device).float() / (H - 1),
                                      torch.arange(W, device=device).float() / (W - 1))
        xy_map = torch.stack((x_map, y_map), dim=-1)
        loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])

        # split into grid and adaptive tokens
        pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        tmp = pos.reshape([H, W])
        grid_stride = self.grid_stride
        pos_grid = tmp[grid_stride // 2:H:grid_stride, grid_stride // 2:W:grid_stride]
        pos_grid = pos_grid.reshape([-1])
        mask = torch.ones(pos.shape, dtype=torch.bool, device=pos.device)
        mask[pos_grid] = 0
        pos_ada = torch.masked_select(pos, mask)

        x_grid = torch.index_select(x, 1, pos_grid)
        x_ada = torch.index_select(x, 1, pos_ada)
        loc_grid = torch.index_select(loc, 1, pos_grid)
        loc_ada = torch.index_select(loc, 1,  pos_ada)

        x = torch.cat([x_grid, x_ada], 1)
        loc = torch.cat([loc_grid, loc_ada], 1)
        N_grid = x_grid.shape[1]

        if vis:
            outs.append((x, loc, [H, W]))

        # stage 2
        x, loc = self.down_layers1(x, loc, self.pos_embed2, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x, loc = self.extra_layer1(x, loc, img, self.pos_embed2, H, W, kernel_size=5)

        for blk in self.block2:
            x = blk(x, x, loc, H, W)
        if vis:
            outs.append((x, loc, [H, W]))

        # stage 3
        x, loc = self.down_layers2(x, loc, self.pos_embed3, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x, loc = self.extra_layer2(x, loc, img, self.pos_embed3, H, W, kernel_size=3)
        for blk in self.block3:
            x = blk(x, x, loc, H, W)
        if vis:
            outs.append((x, loc, [H, W]))

        # stage 4
        x, loc = self.down_layers3(x, loc, self.pos_embed4, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x, loc = self.extra_layer3(x, loc, img, self.pos_embed4, H, W, kernel_size=1)
        for blk in self.block4:
            x = blk(x, x, loc, H, W)

        if vis:
            outs.append((x, loc, [H, W]))
            # show_tokens(img, outs, N_grid)
            if self.num % 1 == 0:
                show_tokens(img, outs, N_grid)
            self.num = self.num + 1

        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def mypvt17_small(pretrained=False, **kwargs):
    model = MyPVT17(
        img_size=448,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def mypvt17_2_small(pretrained=False, **kwargs):
    model = MyPVT17(
        img_size=224,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model



class MyPVT19(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], alpha=1):
        super().__init__()
        self.ave_layer = nn.AvgPool2d(2)
        img_size = img_size // 2

        self.num_classes = num_classes
        self.depths = depths
        self.alpha = alpha
        self.grid_stride = sr_ratios[0]
        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        # pos_embed
        H = W = img_size // patch_size
        self.pos_embed1 = nn.Parameter(torch.zeros(1, H, W, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, H // 2, W // 2, embed_dims[1]))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, H // 4, W // 4, embed_dims[2]))
        self.pos_embed4 = nn.Parameter(torch.zeros(1, H // 8, W // 8, embed_dims[3]))
        self.pos_size = [img_size // patch_size, img_size // patch_size]

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        sample_num = self.patch_embed1.num_patches
        cur = 0
        # N_grid = sample_num // self.grid_stride // self.grid_stride
        N_grid = (H // self.grid_stride) * (W // self.grid_stride)

        # stage 1
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])  # , alpha=alpha)
            for i in range(depths[0])])
        cur += depths[0]

        # stage 2
        sample_num = sample_num // 4
        self.down_layers1 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[0], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[1], dim_out=embed_dims[1], num_heads=num_heads[1],
                                            mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[0], alpha=alpha))
        self.extra_layer1 = ExtraSampleLayer(embed_dim=embed_dims[1])

        self.block2 = nn.ModuleList([MyBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], alpha=alpha)
            for i in range(1, depths[1])])
        cur += depths[1]

        # stage 3
        sample_num = sample_num // 4
        self.down_layers2 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[1], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[2], dim_out=embed_dims[2], num_heads=num_heads[2],
                                            mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[1], alpha=alpha))
        self.extra_layer2 = ExtraSampleLayer(embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([MyBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], alpha=alpha)
            for i in range(1, depths[2])])
        cur += depths[2]

        # stage 4
        sample_num = sample_num // 4
        self.down_layers3 = DownLayer(sample_num=sample_num-N_grid, embed_dim=embed_dims[2], drop_rate=drop_rate,
                                      down_block=MyBlock(
                                            dim=embed_dims[3], dim_out=embed_dims[3], num_heads=num_heads[3],
                                            mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                            norm_layer=norm_layer, sr_ratio=sr_ratios[2], alpha=alpha))
        self.extra_layer3 = ExtraSampleLayer(embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([MyBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], alpha=alpha)
            for i in range(1, depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # cls_token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.num = 0

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]
        device = x.device
        outs = []
        img = x
        x = self.ave_layer(x)

        # stage 1 Unchanged
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1.flatten(start_dim=1, end_dim=2)
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)

        # stage 2
        y_map, x_map = torch.meshgrid(torch.arange(H, device=device).float() / (H - 1),
                                      torch.arange(W, device=device).float() / (W - 1))
        xy_map = torch.stack((x_map, y_map), dim=-1)
        loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])

        # split into grid and adaptive tokens
        pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        tmp = pos.reshape([H, W])
        grid_stride = self.grid_stride
        pos_grid = tmp[grid_stride // 2:H:grid_stride, grid_stride // 2:W:grid_stride]
        pos_grid = pos_grid.reshape([-1])
        mask = torch.ones(pos.shape, dtype=torch.bool, device=pos.device)
        mask[pos_grid] = 0
        pos_ada = torch.masked_select(pos, mask)

        x_grid = torch.index_select(x, 1, pos_grid)
        x_ada = torch.index_select(x, 1, pos_ada)
        loc_grid = torch.index_select(loc, 1, pos_grid)
        loc_ada = torch.index_select(loc, 1,  pos_ada)

        x = torch.cat([x_grid, x_ada], 1)
        loc = torch.cat([loc_grid, loc_ada], 1)
        N_grid = x_grid.shape[1]

        if vis:
            outs.append((x, loc, [H, W]))

        # stage 2
        x, loc = self.down_layers1(x, loc, self.pos_embed2, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x_e, loc_e = self.extra_layer1(x, loc, img, self.pos_embed2, H, W, kernel_size=5)
        # x_e, loc_e = x, loc
        for n, blk in enumerate(self.block2):
            if n == 0:
                x = blk(x, x_e, loc_e, H, W)
            else:
                x = blk(x, x, loc, H, W)
        if vis:
            outs.append((x_e, loc_e, [H, W]))

        # stage 3
        x, loc = self.down_layers2(x, loc, self.pos_embed3, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x_e, loc_e = self.extra_layer2(x, loc, img, self.pos_embed3, H, W, kernel_size=3)
        # x_e, loc_e = x, loc
        for n, blk in enumerate(self.block3):
            if n == 0:
                x = blk(x, x_e, loc_e, H, W)
            else:
                x = blk(x, x, loc, H, W)
        if vis:
            outs.append((x_e, loc_e, [H, W]))

        # stage 4
        x, loc = self.down_layers3(x, loc, self.pos_embed4, H, W, self.pos_size, N_grid)     # down sample
        H, W = H // 2, W // 2
        x_e, loc_e = self.extra_layer3(x, loc, img, self.pos_embed4, H, W, kernel_size=1)
        # x_e, loc_e = x, loc
        for n, blk in enumerate(self.block4):
            if n == 0:
                x = blk(x, x_e, loc_e, H, W)
            else:
                x = blk(x, x, loc, H, W)

        if vis:
            outs.append((x_e, loc_e, [H, W]))
            if self.num % 1 == 0:
                show_tokens(img, outs, N_grid)
            self.num = self.num + 1

        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def mypvt19_small(pretrained=False, **kwargs):
    model = MyPVT19(
        img_size=448,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def mypvt19_2_small(pretrained=False, **kwargs):
    model = MyPVT19(
        img_size=224,
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


# For test
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = mypvt19_small(drop_path_rate=0.1).to(device)
    model.reset_drop_path(0.1)

    empty_input = torch.rand([2, 3, 448, 448], device=device)
    del device

    output = model(empty_input)
    tmp = output.sum()
    print(tmp)

    print('Finish')

