import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


def get_grid_loc(B, H, W, device):
    y_g, x_g = torch.arange(H, device=device).float(), torch.arange(W, device=device).float()
    y_g = 2 * ((y_g + 0.5) / H) - 1
    x_g = 2 * ((x_g + 0.5) / W) - 1
    y_map, x_map = torch.meshgrid(y_g, x_g)
    xy_map = torch.stack((x_map, y_map), dim=-1)

    loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])
    return loc


def get_loc(x, H, W, grid_stride):
        B = x.shape[0]
        device = x.device
        y_g, x_g = torch.arange(H, device=device).float(), torch.arange(W, device=device).float()
        y_g = 2 * ((y_g + 0.5) / H) - 1
        x_g = 2 * ((x_g + 0.5) / W) - 1
        y_map, x_map = torch.meshgrid(y_g, x_g)
        xy_map = torch.stack((x_map, y_map), dim=-1)

        loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])

        # split into grid and adaptive tokens
        pos = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
        tmp = pos.reshape([H, W])
        pos_grid = tmp[grid_stride // 2:H:grid_stride, grid_stride // 2:W:grid_stride]
        pos_grid = pos_grid.reshape([-1])
        mask = torch.ones(pos.shape, dtype=torch.bool, device=pos.device)
        mask[pos_grid] = 0
        pos_ada = torch.masked_select(pos, mask)

        x_grid = torch.index_select(x, 1, pos_grid)
        x_ada = torch.index_select(x, 1, pos_ada)
        loc_grid = torch.index_select(loc, 1, pos_grid)
        loc_ada = torch.index_select(loc, 1, pos_ada)

        x = torch.cat([x_grid, x_ada], 1)
        loc = torch.cat([loc_grid, loc_ada], 1)
        N_grid = x_grid.shape[1]
        return x, loc, N_grid


def extract_local_feature(src, loc, kernel_size=(3, 3)):
    B, C, H, W = src.shape
    B, N, _ = loc.shape

    h, w = kernel_size
    x = torch.arange(w, device=loc.device, dtype=loc.dtype)
    x = (x - 0.5 * (w-1)) * 2 / W
    y = torch.arange(h, device=loc.device, dtype=loc.dtype)
    y = (y - 0.5 * (h-1)) * 2 / H
    y, x = torch.meshgrid(y, x)
    grid = torch.stack([x, y], dim=-1)
    grid = loc[:, :, None, None, :] + grid[None, None, ...]     # (B, N, h, w, 2)

    loc_feature = F.grid_sample(src, grid.flatten(2, 3))        # (B, C, N, h * w)
    loc_feature = loc_feature.reshape(B, C, N, h, w)            # (B, C, N, h, w)
    loc_feature = loc_feature.permute(0, 2, 1, 3, 4).contiguous()            # (B, N, C, h, w)
    return loc_feature.flatten(0, 1)                            # (B * N, C, h, w)


def extract_neighbor_feature(src, loc, kernel_size=(3, 3)):
    B, C, H, W = src.shape
    B, N, _ = loc.shape

    h, w = kernel_size
    x = torch.arange(w, device=loc.device, dtype=loc.dtype)
    x = (x - 0.5 * (w-1)) * 2 / W
    y = torch.arange(h, device=loc.device, dtype=loc.dtype)
    y = (y - 0.5 * (h-1)) * 2 / H
    y, x = torch.meshgrid(y, x)
    grid = torch.stack([x, y], dim=-1)
    grid = loc[:, :, None, None, :] + grid[None, None, ...]     # (B, N, h, w, 2)
    loc_feature = F.grid_sample(src, grid.flatten(2, 3))        # (B, C, N, h * w)
    loc_feature = loc_feature.permute(0, 2, 3, 1)               # (B, N, h * w, C)
    return loc_feature


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
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).contiguous()
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).contiguous()
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
    if kernel_size < 3:
        return feature
    feature = feature * mask
    out = guassian_filt(torch.cat([feature, mask], dim=1),
                        kernel_size=kernel_size, sigma=sigma)
    C = out.shape[1] - 1
    feature_inter = out[:, :C]
    mask_inter = out[:, C:]
    # tmp = mask_inter.min()
    feature_inter = feature_inter / (mask_inter + 1e-6)
    mask_inter = (mask_inter > 0).float()
    feature_inter = feature_inter * mask_inter
    out = feature + (1 - mask) * feature_inter
    return out


def token2map_partical(x, loc, map_size, conf=None, method=0):
    H, W = map_size
    B, N, C = x.shape
    loc = loc.clamp(-1, 1)
    loc = 0.5 * (loc + 1) * torch.FloatTensor([W, H]).to(loc.device)[None, None, :] - 0.5
    loc = loc.round().long()
    loc[..., 0] = loc[..., 0].clamp(0, W-1)
    loc[..., 1] = loc[..., 1].clamp(0, H-1)
    idx = loc[..., 0] + loc[..., 1] * W
    idx = idx + torch.arange(B)[:, None].to(loc.device) * H * W
    if conf is None:
        out = x.new_zeros(B * H * W, C + 1)
        weight = x.new_ones(B, N, 1)
        tmp = torch.cat([x, weight], dim=-1)
        out.index_add_(dim=0, index=idx.reshape(B*N), source=tmp.reshape(B*N, C+1))
        out = out.reshape(B, H, W, C + 1).permute(0, 3, 1, 2).contiguous()
        feature = out[:, :C, :, :]
        weight = out[:, C:, :, :]
        feature = feature / (weight + 1e-6)
        mask = (weight > 0).float()
    else:
        conf = conf - conf.max(dim=1, keepdim=True)[0]
        if method == 0:
            # 1 as weight, mean feature, mean conf as mask
            out = x.new_zeros(B * H * W, C + 2)
            conf = conf.exp()
            weight = x.new_ones(B, N, 1)
            tmp = torch.cat([x, conf], dim=-1)
            tmp = tmp * weight
            tmp = torch.cat([tmp, weight], dim=-1)
            out.index_add_(dim=0, index=idx.reshape(B * N), source=tmp.reshape(B * N, C + 2))
            out = out.reshape(B, H, W, C + 2).permute(0, 3, 1, 2).contiguous()

            feature = out[:, :C, :, :]
            conf = out[:, C:C+1, :, :]
            weight = out[:, C+1:, :, :]
            feature = feature / (weight + 1e-6)
            mask = conf / (weight + 1e-6)
        elif method == 1:
            # conf as weight, weighted mean feature, weighted mean conf as mask
            out = x.new_zeros(B * H * W, C + 2)
            conf = conf.exp()
            weight = conf
            tmp = torch.cat([x, conf], dim=-1)
            tmp = tmp * weight
            tmp = torch.cat([tmp, weight], dim=-1)
            out.index_add_(dim=0, index=idx.reshape(B * N), source=tmp.reshape(B * N, C + 2))
            out = out.reshape(B, H, W, C + 2).permute(0, 3, 1, 2).contiguous()

            feature = out[:, :C, :, :]
            conf = out[:, C:C+1, :, :]
            weight = out[:, C+1:, :, :]
            feature = feature / (weight + 1e-6)
            mask = conf / (weight + 1e-6)
    return feature, mask


def token2map(x, loc, map_size, kernel_size, sigma, return_mask=False):
    H, W = map_size
    B, N, C = x.shape
    loc = loc.clamp(-1, 1)
    loc = 0.5 * (loc + 1) * torch.FloatTensor([W, H]).to(loc.device)[None, None, :] - 0.5
    loc = loc.round().long()
    loc[..., 0] = loc[..., 0].clamp(0, W-1)
    loc[..., 1] = loc[..., 1].clamp(0, H-1)
    idx = loc[..., 0] + loc[..., 1] * W
    idx = idx + torch.arange(B)[:, None].to(loc.device) * H * W

    out = x.new_zeros(B*H*W, C+1)
    out.index_add_(dim=0, index=idx.reshape(B*N),
                   source=torch.cat([x, x.new_ones(B, N, 1)], dim=-1).reshape(B*N, C+1))
    out = out.reshape(B, H, W, C+1).permute(0, 3, 1, 2).contiguous()
    assert out.shape[1] == C+1
    feature = out[:, :C, :, :]
    mask = out[:, C:, :, :]

    # try:
    #     feature, mask = out[:, :C, :, :], out[:, C:, :, :]
    # except:
    #     info = 'out shape: ' + str(out.shape) + ' C: ' + str(C)
    #     print(info)
    #     print(info)
    #     raise KeyError(info)

    # del out

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
    # loc_xy = loc_xy.type(feature_map.dtype) * 2 - 1
    loc_xy = loc_xy.unsqueeze(1).type(feature_map.dtype)
    tokens = F.grid_sample(feature_map, loc_xy, mode=mode, align_corners=align_corners)
    tokens = tokens.permute(0, 2, 3, 1).squeeze(1).contiguous()
    return tokens


def show_tokens(x, out, N_grid=14*14):
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
            loc = out[lv][1]
            loc = 0.5 * loc + 0.5
            loc_grid = loc[i, :N_grid].detach().cpu().numpy()
            ax.scatter(loc_grid[:, 0], 1 - loc_grid[:, 1], c='blue', s=0.4+lv*0.1)
            loc_ada = loc[i, N_grid:].detach().cpu().numpy()
            ax.scatter(loc_ada[:, 0], 1 - loc_ada[:, 1], c='red', s=0.4+lv*0.1)
    return


def show_conf(conf, loc):
    H = int(conf.shape[1]**0.5)
    if H == 28:
        conf = F.softmax(conf, dim=1)
        conf_map = token2map(conf,  map_size=[H, H], loc=loc, kernel_size=3, sigma=2)
        lv = 3
        ax = plt.subplot(1, 6, lv)
        ax.clear()
        ax.imshow(conf_map[0, 0].detach().cpu())


def token2critcal(x, loc, loc_critical, return_mask=False):
    B, N, C = x.shape
    k = loc_critical.shape[1]
    dists = square_distance(loc, loc_critical)
    idx = dists.argmin(dim=-1)

    idx = idx + torch.arange(B)[:, None].to(loc.device) * k
    out = x.new_zeros(B * k, C + 1)

    out.index_add_(dim=0, index=idx.reshape(B * N),
                   source=torch.cat([x, x.new_ones(B, N, 1)], dim=-1).reshape(B * N, C + 1))
    out = out.reshape(B, k, C + 1)
    feature = out[:, :, :C]
    mask = out[:, :, C:]
    feature = feature / (mask + 1e-6)
    mask = (mask > 0).float()
    feature = feature * mask

    if return_mask:
        return feature, mask
    return feature


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    dist = src.unsqueeze(2) - dst.unsqueeze(1)
    dist = (dist**2).sum(dim=-1)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def inter_points(x_src, loc_src, loc_tar):
    B, N, _ = loc_tar.shape

    dists = square_distance(loc_tar, loc_src)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3]     # [B, N, 3]

    dist_recip = 1.0 / (dists + 1e-6)

    one_mask = dists == 0
    zero_mask = one_mask.sum(dim=-1) > 0
    dist_recip[zero_mask, :] = 0
    dist_recip[one_mask] = 1
    # t = one_mask.max()

    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    x_tar = torch.sum(index_points(x_src, idx) * weight.view(B, N, 3, 1), dim=2)
    return x_tar


def get_critical_idx(x, k=49):
    # x： [B, N, C]
    value, idx = x.max(dim=1)
    tmp = (x >= value[:, None, :]) * x
    tmp, _ = tmp.max(dim=-1)
    _, idx = torch.topk(tmp, k, -1)
    return idx


def get_gaussian_kernel(kernel_size, sigma, device):
    x_coord = torch.arange(kernel_size, device=device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).contiguous()
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).contiguous()
    return gaussian_kernel


def get_sample_grid(weight_map):
    B, _, H, W = weight_map.shape
    max_size = max(H, W)
    device = weight_map.device
    dtype = weight_map.dtype

    kernel_size = 2 * max_size - 1
    pad_size = max_size - 1

    kernel_gaussian = get_gaussian_kernel(kernel_size, sigma=3, device=device)

    h, w = kernel_size, kernel_size
    x = torch.arange(w, device=device, dtype=dtype)
    x = (x - 0.5 * (w-1)) * 2 / W
    y = torch.arange(h, device=device, dtype=dtype)
    y = (y - 0.5 * (h-1)) * 2 / H
    y, x = torch.meshgrid(y, x)
    kernel_delta = torch.stack([x, y], dim=-1)
    kernel_delta = kernel_delta.permute(2, 0, 1).unsqueeze(1)

    kernel = torch.cat([kernel_gaussian * kernel_delta, kernel_gaussian], dim=0)

    weight_map = F.pad(weight_map, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
    tmp = F.conv2d(weight_map, kernel, stride=1, padding=0)
    loc_delta, norm_weight = tmp[:, :2], tmp[:, 2:]
    loc_delta = loc_delta / (norm_weight + 1e-6)

    y_g, x_g = torch.arange(H, device=device).float(), torch.arange(W, device=device).float()
    y_g = 2 * ((y_g + 0.5) / H) - 1
    x_g = 2 * ((x_g + 0.5) / W) - 1
    y_map, x_map = torch.meshgrid(y_g, x_g)
    loc = torch.stack((x_map, y_map), dim=-1)
    loc = loc.permute(2, 0, 1)[None, ...]

    loc = loc + loc_delta
    loc = loc.clamp(-1, 1)
    return loc


def merge_tokens(x, loc, loc_down, weight=None):
    B, N, C = x.shape
    Ns = loc_down.shape[1]

    dists = square_distance(loc, loc_down)
    idx = dists.argmin(axis=2)
    idx = idx + torch.arange(B)[:, None].to(loc.device) * Ns

    if weight is None:
        weight = x.new_ones(B, N, 1)
    tmp = x.new_zeros(B*Ns, C+3)
    source = torch.cat([x * weight, loc * weight, weight], dim=-1)
    source = source.to(x.device).type(x.dtype)
    tmp.index_add_(dim=0, index=idx.reshape(B*N), source=source.reshape(B*N, C+3))
    tmp = tmp.reshape(B, Ns, C+3)

    x_out = tmp[..., :C]
    loc_out = tmp[..., C:C+2]
    norm_weight = tmp[:, :, C+2:]

    # assert norm_weight.min() > 0
    # print(norm_weight.min())
    if norm_weight.min() <= 0:
        print('norm_weight: '); print(norm_weight.min())
        err_idx = (norm_weight <=0).non_zeros()
        print('err_idx: '); print(err_idx)
        bid = err_idx[0, 0]
        print('loc: '); print(loc[bid])
        print('loc down: '); print(loc_down[bid])
        print('idx:'); print(idx[bid])
        print('weight:'); print(weight[bid])
        print('norm_weight:'); print(norm_weight[bid])



    x_out = x_out / (norm_weight + 1e-6)
    loc_out = loc_out / (norm_weight + 1e-6)

    # t1 = weight.min()
    # t2 = norm_weight.min()

    return x_out, loc_out


# '''for debug'''
#
# conf_map = torch.ones(2, 1, 28, 28) * 0.5
# conf_map[0, 0, 7:14, 7:14] = 5
# conf_map[0, 0, 3:6, 10:13] = 10
# conf_map[1, 0, 1, 10] = 5
# # conf_map = torch.rand(2, 1, 28, 28)
# # conf_map = guassian_filt(conf_map)
#
# loc = get_sample_grid(conf_map)
# loc = loc.reshape(2, 2, -1).permute(0, 2,  1)
#
#
# ax = plt.subplot(1, 2, 1)
# ax.imshow(conf_map[0, 0].detach().cpu(), extent=[-1, 1, 1, -1])
# ax.scatter(loc[0, :, 0], loc[0, :, 1], c='red', s=0.5)
#
# ax = plt.subplot(1, 2, 2)
# ax.imshow(conf_map[1, 0].detach().cpu(), extent=[-1, 1, 1, -1])
# ax.scatter(loc[1, :, 0], loc[1, :, 1], c='red', s=0.5)
#
# plt.show()
# t = 0