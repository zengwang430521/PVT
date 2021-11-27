import torch
from torch_sparse import spmm
from mmcv.utils import get_logger
from mmcv.runner import _load_checkpoint, load_state_dict
import logging
import re
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].


    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}
    # load state_dict
    _ = load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger


def get_grid_loc(B, H, W, device):
    y_g, x_g = torch.arange(H, device=device).float(), torch.arange(W, device=device).float()
    y_g = 2 * ((y_g + 0.5) / H) - 1
    x_g = 2 * ((x_g + 0.5) / W) - 1
    y_map, x_map = torch.meshgrid(y_g, x_g)
    xy_map = torch.stack((x_map, y_map), dim=-1)

    loc = xy_map.reshape(-1, 2)[None, ...].repeat([B, 1, 1])
    return loc


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


def token2map(x, loc, loc_orig, idx_agg, map_size, weight=None):
    H, W = map_size
    B, N, C = x.shape
    N0 = loc_orig.shape[1]
    device = x.device
    loc_orig = loc_orig.clamp(-1, 1)
    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    loc_orig = loc_orig.round().long()
    loc_orig[..., 0] = loc_orig[..., 0].clamp(0, W-1)
    loc_orig[..., 1] = loc_orig[..., 1].clamp(0, H-1)
    idx_HW_orig = loc_orig[..., 0] + loc_orig[..., 1] * W
    idx_HW_orig = idx_HW_orig + torch.arange(B)[:, None].to(device) * H * W

    idx_tokens = idx_agg + torch.arange(B)[:, None].to(device) * N

    coor = torch.stack([idx_HW_orig, idx_tokens], dim=0).reshape(2, B*N0)
    if weight is None:
        weight = x.new_ones(B, N, 1)
    value = index_points(weight, idx_agg).reshape(B*N0)

    all_weight = spmm(coor, value, B*H*W, B*N, x.new_ones(B*N, 1)) + 1e-6
    value = value / all_weight[idx_HW_orig.reshape(-1), 0]

    x_out = spmm(coor, value, B*H*W, B*N, x.reshape(B*N, C))
    x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    all_weight = all_weight.reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()

    return x_out, all_weight


def map2token(feature_map, N, loc_orig, idx_agg, agg_weight=None):

    dtype = feature_map.dtype
    B, C, H, W = feature_map.shape
    device = feature_map.device
    N0 = loc_orig.shape[1]

    if N0 == N and N == H * W:
        return feature_map.flatten(2).permute(0, 2, 1).contiguous()

    loc_orig = 0.5 * (loc_orig + 1) * torch.FloatTensor([W, H]).to(device)[None, None, :] - 0.5
    x = loc_orig[:, :, 0]
    y = loc_orig[:, :, 1]

    h, w = H, W
    x_grid = x.round().long().clamp(min=0, max=w - 1)
    y_grid = y.round().long().clamp(min=0, max=h - 1)
    idx_HW_orig = (y_grid * w + x_grid).detach()
    index_batch = torch.arange(B, device=device)[:, None].expand(B, N0)

    # use sparse matrix
    idx_agg = idx_agg + index_batch * N
    idx_HW_orig = idx_HW_orig + index_batch * H * W

    indices = torch.stack([idx_agg, idx_HW_orig], dim=0).reshape(2, -1)

    if agg_weight is None:
        value = torch.ones(B * N0, device=feature_map.device, dtype=feature_map.dtype)
    else:
        value = agg_weight.reshape(B * N0) #.type(torch.float32)

    all_weight = spmm(indices, value, B*N, B*H*W, feature_map.new_ones([B*H*W, 1])) + 1e-6
    value = value / all_weight[idx_agg.reshape(-1), 0]
    out = spmm(indices, value, B*N, B*H*W,
               feature_map.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C))
    out = out.reshape(B, N, C)
    return out


def token_downup(target_dict, source_dict):
    x_s = source_dict['x']
    x_t = target_dict['x']
    idx_agg_s = source_dict['idx_agg']
    idx_agg_t = target_dict['idx_agg']
    agg_weight_t = target_dict['agg_weight']
    B, T, C = x_t.shape
    B, S, C = x_s.shape
    N0 = idx_agg_s.shape[1]

    idx_agg_t = idx_agg_t + torch.arange(B, device=x_s.device)[:, None] * T
    idx_agg_s = idx_agg_s + torch.arange(B, device=x_s.device)[:, None] * S

    coor = torch.stack([idx_agg_t, idx_agg_s], dim=0).reshape(2, B*N0)
    weight = agg_weight_t
    if weight is None:
        weight = x_s.new_ones(B, N0, 1)
    weight = weight.reshape(-1)

    all_weight = spmm(coor, weight, B*T, B*S, x_s.new_ones(B*S, 1)) + 1e-6
    weight = weight / all_weight[(idx_agg_t).reshape(-1), 0]
    x_out = spmm(coor, weight, B*T, B*S, x_s.reshape(B*S, C))
    x_out = x_out.reshape(B, T, C)
    return x_out


def DPC_flops(N, C):
    flops = 0
    flops += N * N * C  # dist_matrix
    flops += N * 5  # density
    flops += N * N  # dist indicator
    flops += N * C  # gather
    return flops


def map2token_flops(N0, C):
    return N0 * (2 + 1 + 1 + C)


def token2map_flops(N0, C):
    return N0 * (2 + 1 + 1 + C)


def downup_flops(N0, C):
    return N0 * (2 + 1 + 1 + C)


# flops for attention
def sra_flops(h, w, r, dim):
    return 2 * h * w * (h // r) * (w // r) * dim


def gumble_top_k(x, k, dim, T=1, p_value=1e-6):
    # Noise
    noise = torch.rand_like(x)
    noise = -1 * (noise + p_value).log()
    noise = -1 * (noise + p_value).log()
    # add
    x = x / T + noise
    _, index_k = torch.topk(x, k, dim)
    return index_k



# DPC-KNN based token clustering and token feature averaging
def token_cluster_merge(x, Ns, idx_agg, weight=None, return_weight=False, k=5):
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape

    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        dist_matrix = torch.cdist(x, x)
        # normalize dist_matrix for stable
        dist_matrix = dist_matrix / (dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] + 1e-6)

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)

        # get distance indicator
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1-mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)
        idx_agg_t = dist_matrix.argmin(dim=1)

        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg


from function import f_distance
def token_cluster_hir(x, Ns, idx_agg, conf, weight=None, return_weight=False, **kwargs):
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    conf = conf.squeeze(-1)
    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        index_down = gumble_top_k(conf, Ns, dim=1)

        if N <= 256:
            '''nearest assign'''
            centers = index_points(x, index_down)
            dist_matrix = torch.cdist(x, centers)
            idx_agg_t = dist_matrix.argmin(dim=2)
        else:

            Nr = int(math.sqrt(Ns))
            K = int(2 * Ns / Nr)
            index_rough_center = index_down[:, :Nr]

            centers = index_points(x, index_down)
            rough_centers = index_points(x, index_rough_center)

            dist_matrix1 = torch.cdist(rough_centers, centers, p=2)
            _, idx_k_rough = torch.topk(-dist_matrix1, k=K, dim=-1)

            idx_tmp = torch.cdist(x, rough_centers, p=2).argmin(axis=2)
            idx_k = index_points(idx_k_rough, idx_tmp)

            with torch.cuda.amp.autocast(enabled=False):
                '''I only support float, float, int Now'''
                dist_k = f_distance(x.float(), centers.float(), idx_k.int())

            idx_tmp = dist_k.argmin(dim=2)
            idx_agg_t = torch.gather(idx_k, -1, idx_tmp[:,:, None])
            idx_agg_t = idx_agg_t.squeeze(-1)

        # make sure selected tokens merge to itself
        if index_down is not None:
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
            idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
            idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns


    # # # for debug only
    # loc_orig = get_grid_loc(x.shape[0], 56, 56, x.device)
    # show_conf_merge(density[:, :, None], None, loc_orig, idx_agg, n=1, vmin=None)
    # show_conf_merge(dist[:, :, None], None, loc_orig, idx_agg, n=2, vmin=None)
    # show_conf_merge(score[:, :, None], None, loc_orig, idx_agg, n=3, vmin=None)
    # show_conf_merge(conf[:, :, None], None, loc_orig, idx_agg, n=4, vmin=None)
    # if use_conf:
    #     show_conf_merge(score_log[:, :, None], None, loc_orig, idx_agg, n=5)


    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg


# use dpc to determine center and use hir for cluster
# just for comparison
def token_cluster_dpc_hir(x, Ns, idx_agg, weight=None, return_weight=False, k=5):
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape

    if weight is None:
        weight = x.new_ones(B, N, 1)

    with torch.no_grad():
        dist_matrix = torch.cdist(x, x)
        # normalize dist_matrix for stable
        dist_matrix = dist_matrix / (dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] + 1e-6)

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)

        # get distance indicator
        dist, index_parent = (dist_matrix * mask +
                              dist_matrix.flatten(1).max(dim=-1)[0][:, None, None] * (1-mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=Ns, dim=-1)

        # if N <= 256:
        if N < 0:
            # assign tokens to the nearest center
            dist_matrix = index_points(dist_matrix, index_down)
            idx_agg_t = dist_matrix.argmin(dim=1)
        else:
            # assign tokens to the nearest center use hir way
            Nr = int(math.sqrt(Ns))
            K = int(2 * Ns / Nr)
            index_rough_center = index_down[:, :Nr]

            centers = index_points(x, index_down)
            rough_centers = index_points(x, index_rough_center)

            dist_matrix1 = torch.cdist(rough_centers, centers, p=2)
            _, idx_k_rough = torch.topk(-dist_matrix1, k=K, dim=-1)

            idx_tmp = torch.cdist(x, rough_centers, p=2).argmin(axis=2)
            idx_k = index_points(idx_k_rough, idx_tmp)

            with torch.cuda.amp.autocast(enabled=False):
                '''I only support float, float, int Now'''
                dist_k = f_distance(x.float(), centers.float(), idx_k.int())

            idx_tmp = dist_k.argmin(dim=2)
            idx_agg_t = torch.gather(idx_k, -1, idx_tmp[:, :, None])
            idx_agg_t = idx_agg_t.squeeze(-1)


        # make sure selected centers merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
        idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
        idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns


    # normalize the weight
    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    # average token features
    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)
    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg


def token_cluster_lsh(x, Ns, idx_agg, weight=None, return_weight=False,  **kwargs):
    dtype = x.dtype
    device = x.device
    B, N, C = x.shape
    if weight is None:
        weight = x.new_ones(B, N, 1)

    Nbit = math.ceil(math.log2(Ns))
    Ns = 2**Nbit
    with torch.no_grad():
        weight_proj = torch.rand([C, Nbit], dtype=dtype, device=device)
        x_proj = torch.matmul((x - x.mean(dim=-1, keepdim=True)), weight_proj)
        tmp = 2**torch.arange(Nbit, device=device)
        idx_agg_t = ((x_proj > 0) * tmp[None, None, :]).sum(dim=-1)

        index_down = None
        # make sure selected tokens merge to itself
        if index_down is not None:
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, Ns)
            idx_tmp = torch.arange(Ns, device=x.device)[None, :].expand(B, Ns)
            idx_agg_t[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        idx = idx_agg_t + torch.arange(B, device=x.device)[:, None] * Ns

    all_weight = weight.new_zeros(B * Ns, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N), source=weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = weight / all_weight[idx]

    x_out = x.new_zeros(B * Ns, C)
    source = x * norm_weight
    x_out.index_add_(dim=0, index=idx.reshape(B * N), source=source.reshape(B * N, C).type(x.dtype))
    x_out = x_out.reshape(B, Ns, C)

    idx_agg = index_points(idx_agg_t[..., None], idx_agg).squeeze(-1)

    if return_weight:
        weight_t = index_points(norm_weight, idx_agg)
        return x_out, idx_agg, weight_t
    return x_out, idx_agg




def show_tokens_merge(x, out, count=0):
    # import matplotlib.pyplot as plt
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    x = x * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
    save_x = False
    save_img = False
    save_fig = True

    if save_x:
        save_dict = {
            'x': x,
            'out': out
        }
        fname = f'vis/{count}.pth'
        torch.save(save_dict, fname)

    B, _, h, w = x.shape
    h, w = h // 4, w//4
    device = x.device
    color_map = F.avg_pool2d(x, kernel_size=4)


    N0 = h*w

    for i in range(1):
        img = x[i].permute(1, 2, 0).detach().cpu()
        ax = plt.subplot(1, 6, 1)
        ax.clear()
        ax.imshow(img)

        if save_img:
            fname = f'vis/{count}_img.png'
            import cv2
            cv2.imwrite(fname, img.numpy()[:, :, ::-1] * 255)

        lv = 3
        x = out[lv]['x']
        idx_agg = out[lv]['idx_agg']
        loc_orig = out[lv]['loc_orig']

        B, N, _ = x.shape

        tmp = torch.arange(N, device=x.device)[None, :, None].expand(B, N, 1).float()
        H, W, _ = img.shape
        idx_map, _ = token2map(tmp, loc_orig, loc_orig, idx_agg, [H // 4, W // 4])
        idx_map = F.interpolate(idx_map, [H, W], mode='nearest')
        # idx_map = idx_map[0].permute(1, 2, 0).detach().cpu().float()
        ax = plt.subplot(1, 6, 6)
        ax.imshow(idx_map[0].permute(1, 2, 0).detach().cpu().float())

        for lv in range(len(out)):

            x = out[lv]['x']
            idx_agg = out[lv]['idx_agg']
            loc_orig = out[lv]['loc_orig']
            agg_weight = out[lv]['agg_weight']
            B, N, _ = x.shape

            token_c = map2token(color_map, N, loc_orig, idx_agg, agg_weight)
            idx_map, _ = token2map(token_c, loc_orig, loc_orig, idx_agg, [H // 4, W // 4])
            idx_map_grid = F.avg_pool2d(color_map, kernel_size=2**lv)

            idx_map_our = idx_map
            idx_map_our = F.interpolate(idx_map, [H*4, W*4], mode='nearest')
            idx_map_grid = F.interpolate(idx_map_grid, [H * 4, W * 4], mode='nearest')

            sharpen = torch.FloatTensor([   [0, -1, 0],
                                            [-1, 4, -1],
                                            [0, -1, 0]])
            sharpen = sharpen[None, None, :, :].to(idx_map.device).expand([3,1,3,3])

            mask_our = F.conv2d(F.pad(idx_map_our, [1, 1, 1, 1], mode='replicate'), sharpen, groups=3)
            mask_grid = F.conv2d(F.pad(idx_map_grid, [1, 1, 1, 1], mode='replicate'), sharpen, groups=3)

            mask_our = (mask_our.abs() > 0).float()
            mask_grid = (mask_grid.abs() > 0).float()
            # for t in range(lv - 1):
            for t in range(1):
                kernel = torch.FloatTensor([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])
                kernel = kernel[None, None, :, :].to(idx_map.device).expand([3, 1, 3, 3])

                mask_our = F.conv2d(F.pad(mask_our, [1, 1, 1, 1], mode='replicate'), kernel, groups=3)
                mask_grid = F.conv2d(F.pad(mask_grid, [1, 1, 1, 1], mode='replicate'), kernel, groups=3)

            idx_map_our = (idx_map_our + mask_our * 10).clamp(0, 1)
            idx_map_grid = (idx_map_grid + mask_grid * 10).clamp(0, 1)

            if save_img:
                fname = f'vis/{count}_{lv}.png'
                import cv2
                cv2.imwrite(fname, idx_map_our[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1] * 255)

                fname = f'vis/{count}_{lv}_grid.png'
                import cv2
                cv2.imwrite(fname, idx_map_grid[0].permute(1, 2, 0).detach().cpu().float().numpy()[:, :, ::-1] * 255)

            ax = plt.subplot(1, 6, lv+2)
            ax.clear()
            ax.imshow(idx_map_our[0].permute(1, 2, 0).detach().cpu().float())

    # plt.show()
    if save_fig:
        fname = f'vis/{count}.jpg'
        plt.savefig(fname, dpi=200)


    return

