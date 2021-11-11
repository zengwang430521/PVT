import argparse
import torch
from timm.models import create_model
import pvt
import pvt_v2
import math
import train

try:
    from mmcv.cnn import get_model_complexity_info
    from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Get FLOPS of a classification model')
    parser.add_argument('model', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def sra_flops(h, w, r, dim):
    return 2 * h * w * (h // r) * (w // r) * dim


def li_sra_flops(h, w, dim):
    return 2 * h * w * 7 * 7 * dim


def get_cluster_flops(model, input_shape):
    flops = 0
    dims = [64, 128, 320, 512]
    _, H, W = input_shape
    N = H // 4 * W // 4

    for i in range(3):
        C = dims[i+1]
        flops += N * N * C      # dist_matrix
        flops += N * 5          # density
        flops += N * N          # dist
        flops += N * C          # gather
        N = N // 4
    return flops


'''
token2map nad map2token all have N0 * (2 + 1 + 1 + C) flops
'''

def map2token_flops(N0, C):
    return N0 * (2 + 1 + 1 + C)

def token2map_flops(N0, C):
    return N0 * (2 + 1 + 1 + C)


def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)
    if 'den' in model.name:
        tmp = get_cluster_flops(model, input_shape)
        flops += tmp

        mlp_ratios = [8, 8, 4, 4]
        # token and map
        tmp = 0
        _, H, W = input_shape
        N0 = H // 4 * W // 4

        dim, mlp_r, num = model.block2[0].attn.dim, 8, len(model.block2) + 1
        tmp += (map2token_flops(N0, dim) + map2token_flops(N0, dim*mlp_r) + token2map_flops(N0, dim*mlp_r)) * num

        dim, mlp_r, num = model.block3[0].attn.dim, 4, len(model.block3) + 1
        tmp += (map2token_flops(N0, dim) + map2token_flops(N0, dim*mlp_r) + token2map_flops(N0, dim*mlp_r)) * num

        dim, mlp_r, num = model.block4[0].attn.dim, 4, len(model.block4) + 1
        tmp += (map2token_flops(N0, dim*mlp_r) + token2map_flops(N0, dim*mlp_r)) * num

        flops += tmp

        if 'li' in model.name:  # calculate flops of PVTv2_li
            stage1 = li_sra_flops(H // 4, W // 4,
                                  model.block1[0].attn.dim) * (len(model.block1) + 1)
            stage2 = li_sra_flops(H // 8, W // 8,
                                  model.block2[0].attn.dim) * (len(model.block2)+1)
            stage3 = li_sra_flops(H // 16, W // 16,
                                  model.block3[0].attn.dim) * (len(model.block3)+1)
            stage4 = li_sra_flops(H // 32, W // 32,
                                  model.block4[0].attn.dim) * (len(model.block4)+1)
        else:  # calculate flops of PVT/PVTv2
            stage1 = sra_flops(H // 4, W // 4,
                               model.block1[0].attn.sr_ratio,
                               model.block1[0].attn.dim) * (len(model.block1)+1)
            stage2 = sra_flops(H // 8, W // 8,
                               model.block2[0].attn.sr_ratio,
                               model.block2[0].attn.dim) * (len(model.block2)+1)
            stage3 = sra_flops(H // 16, W // 16,
                               model.block3[0].attn.sr_ratio,
                               model.block3[0].attn.dim) * (len(model.block3)+1)
            stage4 = sra_flops(H // 32, W // 32,
                               model.block4[0].attn.sr_ratio,
                               model.block4[0].attn.dim) * (len(model.block4)+1)
        flops += stage1 + stage2 + stage3 + stage4


    # if 'my' in model.name:
    #     flops += get_mypvt_flops(model, input_shape)
    # elif 'den' in model.name:
    #     flops += get_cluster_flops(model, input_shape)

    elif 'pvt' in model.name:
        _, H, W = input_shape
        if 'li' in model.name:  # calculate flops of PVTv2_li
            stage1 = li_sra_flops(H // 4, W // 4,
                                  model.block1[0].attn.dim) * len(model.block1)
            stage2 = li_sra_flops(H // 8, W // 8,
                                  model.block2[0].attn.dim) * len(model.block2)
            stage3 = li_sra_flops(H // 16, W // 16,
                                  model.block3[0].attn.dim) * len(model.block3)
            stage4 = li_sra_flops(H // 32, W // 32,
                                  model.block4[0].attn.dim) * len(model.block4)
        else:  # calculate flops of PVT/PVTv2
            stage1 = sra_flops(H // 4, W // 4,
                               model.block1[0].attn.sr_ratio,
                               model.block1[0].attn.dim) * len(model.block1)
            stage2 = sra_flops(H // 8, W // 8,
                               model.block2[0].attn.sr_ratio,
                               model.block2[0].attn.dim) * len(model.block2)
            stage3 = sra_flops(H // 16, W // 16,
                               model.block3[0].attn.sr_ratio,
                               model.block3[0].attn.dim) * len(model.block3)
            stage4 = sra_flops(H // 32, W // 32,
                               model.block4[0].attn.sr_ratio,
                               model.block4[0].attn.dim) * len(model.block4)
        flops += stage1 + stage2 + stage3 + stage4


    return flops_to_string(flops), params_to_string(params)


def my_sra_flops(n1, n2, dim, num_heads):
    dim_h = dim / num_heads
    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads
    return f1 + f2


def get_map2token_flops(N, dim, h, w):
    # grid-sample
    flops = N * dim * 4
    return flops


def get_token2map_flops(N, dim, h, w, k):
    flops = 0

    # location process (may not accurate)
    flops += 2 * N * (1 + 1 + 1 + 1 + 2 + 1)

    # index-add
    flops += N * (dim + 1)

    # norm
    flops += h * w * dim * 2

    # inter
    if k >= 3:
        # gaussian filtering
        flops += h * w * (dim+1) * (k*k)

        # add
        flops += h * w * dim * 5

    return flops


def myblock_flops(N, N_source, dim, num_heads, H, W, sr, kernel, flag_g=False):
    flops = 0

    # Att block
    if flag_g and sr <= 1:
        N2 = N_source
    else:
        # token2map
        h, w = H // sr, W // sr
        flops += get_token2map_flops(N_source, dim, h, w, 1)
        N2 = h * w

    # Att
    flops += my_sra_flops(N, N2, dim, num_heads)

    # MyMlp block
    # token2map
    flops += get_token2map_flops(N, dim, H, W, kernel)
    # map2token
    flops += get_map2token_flops(N, dim, H, W)

    return flops


def down_layer_flops(N, dim, H, W, N_grid, sample_ratio, kernel):
    flops = 0
    # token2map
    flops += get_token2map_flops(N, dim, H, W, kernel)
    # map2token
    flops += get_map2token_flops(N, dim, H, W,)
    # gumble top-k
    flops += N
    return flops


def get_mypvt_flops(net, input_shape, flag_g=False):
    _, H, W = input_shape

    # for blocks
    # stage1 is unchanged
    stage1 = sra_flops(H // 4, W // 4,
                       net.block1[0].attn.sr_ratio,
                       net.block1[0].attn.dim) * len(net.block1)

    H, W = H // 4, W // 4
    N = H * W
    dim = net.block1[0].attn.dim
    N_grid = (H // net.grid_stride) * (W // net.grid_stride)

    # stage2
    # dim_in = dim
    # Ns = N
    # N = max(math.ceil(Ns * net.down_layers1.sample_ratio), 0) + N_grid
    # N = max(math.ceil((Ns-N_grid) * net.down_layers3.sample_ratio), 0) + N_grid

    dim_in = dim
    Ns = N
    sample_ratio = net.down_layers1.sample_ratio
    sample_num = max(math.ceil(Ns * sample_ratio) - N_grid, 0)
    if sample_num == 0:
        sample_num = max(math.ceil(Ns * sample_ratio), 0)
    N = sample_num + N_grid

    sr = net.down_layers1.block.attn.sr_ratio
    dim = net.down_layers1.block.attn.dim
    num_heads = net.down_layers1.block.attn.num_heads
    kernel = sr + 1
    stage2 = down_layer_flops(N, dim_in, H, W, N_grid, sr, kernel)
    stage2 += myblock_flops(N, Ns, dim, num_heads, H, W, sr, kernel, flag_g)

    sr = net.block2[0].attn.sr_ratio
    dim = net.block2[0].attn.dim
    num_heads = net.block2[0].attn.num_heads
    kernel = sr + 1
    H, W = H // 2, W // 2
    stage2 += myblock_flops(N, N, dim, num_heads, H, W, sr, kernel, flag_g) * len(net.block2)


    # stage3
    dim_in = dim
    Ns = N
    sample_ratio = net.down_layers2.sample_ratio
    sample_num = max(math.ceil(Ns * sample_ratio) - N_grid, 0)
    if sample_num == 0:
        sample_num = max(math.ceil(Ns * sample_ratio), 0)
    N = sample_num + N_grid


    sr = net.down_layers2.block.attn.sr_ratio
    dim = net.down_layers2.block.attn.dim
    num_heads = net.down_layers2.block.attn.num_heads
    kernel = sr + 1
    stage3 = down_layer_flops(N, dim_in, H, W, N_grid, sr, kernel)
    stage3 += myblock_flops(N, Ns, dim, num_heads, H, W, sr, kernel, flag_g)

    sr = net.block3[0].attn.sr_ratio
    dim = net.block3[0].attn.dim
    num_heads = net.block3[0].attn.num_heads
    kernel = sr + 1
    H, W = H // 2, W // 2
    stage3 += myblock_flops(N, N, dim, num_heads, H, W, sr, kernel, flag_g) * len(net.block3)

    # stage4
    dim_in = dim
    Ns = N
    sample_ratio = net.down_layers3.sample_ratio
    sample_num = max(math.ceil(Ns * sample_ratio) - N_grid, 0)
    if sample_num == 0:
        sample_num = max(math.ceil(Ns * sample_ratio), 0)
    N = sample_num + N_grid


    sr = net.down_layers3.block.attn.sr_ratio
    dim = net.down_layers3.block.attn.dim
    num_heads = net.down_layers3.block.attn.num_heads
    kernel = sr + 1
    stage4 = down_layer_flops(N, dim_in, H, W, N_grid, sr, kernel)
    stage4 += myblock_flops(N, Ns, dim, num_heads, H, W, sr, kernel, flag_g)

    sr = net.block4[0].attn.sr_ratio
    dim = net.block4[0].attn.dim
    num_heads = net.block4[0].attn.num_heads
    kernel = sr + 1
    H, W = H // 2, W // 2
    stage4 += myblock_flops(N, N, dim, num_heads, H, W, sr, kernel, flag_g) * len(net.block4)

    print(stage1 + stage2 + stage3 + stage4)
    return stage1 + stage2 + stage3 + stage4

    # return flops_to_string(flops), params_to_string(params)


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000
    )
    model.name = args.model
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    flops, params = get_flops(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
