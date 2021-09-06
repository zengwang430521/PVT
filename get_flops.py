import torchvision.models as models
import pvt
import my_pvt20_2
import my_pvt20_2g
import torch
from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
import math
import pvt_impr8


def mha_flops(h, w, dim, num_heads):
    dim_h = dim / num_heads
    n = h * w
    f1 = n * dim_h * n * num_heads
    f2 = n * n * dim_h * num_heads
    return f1 + f2


def sra_flops(h, w, r, dim, num_heads):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2


def get_pvt_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    stage1 = sra_flops(H // 4, W // 4,
                       net.block1[0].attn.sr_ratio,
                       net.block1[0].attn.dim,
                       net.block1[0].attn.num_heads) * len(net.block1)
    stage2 = sra_flops(H // 8, W // 8,
                       net.block2[0].attn.sr_ratio,
                       net.block2[0].attn.dim,
                       net.block2[0].attn.num_heads) * len(net.block2)
    stage3 = sra_flops(H // 16, W // 16,
                       net.block3[0].attn.sr_ratio,
                       net.block3[0].attn.dim,
                       net.block3[0].attn.num_heads) * len(net.block3)
    stage4 = sra_flops(H // 32, W // 32,
                       net.block4[0].attn.sr_ratio,
                       net.block4[0].attn.dim,
                       net.block4[0].attn.num_heads) * len(net.block4)
    print(stage1 + stage2 + stage3 + stage4)
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)


def get_vit_flops(net, input_shape, patch_size):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    stage = mha_flops(H // patch_size, W // patch_size,
                      net.blocks[0].attn.dim,
                      net.blocks[0].attn.num_heads) * len(net.blocks)
    flops += stage
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


def get_mypvt20_2_flops(net, input_shape, flag_g=False):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape

    # for blocks
    # stage1 is unchanged
    stage1 = sra_flops(H // 4, W // 4,
                       net.block1[0].attn.sr_ratio,
                       net.block1[0].attn.dim,
                       net.block1[0].attn.num_heads) * len(net.block1)

    H, W = H // 4, W // 4
    N = H * W
    dim = net.block1[0].attn.dim
    N_grid = (H // net.grid_stride) * (W // net.grid_stride)

    # stage2
    dim_in = dim
    Ns = N
    N = max(math.ceil(Ns * net.down_layers1.sample_ratio), 0) + N_grid
    # N = max(math.ceil((Ns-N_grid) * net.down_layers3.sample_ratio), 0) + N_grid

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
    N = max(math.ceil(Ns * net.down_layers2.sample_ratio), 0) + N_grid
    # N = max(math.ceil((Ns-N_grid) * net.down_layers3.sample_ratio), 0) + N_grid

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
    N = max(math.ceil(Ns * net.down_layers3.sample_ratio), 0) + N_grid
    # N = max(math.ceil((Ns-N_grid) * net.down_layers3.sample_ratio), 0) + N_grid


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
    flops += stage1 + stage2 + stage3 + stage4

    return flops_to_string(flops), params_to_string(params)


# with torch.cuda.device(0):
#     # net = models.vit_small_patch16_224()
#     # input_shape = (3, 224, 224)
#     # flops, params = get_vit_flops(net, input_shape, 16)
#     # print(flops)
#     net = pvt.pvt_small()
#     input_shape = (3, 224, 224)
#     flops, params = get_pvt_flops(net, input_shape)
#     split_line = '=' * 30
#     print(f'{split_line}\nInput shape: {input_shape}\n'
#           f'Flops: {flops}\nParams: {params}\n{split_line}')
#


# with torch.cuda.device(0):
#     # net = models.vit_small_patch16_224()
#     # input_shape = (3, 224, 224)
#     # flops, params = get_vit_flops(net, input_shape, 16)
#     # print(flops)
#     net = pvt_impr8.pvt_small_impr8_peg()
#     input_shape = (3, 224, 224)
#     flops, params = get_pvt_flops(net, input_shape)
#     split_line = '=' * 30
#     print(f'{split_line}\nInput shape: {input_shape}\n'
#           f'Flops: {flops}\nParams: {params}\n{split_line}')


with torch.cuda.device(0):
    # net = models.vit_small_patch16_224()
    # input_shape = (3, 224, 224)
    # flops, params = get_vit_flops(net, input_shape, 16)
    # print(flops)
    net = my_pvt20_2.mypvt20_2_small()
    input_shape = (3, 224, 224)
    flops, params = get_mypvt20_2_flops(net, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')


# with torch.cuda.device(0):
#     # net = models.vit_small_patch16_224()
#     # input_shape = (3, 224, 224)
#     # flops, params = get_vit_flops(net, input_shape, 16)
#     # print(flops)
#     net = my_pvt20_2g.mypvt20_2g_small()
#     input_shape = (3, 224, 224)
#     flops, params = get_mypvt20_2_flops(net, input_shape, flag_g=True)
#     split_line = '=' * 30
#     print(f'{split_line}\nInput shape: {input_shape}\n'
#           f'Flops: {flops}\nParams: {params}\n{split_line}')
