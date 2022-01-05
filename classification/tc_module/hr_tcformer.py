# Copyright (c) OpenMMLab. All rights reserved.

import copy

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
# from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import BaseModule
from timm.models.layers import to_2tuple, trunc_normal_
from torch.nn import functional as F



from .tc_layers import TCWinBlock
from .tcformer_utils import (
    map2token, token2map, token_downup, get_grid_loc,
    token_cluster_part_pad, token_cluster_part_follow,
    show_tokens_merge, token_cluster_grid)
import math
from timm.models.registry import register_model

vis = False


# from ...builder import BACKBONES
# from ..hrnet import BasicBlock, Bottleneck, HRModule, HRNet


class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class HRModule(nn.Module):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=False,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 upsample_cfg=dict(mode='nearest', align_corners=None)):

        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _check_branches(num_branches, num_blocks, in_channels, num_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """Make one branch."""
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * get_expansion(block):
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(
                    self.norm_cfg,
                    num_channels[branch_index] * get_expansion(block))[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index] * get_expansion(block),
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * get_expansion(block)
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1

        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i),
                                mode=self.upsample_cfg['mode'],
                                align_corners=self.
                                upsample_cfg['align_corners'])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Module):
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
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
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        self.upsample_cfg = self.extra.get('upsample', {
            'mode': 'nearest',
            'align_corners': None
        })

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * get_expansion(block)
        self.layer1 = self._make_layer(block, 64, stage1_out_channels,
                                       num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=self.stage4_cfg.get('multiscale_output', False))

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Make layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, out_channels)[1])

        layers = []
        layers.append(
            block(
                in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    upsample_cfg=self.upsample_cfg))

            in_channels = hr_modules[-1].in_channels

        return nn.Sequential(*hr_modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()




# part wise merge with padding with dict as input and output
# no block in this layer, use BN layer.
class CTM_partpad_dict(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate,
                 k=5, nh=1, nw=None, nh_list=None, nw_list=None,
                 use_agg_weight=True, agg_weight_detach=False, with_act=True,
                 ):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out

        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k

        # for partwise
        self.nh = nh
        self.nw = nw or nh
        self.nh_list = nh_list
        self.nw_list = nw_list or nh_list
        self.use_agg_weight = use_agg_weight
        self.agg_weight_detach = agg_weight_detach
        self.with_act = with_act
        if self.with_act:
            self.act = nn.ReLU(inplace=False)

    def forward(self, input_dict):
        input_dict = input_dict.copy()
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        if not self.use_agg_weight:
            agg_weight = None

        if agg_weight is not None and self.agg_weight_detach:
            agg_weight = agg_weight.detach()

        B, N, C = x.shape
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()
        input_dict['x'] = x

        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)
        nh, nw = self.nh, self.nw
        num_part = nh * nw
        sample_num = round(sample_num // num_part) * num_part

        # print('ONLY FOR DEBUG')
        # Ns = x_map.shape[1] * x_map.shape[2]
        # x_down, idx_agg_down, weight_t, _ = token_cluster_grid(input_dict, Ns, conf, weight=None, k=5)

        if self.nh_list is not None and self.nw_list is not None:
            x_down, idx_agg_down, weight_t = token_cluster_part_pad(
                input_dict, sample_num, weight=weight, k=self.k,
                nh_list=self.nh_list, nw_list=self.nw_list
            )
        else:
            x_down, idx_agg_down, weight_t = token_cluster_part_follow(
                input_dict, sample_num, weight=weight, k=self.k, nh=nh, nw=nw
            )

        if agg_weight is not None:
            agg_weight_down = agg_weight * weight_t
            agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]
            if self.agg_weight_detach:
                agg_weight_down = agg_weight_down.detach()
        else:
            agg_weight_down = None

        _, _, H, W = x_map.shape
        input_dict['conf'] = conf
        input_dict['map_size'] = [H, W]

        out_dict = {
            'x': x_down,
            'idx_agg': idx_agg_down,
            'agg_weight': agg_weight_down,
            'loc_orig': loc_orig,
            'map_size': [H, W]
        }

        if self.with_act:
            out_dict['x'] = self.act(out_dict['x'])
            input_dict['x'] = self.act(input_dict['x'])

        return out_dict, input_dict


# part wise merge with padding with dict as input and output
# no block in this layer, use BN layer.
class CTM_partpad_dict_BN(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate,
                 k=5, nh=1, nw=None, nh_list=None, nw_list=None,
                 use_agg_weight=True, agg_weight_detach=False, with_act=True,
                 norm_cfg=None,
                 ):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.norm_cfg = norm_cfg

        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        self.norm_name, self.norm = build_norm_layer(self.norm_cfg, self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k

        # for partwise
        self.nh = nh
        self.nw = nw or nh
        self.nh_list = nh_list
        self.nw_list = nw_list or nh_list
        self.use_agg_weight = use_agg_weight
        self.agg_weight_detach = agg_weight_detach
        self.with_act = with_act
        if self.with_act:
            self.act = nn.ReLU(inplace=False)

    def forward(self, input_dict):
        input_dict = input_dict.copy()
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        if not self.use_agg_weight:
            agg_weight = None

        if agg_weight is not None and self.agg_weight_detach:
            agg_weight = agg_weight.detach()

        B, N, C = x.shape
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = token_norm(self.norm, self.norm_name, x)

        conf = self.conf(x)
        weight = conf.exp()
        input_dict['x'] = x

        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)
        nh, nw = self.nh, self.nw
        num_part = nh * nw
        sample_num = round(sample_num // num_part) * num_part

        # print('ONLY FOR DEBUG')
        # Ns = x_map.shape[1] * x_map.shape[2]
        # x_down, idx_agg_down, weight_t, _ = token_cluster_grid(input_dict, Ns, conf, weight=None, k=5)

        if self.nh_list is not None and self.nw_list is not None:
            x_down, idx_agg_down, weight_t = token_cluster_part_pad(
                input_dict, sample_num, weight=weight, k=self.k,
                nh_list=self.nh_list, nw_list=self.nw_list
            )
        else:
            x_down, idx_agg_down, weight_t = token_cluster_part_follow(
                input_dict, sample_num, weight=weight, k=self.k, nh=nh, nw=nw
            )

        if agg_weight is not None:
            agg_weight_down = agg_weight * weight_t
            agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]
            if self.agg_weight_detach:
                agg_weight_down = agg_weight_down.detach()
        else:
            agg_weight_down = None

        _, _, H, W = x_map.shape
        input_dict['conf'] = conf
        input_dict['map_size'] = [H, W]

        out_dict = {
            'x': x_down,
            'idx_agg': idx_agg_down,
            'agg_weight': agg_weight_down,
            'loc_orig': loc_orig,
            'map_size': [H, W]
        }

        # print('ONLY FOR DEBUG.')
        # xt = x_down.permute(0, 2, 1).reshape(x_map.shape)
        # xt = token2map(x, None, loc_orig, idx_agg, [H, W])[0]

        if self.with_act:
            out_dict['x'] = self.act(out_dict['x'])
            input_dict['x'] = self.act(input_dict['x'])

        return out_dict, input_dict


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

        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])
        x_map = super().forward(x_map)
        x = map2token(x_map, x.shape[1], loc_orig, idx_agg, agg_weight) + self.skip(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


def token_norm(norm_layer, norm_name, x):
    if 'ln' in norm_name:
        x = norm_layer(x)
    else:
        x = norm_layer(x.permute(0, 2, 1).unsqueeze(-1)).flatten(2).permute(0, 2, 1)
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


class TokenDownLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg,
                 norm_cfg,
                 with_act=True,
                 ):
        super().__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.dw_conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False)

        self.dw_skip = build_conv_layer(
            self.conv_cfg,
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            groups=in_channels,
            bias=False)

        self.norm1 = build_norm_layer(self.norm_cfg, in_channels)[1]
        self.conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self.norm2 = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.with_act = with_act
        if self.with_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input_dict, tar_dict):
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        Nt = tar_dict['x'].shape[1]
        idx_agg_t = tar_dict['idx_agg']
        agg_weight_t = tar_dict['agg_weight']

        # real 2D
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])
        x_map = self.dw_conv(x_map)
        x_map = map2token(x_map, Nt, loc_orig, idx_agg_t, agg_weight_t)

        # fake 2D
        x = token_downup(source_dict=input_dict, target_dict=tar_dict)
        x = x.permute(0, 2, 1)[..., None]
        x = self.dw_skip(x)
        x = x + x_map.permute(0, 2, 1)[..., None]

        x = self.norm1(x)
        x = self.conv(x)
        x = self.norm2(x)
        x = x.squeeze(-1).permute(0, 2, 1)

        if self.with_act:
            x = self.act(x)

        out_dict = tar_dict.copy()
        out_dict['x'] = x
        return out_dict


# one step for multi-level sampling
class TokenFuseLayer(nn.Module):
    def __init__(
            self,
            num_branches,
            in_channels,
            multiscale_output,
            conv_cfg,
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
                if j > i:
                    # upsample
                    fuse_link = DictLayer(
                        nn.Sequential(
                            nn.Linear(in_channels[j], in_channels[i], bias=False),
                            TokenNorm(build_norm_layer(self.norm_cfg, in_channels[i]))),
                        input_decap=True,
                        output_cap=True,
                    )
                elif j == i:
                    # same stage
                    fuse_link = None
                else:
                    # down sample
                    fuse_link = []
                    for k in range(i - j):
                        fuse_link.append(
                            TokenDownLayer(
                                in_channels=in_channels[j],
                                out_channels=in_channels[i] if k == i - j - 1 else in_channels[j],
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                with_act=(k != i - j - 1)),
                        )
                    fuse_link = nn.ModuleList(fuse_link)

                fuse_layer.append(fuse_link)
            fuse_layer = nn.ModuleList(fuse_layer)
            fuse_layers.append(fuse_layer)
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_lists):
        assert len(input_lists) == self.num_branches
        out_lists = []

        # target loop
        for i in range(self.num_out_branches):
            tar_dict = input_lists[i]
            x = tar_dict['x']
            idx_agg = tar_dict['idx_agg']
            agg_weight = tar_dict['agg_weight']

            out_dict = {
                'x': x,
                'idx_agg': idx_agg,
                'agg_weight': agg_weight,
                'map_size': tar_dict['map_size'],
                'loc_orig': tar_dict['loc_orig']
            }

            # source loop
            for j in range(self.num_branches):
                if j > i:
                    # upsample
                    src_dict = input_lists[j].copy()
                    src_dict = self.fuse_layers[i][j](src_dict)
                    x_tmp = token_downup(target_dict=out_dict, source_dict=src_dict)
                    out_dict['x'] = out_dict['x'] + x_tmp

                elif j == i:
                    pass

                else:
                    # down sample
                    src_dict = input_lists[j].copy()
                    fuse_link = self.fuse_layers[i][j]
                    for k in range(i - j):
                        tar_dict = input_lists[k + j + 1]
                        src_dict = fuse_link[k](src_dict, tar_dict)
                    out_dict['x'] = out_dict['x'] + src_dict['x']

            out_dict['x'] = self.relu(out_dict['x'])
            out_lists.append(out_dict)
        return out_lists


class HRTCModule(HRModule):

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 num_heads=None,
                 num_window_sizes=None,
                 num_mlp_ratios=None,
                 drop_paths=0.0,
                 upsample_cfg=dict(mode='bilinear', align_corners=False)):

        super(HRModule, self).__init__()
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.upsample_cfg = upsample_cfg
        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.drop_paths = drop_paths

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
        stride=1,
    ):
        """Make one branch."""
        # LocalWindowTransformerBlock does not support down sample layer yet.
        assert stride == 1 and self.in_channels[branch_index] == num_channels[
            branch_index]
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                    mlp_norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                ))
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
                ))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None
        return TokenFuseLayer(
            self.num_branches,
            self.in_channels,
            self.multiscale_output,
            conv_cfg=self.conv_cfg,
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


class HRTCFormer(HRNet):
    """HRFormer backbone.

    High Resolution Transformer Backbone
    """

    blocks_dict = {
        'BASIC': BasicBlock,
        'BOTTLENECK': Bottleneck,
        'TCWINBLOCK': TCWinBlock
    }

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,
                 return_map=False):
        super(HRNet, self).__init__()

        # generate drop path rate list
        depth_s2 = (
            extra['stage2']['num_blocks'][0] * extra['stage2']['num_modules'])
        depth_s3 = (
            extra['stage3']['num_blocks'][0] * extra['stage3']['num_modules'])
        depth_s4 = (
            extra['stage4']['num_blocks'][0] * extra['stage4']['num_modules'])
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = extra['drop_path_rate']
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        extra['stage2']['drop_path_rates'] = dpr[0:depth_s2]
        extra['stage3']['drop_path_rates'] = dpr[depth_s2:depth_s2 + depth_s3]
        extra['stage4']['drop_path_rates'] = dpr[depth_s2 + depth_s3:]

        upsample_cfg = extra.get('upsample', {
            'mode': 'bilinear',
            'align_corners': False
        })
        extra['upsample'] = upsample_cfg

        self.return_map = return_map

        # for partwise clustering
        self.nh_list = extra.get('nh_list', [1, 1, 1])
        self.nw_list = extra.get('nw_list', [1, 1, 1])

        self.ctm_with_act = extra.get('ctm_with_act', True)
        if vis:
            self.count = 0
        super().__init__(extra, in_channels, conv_cfg, norm_cfg, norm_eval,
                         with_cp, zero_init_residual)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        num_heads = layer_config['num_heads']
        num_window_sizes = layer_config['num_window_sizes']
        num_mlp_ratios = layer_config['num_mlp_ratios']
        drop_path_rates = layer_config['drop_path_rates']

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRTCModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    upsample_cfg=self.upsample_cfg,
                    num_heads=num_heads,
                    num_window_sizes=num_window_sizes,
                    num_mlp_ratios=num_mlp_ratios,
                    drop_paths=drop_path_rates[num_blocks[0] *
                                               i:num_blocks[0] * (i + 1)],
                ))

        return nn.Sequential(*hr_modules), in_channels

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        pre_stage = len(num_channels_pre_layer) - 1

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
                down_layers = CTM_partpad_dict_BN(
                    embed_dim=num_channels_pre_layer[-1],
                    dim_out=num_channels_cur_layer[i],
                    drop_rate=0,
                    sample_ratio=0.25,
                    nh_list=self.nh_list if pre_stage == 0 else None,
                    nw_list=self.nw_list if pre_stage == 0 else None,
                    nh=self.nh_list[pre_stage],
                    nw=self.nw_list[pre_stage],
                    with_act=self.ctm_with_act,
                    norm_cfg=self.norm_cfg
                )
                transition_layers.append(down_layers)

        return nn.ModuleList(transition_layers)

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

    def tran2map(self, input_list):
        for i in range(len(input_list)):
            input_dict = input_list[i]
            x = input_dict['x']
            H, W = input_dict['map_size']
            idx_agg = input_dict['idx_agg']
            loc_orig = input_dict['loc_orig']
            x, _ = token2map(x, None, loc_orig, idx_agg, [H, W])
            input_list[i] = x
        return input_list

    def forward(self, x):
        """Forward function."""
        if vis:
            img = x.clone()

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

        if vis:
            show_tokens_merge(img, x_list, self.count)
            self.count += 1

        return y_list


class HirAttNeck(nn.Module):

    def __init__(self,
                 in_channels=[32, 64, 128, 256],
                 query_channels=256,
                 out_channels=512,
                 qkv_bias=True,
                 num_heads=[8, 8, 8, 8],
                 drop_rate=0.5
                 ):
        super().__init__()
        self.qs = nn.ModuleList()
        self.kvs = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.num_heads = num_heads
        self.query_channels = query_channels
        self.out_channels = out_channels
        # self.mlps = []

        # self.pre_query = nn.Sequential(
        #         nn.Linear(query_channels, out_channels),
        #         nn.GELU(),
        #         nn.Linear(out_channels, out_channels),
        #         nn.Dropout(drop_rate)
        #     )

        self.pre_query = nn.Linear(query_channels, out_channels)

        for i, in_channel in enumerate(in_channels):
            self.qs.append(nn.Linear(out_channels, out_channels, bias=qkv_bias))
            self.kvs.append(nn.Linear(in_channel, out_channels * 2, bias=qkv_bias))
            self.projs.append(nn.Linear(out_channels, out_channels))

            # self.mlps.append(
            #     nn.Sequential(
            #         nn.Linear(out_channels, out_channels),
            #         nn.GELU(),
            #         nn.Linear(out_channels, out_channels),
            #         nn.Dropout(drop_rate)
            #     )
            # )

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

    def forward(self, x_list):
        if isinstance(x_list[0], dict):
            # only need tokens
            x_list = [tmp['x'] for tmp in x_list]

        query = x_list[-1].mean(dim=1, keepdim=True)
        query = self.pre_query(query)
        out = query

        for i in range(len(self.num_heads)-1, -1, -1):
            x = x_list[i]
            B, N, _ = x.shape

            C = self.out_channels
            num_head = self.num_heads[i]
            head_dim = C // num_head
            scale = head_dim ** -0.5

            q = self.qs[i](out).reshape(B, 1, num_head, C // num_head).permute(0, 2, 1, 3)
            kv = self.kvs[i](x).reshape(B, -1, 2, num_head, C // num_head).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q * scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            out_attn = (attn @ v).transpose(1, 2).reshape(B, 1, C)
            out_attn = self.projs[i](out_attn)
            out = out + out_attn

            # out = out + self.mlps[i](out)

        return out.squeeze(1)



@register_model
class hrtcformer_w32(HRTCFormer):
    def __init__(self, num_classes=1000, **kwargs):
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        # debug only
        print('FOR DEBUG ONLY!')
        norm_cfg = dict(type='BN', requires_grad=True)
        super().__init__(
            in_channels=3,
            norm_cfg=norm_cfg,
            return_map=False,
            extra=dict(
                nh_list=[4, 2, 1],
                nw_list=[4, 2, 1],
                drop_path_rate=0.0,
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
                    block='TCWINBLOCK',
                    num_blocks=(2, 2),
                    num_channels=(32, 64),
                    num_heads=[1, 2],
                    num_mlp_ratios=[4, 4],
                    num_window_sizes=[7, 7]),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='TCWINBLOCK',
                    num_blocks=(2, 2, 2),
                    num_channels=(32, 64, 128),
                    num_heads=[1, 2, 4],
                    num_mlp_ratios=[4, 4, 4],
                    num_window_sizes=[7, 7, 7]),
                stage4=dict(
                    num_modules=2,
                    num_branches=4,
                    block='TCWINBLOCK',
                    num_blocks=(2, 2, 2, 2),
                    num_channels=(32, 64, 128, 256),
                    num_heads=[1, 2, 4, 8],
                    num_mlp_ratios=[4, 4, 4, 4],
                    num_window_sizes=[7, 7, 7, 7],
                    multiscale_output=True)
            )
        )

        self.neck = HirAttNeck(
            in_channels=[32, 64, 128, 256],
            query_channels=256,
            out_channels=512,
            qkv_bias=True,
            num_heads=[8, 8, 8, 8],
            drop_rate=0.5
        )

        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = super().forward(x)
        x = self.neck(x)
        # x = x[1]['x'].mean(dim=1)
        x = self.head(x)
        return x



