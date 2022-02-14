import math
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer, build_conv_layer
from mmcv.runner import BaseModule
from .transformer_utils import trunc_normal_, DropPath
from .tc_layers import TCMlp
from .tcformer_utils import token2map, map2token, token_downup, gaussian_filt
from .tcformer_utils import token2map_flops, map2token_flops, downup_flops
from mmdet.models.builder import NECKS
from mmdet.utils import get_root_logger
import warnings
import torch.nn.functional as F
import copy
import torch

# MTA head
@NECKS.register_module()
# neck like the fuse layer in HRNet
class HRNeck(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            return_map=True,
            norm_cfg=dict(type='BN'),
            conv_cfg=None,
            act_cfg=None,
            add_extra_convs=False,
            num_outs=1,
    ):
        super().__init__()

        # only support return map now
        assert return_map
        self.return_map = return_map
        self.norm_cfg = norm_cfg
        self.num_branches = len(in_channels)
        self.num_out_branches = len(out_channels)
        self.conv_cfg = conv_cfg
        assert self.num_out_branches == num_outs

        fuse_layers = []
        for i in range(min(self.num_out_branches, self.num_branches)):
            fuse_layer = []
            for j in range(self.num_branches):
                fuse_layer.append(
                    nn.Sequential(
                        build_conv_layer(
                            self.conv_cfg,
                            in_channels[j],
                            out_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
                        build_norm_layer(self.norm_cfg, out_channels[i])[1],
                    )
                )

            fuse_layer = nn.ModuleList(fuse_layer)
            fuse_layers.append(fuse_layer)
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)

        # extra layer
        self.add_extra_convs = add_extra_convs
        if self.add_extra_convs:
            # add extra conv layers (e.g., RetinaNet)
            self.extra_convs = nn.ModuleList()
            extra_levels = self.num_out_branches - self.num_branches
            if self.add_extra_convs and extra_levels >= 1:
                for i in range(extra_levels):
                    extra_conv = ConvModule(
                        out_channels[self.num_branches+i-1],
                        out_channels[self.num_branches+i],
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                    self.extra_convs.append(extra_conv)

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

    def forward(self, input_lists):
        assert len(input_lists) == self.num_branches
        out_lists = []
        for i in range(min(self.num_out_branches, self.num_branches)):
            tar_dict = input_lists[i]
            map_size = tar_dict['map_size']

            out_map = 0
            for j in range(self.num_branches):
                src_dict = input_lists[j]
                x_t = token2map(src_dict['x'], None, src_dict['loc_orig'], src_dict['idx_agg'], map_size)[0]
                x_t = self.fuse_layers[i][j](x_t)
                out_map = out_map + x_t
            out_lists.append(out_map)

        # part 2: add extra levels
        used_backbone_levels = len(out_lists)
        if self.num_out_branches > len(out_lists):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_out_branches - used_backbone_levels):
                    out_lists.append(F.max_pool2d(out_lists[-1], 1, stride=2))

            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_output':
                    extra_source = out_lists[-1]
                else:
                    raise NotImplementedError

                out_lists.append(self.extra_convs[0](extra_source))
                for i in range(1, self.num_out_branches - used_backbone_levels):
                    out_lists.append(self.extra_convs[i](out_lists[-1]))

        return out_lists



# MTA head
@NECKS.register_module()
# neck like the fuse layer in HRNet
class HRNeck2(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            return_map=True,
            norm_cfg=dict(type='BN'),
            conv_cfg=None,
            act_cfg=None,
            add_extra_convs=False,
            num_outs=1,
    ):
        super().__init__()

        # only support return map now
        assert return_map
        self.return_map = return_map
        self.norm_cfg = norm_cfg
        self.num_branches = len(in_channels)
        self.num_out_branches = len(out_channels)
        self.conv_cfg = conv_cfg
        assert self.num_out_branches == num_outs

        fuse_layers = []
        for i in range(min(self.num_out_branches, self.num_branches)):
            fuse_layer = []
            for j in range(self.num_branches):
                fuse_layer.append(
                    nn.Sequential(
                        build_conv_layer(
                            self.conv_cfg,
                            in_channels[j],
                            out_channels[i],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
                        # build_norm_layer(self.norm_cfg, out_channels[i])[1],
                    )
                )

            fuse_layer = nn.ModuleList(fuse_layer)
            fuse_layers.append(fuse_layer)
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)

        # extra layer
        self.add_extra_convs = add_extra_convs
        if self.add_extra_convs:
            # add extra conv layers (e.g., RetinaNet)
            self.extra_convs = nn.ModuleList()
            extra_levels = self.num_out_branches - self.num_branches
            if self.add_extra_convs and extra_levels >= 1:
                for i in range(extra_levels):
                    extra_conv = ConvModule(
                        out_channels[self.num_branches+i-1],
                        out_channels[self.num_branches+i],
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        # norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                    self.extra_convs.append(extra_conv)

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

    def forward(self, input_lists):
        assert len(input_lists) == self.num_branches
        out_lists = []
        for i in range(min(self.num_out_branches, self.num_branches)):
            tar_dict = input_lists[i]
            map_size = tar_dict['map_size']

            out_map = 0
            for j in range(self.num_branches):
                src_dict = input_lists[j]
                x_t = token2map(src_dict['x'], None, src_dict['loc_orig'], src_dict['idx_agg'], map_size)[0]
                x_t = self.fuse_layers[i][j](x_t)
                out_map = out_map + x_t
            out_lists.append(out_map)

        # part 2: add extra levels
        used_backbone_levels = len(out_lists)
        if self.num_out_branches > len(out_lists):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_out_branches - used_backbone_levels):
                    out_lists.append(F.max_pool2d(out_lists[-1], 1, stride=2))

            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_output':
                    extra_source = out_lists[-1]
                else:
                    raise NotImplementedError

                out_lists.append(self.extra_convs[0](extra_source))
                for i in range(1, self.num_out_branches - used_backbone_levels):
                    out_lists.append(self.extra_convs[i](out_lists[-1]))

        return out_lists
