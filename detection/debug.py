# import argparse
# import copy
# import os
# import os.path as osp
# import time
# import warnings
#
# import mmcv
# import torch
# from mmcv import Config, DictAction
# from mmcv.runner import get_dist_info, init_dist
# from mmcv.utils import get_git_hash
#
# from mmdet import __version__
# # from mmdet.apis import set_random_seed, train_detector
# # from mmdet.datasets import build_dataset
# from mmdet.models import build_detector
# from mmdet.utils import collect_env, get_root_logger
# import pvt
# import my_pvt
# import my_fpn
# import my_pvt20_2
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a detector')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument('--work-dir', help='the dir to save logs and models')
#     parser.add_argument(
#         '--resume-from', help='the checkpoint file to resume from')
#     parser.add_argument(
#         '--no-validate',
#         action='store_true',
#         help='whether not to evaluate the checkpoint during training')
#     group_gpus = parser.add_mutually_exclusive_group()
#     group_gpus.add_argument(
#         '--gpus',
#         type=int,
#         help='number of gpus to use '
#              '(only applicable to non-distributed training)')
#     group_gpus.add_argument(
#         '--gpu-ids',
#         type=int,
#         nargs='+',
#         help='ids of gpus to use '
#              '(only applicable to non-distributed training)')
#     parser.add_argument('--seed', type=int, default=None, help='random seed')
#     parser.add_argument(
#         '--deterministic',
#         action='store_true',
#         help='whether to set deterministic options for CUDNN backend.')
#     parser.add_argument(
#         '--options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#              'in xxx=yyy format will be merged into config file (deprecate), '
#              'change to --cfg-options instead.')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#              'in xxx=yyy format will be merged into config file. If the value to '
#              'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#              'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#              'Note that the quotation marks are necessary and that no white space '
#              'is allowed.')
#     parser.add_argument(
#         '--launcher',
#         choices=['none', 'pytorch', 'slurm', 'mpi'],
#         default='none',
#         help='job launcher')
#     parser.add_argument('--local_rank', type=int, default=0)
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)
#
#     if args.options and args.cfg_options:
#         raise ValueError(
#             '--options and --cfg-options cannot be both '
#             'specified, --options is deprecated in favor of --cfg-options')
#     if args.options:
#         warnings.warn('--options is deprecated in favor of --cfg-options')
#         args.cfg_options = args.options
#
#     return args
#
#
# def main():
#     args = parse_args()
#
#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
#     # import modules from string list.
#     if cfg.get('custom_imports', None):
#         from mmcv.utils import import_modules_from_strings
#         import_modules_from_strings(**cfg['custom_imports'])
#     # set cudnn_benchmark
#     if cfg.get('cudnn_benchmark', False):
#         torch.backends.cudnn.benchmark = True
#
#     # work_dir is determined in this priority: CLI > segment in file > filename
#     if args.work_dir is not None:
#         # update configs according to CLI args if args.work_dir is not None
#         cfg.work_dir = args.work_dir
#     elif cfg.get('work_dir', None) is None:
#         # use config filename as default work_dir if cfg.work_dir is None
#         cfg.work_dir = osp.join('./work_dirs',
#                                 osp.splitext(osp.basename(args.config))[0])
#     if args.resume_from is not None:
#         cfg.resume_from = args.resume_from
#     if args.gpu_ids is not None:
#         cfg.gpu_ids = args.gpu_ids
#     else:
#         cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
#
#     # init distributed env first, since logger depends on the dist info.
#     if args.launcher == 'none':
#         distributed = False
#     else:
#         distributed = True
#         init_dist(args.launcher, **cfg.dist_params)
#         # re-set gpu_ids with distributed training mode
#         _, world_size = get_dist_info()
#         cfg.gpu_ids = range(world_size)
#
#     # # create work_dir
#     # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
#     # # dump config
#     # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
#     # # init the logger before other steps
#     # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#     # log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
#     # logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
#     #
#     # # init the meta dict to record some important information such as
#     # # environment info and seed, which will be logged
#     # meta = dict()
#     # # log env info
#     # env_info_dict = collect_env()
#     # env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
#     # dash_line = '-' * 60 + '\n'
#     # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
#     #             dash_line)
#     # meta['env_info'] = env_info
#     # meta['config'] = cfg.pretty_text
#     # # log some basic info
#     # logger.info(f'Distributed training: {distributed}')
#     # logger.info(f'Config:\n{cfg.pretty_text}')
#     #
#     # # set random seeds
#     # if args.seed is not None:
#     #     logger.info(f'Set random seed to {args.seed}, '
#     #                 f'deterministic: {args.deterministic}')
#     #     set_random_seed(args.seed, deterministic=args.deterministic)
#     # cfg.seed = args.seed
#     # meta['seed'] = args.seed
#     # meta['exp_name'] = osp.basename(args.config)
#
#     model = build_detector(
#         cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
#
#     empty_input = torch.rand([2, 3, 256, 256])
#     features = model.extract_feat(empty_input)
#
#     datasets = [build_dataset(cfg.data.train)]
#     if len(cfg.workflow) == 2:
#         val_dataset = copy.deepcopy(cfg.data.val)
#         val_dataset.pipeline = cfg.data.train.pipeline
#         datasets.append(build_dataset(val_dataset))
#     if cfg.checkpoint_config is not None:
#         # save mmdet version, config file content and class names in
#         # checkpoints as meta data
#         cfg.checkpoint_config.meta = dict(
#             mmdet_version=__version__ + get_git_hash()[:7],
#             CLASSES=datasets[0].CLASSES)
#     # add an attribute for visualization convenience
#     model.CLASSES = datasets[0].CLASSES
#     train_detector(
#         model,
#         datasets,
#         cfg,
#         distributed=distributed,
#         validate=(not args.no_validate),
#         timestamp=timestamp,
#         meta=meta)
#
#
# if __name__ == '__main__':
#     main()

import torch
import warnings
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.necks import FPN

# class TokenFPN(nn.Module):
#     r"""Feature Pyramid Network.
#
#     This is an implementation of paper `Feature Pyramid Networks for Object
#     Detection <https://arxiv.org/abs/1612.03144>`_.
#
#     Args:
#         in_channels (List[int]): Number of input channels per scale.
#         out_channels (int): Number of output channels (used at each scale)
#         num_outs (int): Number of output scales.
#         start_level (int): Index of the start input backbone level used to
#             build the feature pyramid. Default: 0.
#         end_level (int): Index of the end input backbone level (exclusive) to
#             build the feature pyramid. Default: -1, which means the last level.
#         add_extra_convs (bool | str): If bool, it decides whether to add conv
#             layers on top of the original feature maps. Default to False.
#             If True, its actual mode is specified by `extra_convs_on_inputs`.
#             If str, it specifies the source feature map of the extra convs.
#             Only the following options are allowed
#
#             - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
#             - 'on_lateral':  Last feature map after lateral convs.
#             - 'on_output': The last output feature map after fpn convs.
#         extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
#             on the original feature from the backbone. If True,
#             it is equivalent to `add_extra_convs='on_input'`. If False, it is
#             equivalent to set `add_extra_convs='on_output'`. Default to True.
#         relu_before_extra_convs (bool): Whether to apply relu before the extra
#             conv. Default: False.
#         no_norm_on_lateral (bool): Whether to apply norm on lateral.
#             Default: False.
#         caffe2_xavier_init (bool): Whether to apply caffe2_xavier_init on all
#             conv in FPN. Default: False.
#         conv_cfg (dict): Config dict for convolution layer. Default: None.
#         norm_cfg (dict): Config dict for normalization layer. Default: None.
#         act_cfg (str): Config dict for activation layer in ConvModule.
#             Default: None.
#         upsample_cfg (dict): Config dict for interpolate layer.
#             Default: `dict(mode='nearest')`
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#
#     Example:
#         >>> import torch
#         >>> in_channels = [2, 3, 5, 7]
#         >>> scales = [340, 170, 84, 43]
#         >>> inputs = [torch.rand(1, c, s, s)
#         ...           for c, s in zip(in_channels, scales)]
#         >>> self = FPN(in_channels, 11, len(in_channels)).eval()
#         >>> outputs = self.forward(inputs)
#         >>> for i in range(len(outputs)):
#         ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
#         outputs[0].shape = torch.Size([1, 11, 340, 340])
#         outputs[1].shape = torch.Size([1, 11, 170, 170])
#         outputs[2].shape = torch.Size([1, 11, 84, 84])
#         outputs[3].shape = torch.Size([1, 11, 43, 43])
#     """
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_outs,
#                  kernel_size,
#                  sigma,
#                  num_heads=[],
#                  start_level=0,
#                  end_level=-1,
#                  add_extra_convs=False,
#                  extra_convs_on_inputs=True,
#                  relu_before_extra_convs=False,
#                  no_norm_on_lateral=False,
#                  caffe2_xavier_init=False,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  act_cfg=None,
#                  upsample_cfg=dict(mode='nearest'),
#                  init_cfg=dict(
#                      type='Xavier', layer='Conv2d', distribution='uniform')):
#
#         # super().__init__(init_cfg)
#         # super(TokenFPN, self).__init__(init_cfg)
#         super().__init__()
#
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
#         self.relu_before_extra_convs = relu_before_extra_convs
#         self.no_norm_on_lateral = no_norm_on_lateral
#         self.fp16_enabled = False
#         self.upsample_cfg = upsample_cfg.copy()
#         self.caffe2_xavier_init = caffe2_xavier_init
#
#         self.num_heads = num_heads
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#
#         if end_level == -1:
#             self.backbone_end_level = self.num_ins
#             assert num_outs >= self.num_ins - start_level
#         else:
#             # if end_level < inputs, no extra level is allowed
#             self.backbone_end_level = end_level
#             assert end_level <= len(in_channels)
#             assert num_outs == end_level - start_level
#         self.start_level = start_level
#         self.end_level = end_level
#         self.add_extra_convs = add_extra_convs
#         assert isinstance(add_extra_convs, (str, bool))
#         if isinstance(add_extra_convs, str):
#             # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
#             assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
#         elif add_extra_convs:  # True
#             if extra_convs_on_inputs:
#                 # TODO: deprecate `extra_convs_on_inputs`
#                 warnings.simplefilter('once')
#                 warnings.warn(
#                     '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
#                     'Please use "add_extra_convs"', DeprecationWarning)
#                 self.add_extra_convs = 'on_input'
#             else:
#                 self.add_extra_convs = 'on_output'
#
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()
#
#         for i in range(self.start_level, self.backbone_end_level):
#             # l_conv = ConvModule(
#             #     in_channels[i],
#             #     out_channels,
#             #     1,
#             #     conv_cfg=conv_cfg,
#             #     norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#             #     act_cfg=act_cfg,
#             #     inplace=False)
#
#             # l_conv = MyBlock2(
#             #     in_channels[i],
#             #     num_heads=num_heads[i],
#             #     dim_out=out_channels,
#             #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             #
#             # )
#
#             # l_conv = nn.Linear(
#             #     in_channels[i],
#             #     out_channels,)
#
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                 act_cfg=act_cfg,
#                 inplace=False)
#
#             fpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False)
#
#             self.lateral_convs.append(l_conv)
#             self.fpn_convs.append(fpn_conv)
#
#         # add extra conv layers (e.g., RetinaNet)
#         extra_levels = num_outs - self.backbone_end_level + self.start_level
#         if self.add_extra_convs and extra_levels >= 1:
#             for i in range(extra_levels):
#                 if i == 0 and self.add_extra_convs == 'on_input':
#                     in_channels = self.in_channels[self.backbone_end_level - 1]
#                 else:
#                     in_channels = out_channels
#                 extra_fpn_conv = ConvModule(
#                     in_channels,
#                     out_channels,
#                     3,
#                     stride=2,
#                     padding=1,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg,
#                     inplace=False)
#                 self.fpn_convs.append(extra_fpn_conv)
#
#     def forward(self, inputs):
#         """Forward function."""
#         # tmp = inputs[-1]
#         # tmp_out = tmp
#         # inputs = inputs[:-1]
#         assert len(inputs) == len(self.in_channels)
#
#         tokens = [tmp[0] for tmp in inputs]
#         locs = [tmp[1] for tmp in inputs]
#         feature_sizes = [tmp[2] for tmp in inputs]
#
#         # build laterals in token format
#         laterals = [
#             lateral_conv(tokens[i + self.start_level].unsuqeeze(2).permute(0,3,1,2)).permute(0,2,3,1).squeeze(2)
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]
#
#         # build top-down path tokens
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             laterals[i-1] += inter_points(laterals[i], locs[i+self.start_level], locs[i-1+self.start_level])
#
#         # tokens to feature map
#         for i in range(used_backbone_levels - 1, -1, -1):
#             map_size = feature_sizes[i+self.start_level]
#             kernel_size = self.kernel_size[i+self.start_level]
#             sigma = self.sigma[i+self.start_level]
#             laterals[i] = token2map(laterals[i], locs[i+self.start_level], map_size, kernel_size, sigma)
#
#         # build outputs
#         # part 1: from original levels
#         outs = [
#             self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#         ]
#         # part 2: add extra levels
#         if self.num_outs > len(outs):
#             # use max pool to get more levels on top of outputs
#             # (e.g., Faster R-CNN, Mask R-CNN)
#             if not self.add_extra_convs:
#                 for i in range(self.num_outs - used_backbone_levels):
#                     outs.append(F.max_pool2d(outs[-1], 1, stride=2))
#             # add conv layers on top of original feature maps (RetinaNet)
#             else:
#                 if self.add_extra_convs == 'on_input':
#                     src_lv = self.backbone_end_level - 1
#                     extra_source = token2map(tokens[src_lv], locs[src_lv], feature_sizes[src_lv], self.kernel_size[src_lv], self.sigma[src_lv])
#                     # extra_source = extra_source + tokens[0].sum() * 0
#                 elif self.add_extra_convs == 'on_lateral':
#                     extra_source = laterals[-1]
#                 elif self.add_extra_convs == 'on_output':
#                     extra_source = outs[-1]
#                 else:
#                     raise NotImplementedError
#                 outs.append(self.fpn_convs[used_backbone_levels](extra_source))
#                 for i in range(used_backbone_levels + 1, self.num_outs):
#                     if self.relu_before_extra_convs:
#                         outs.append(self.fpn_convs[i](F.relu(outs[-1])))
#                     else:
#                         outs.append(self.fpn_convs[i](outs[-1]))
#
#         # # debug
#         # for tmp_loc in locs:
#         #     tmp_out = tmp_out + tmp_loc.sum() * 0
#         # outs[-1] = outs[-1] + tmp_out
#         return tuple(outs)

class TokenFPN(FPN):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        caffe2_xavier_init (bool): Whether to apply caffe2_xavier_init on all
            conv in FPN. Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self, **kwargs):
        # self.num_heads = kwargs.pop('num_heads')
        self.kernel_size = kwargs.pop('kernel_size')
        self.sigma = kwargs.pop('sigma')
        super(TokenFPN, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        # tmp = inputs[-1]
        # tmp_out = tmp
        # inputs = inputs[:-1]
        assert len(inputs) == len(self.in_channels)

        tokens = [tmp[0] for tmp in inputs]
        locs = [tmp[1] for tmp in inputs]
        feature_sizes = [tmp[2] for tmp in inputs]

        # build laterals in token format
        laterals = [
            lateral_conv(tokens[i + self.start_level].unsqueeze(2).permute(0,3,1,2)).permute(0,2,3,1).squeeze(2)
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path tokens
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i-1] += inter_points(laterals[i], locs[i+self.start_level], locs[i-1+self.start_level])

        # tokens to feature map
        for i in range(used_backbone_levels - 1, -1, -1):
            map_size = feature_sizes[i+self.start_level]
            kernel_size = self.kernel_size[i+self.start_level]
            sigma = self.sigma[i+self.start_level]
            laterals[i] = token2map(laterals[i], locs[i+self.start_level], map_size, kernel_size, sigma)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    src_lv = self.backbone_end_level - 1
                    extra_source = token2map(tokens[src_lv], locs[src_lv], feature_sizes[src_lv], self.kernel_size[src_lv], self.sigma[src_lv])
                    # extra_source = extra_source + tokens[0].sum() * 0
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # # debug
        # for tmp_loc in locs:
        #     tmp_out = tmp_out + tmp_loc.sum() * 0
        # outs[-1] = outs[-1] + tmp_out
        return tuple(outs)


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


def token2map(x, loc, map_size, kernel_size, sigma):
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

    feature = feature / (mask + 1e-8)
    mask = (mask > 0).float()
    feature = reconstruct_feature(feature, mask, kernel_size, sigma)
    return feature


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
    feature_inter = out[:, :-1] / (out[:, [-1]] + 1e-8)
    out = feature + (1 - mask) * feature_inter
    return out


if __name__ == '__main__':
    # test_token2map()

    model = TokenFPN( in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        kernel_size=[1, 5, 3, 1],
        sigma=[2, 2, 2, 2],
        add_extra_convs='on_input',
        num_outs=5)

    input = []
    input.append([torch.zeros(1, 80, 64), torch.rand(1, 80, 2), [56, 56]])
    input.append([torch.zeros(1, 40, 128), torch.rand(1, 40, 2), [28, 28]])
    input.append([torch.zeros(1, 20, 320), torch.rand(1, 20, 2), [14, 14]])
    input.append([torch.zeros(1, 10, 512), torch.rand(1, 10, 2), [7, 7]])
    x = model(input)
