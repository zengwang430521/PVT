# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate, my_train_one_epoch
from losses import DistillationLoss
from samplers import RASampler
# import models
import pvt
import pvt_v2
import pvt_v2_20_2
import pvt_v2_20_2g
import pvt_v2_3
import pvt_v2_3a, pvt_v2_3a0, pvt_v2_3a1, pvt_v2_3a2
import pvt_v2_3b, pvt_v2_3b2, pvt_v2_3b3
import pvt_v2_3c, pvt_v2_3c2, pvt_v2_3c3
import pvt_v2_3d
import pvt_v2_3e
import pvt_v2_3f, pvt_v2_3f2, pvt_v2_3f3, pvt_v2_3f4, pvt_v2_3f5, \
    pvt_v2_3f6, pvt_v2_3f7, pvt_v2_3f8, pvt_v2_3f9, pvt_v2_3f10, pvt_v2_3f11, pvt_v2_3f12
import pvt_v2_4
import pvt_v2_4, pvt_v2_4b0
import pvt_v2_5, pvt_v2_5a, pvt_v2_5b, pvt_v2_5b0, pvt_v2_5c, pvt_v2_5c0

import utils
import collections
import samplers
from torch.utils.data import DataLoader
import os
from main import get_args_parser


def main(args):
    utils.my_init_distributed_mode(args)
    print(args)
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # if True:  # args.distributed:
    #     num_tasks = utils.get_world_size()
    #     global_rank = utils.get_rank()
    #     if args.repeated_aug:
    #         sampler_train = RASampler(
    #             dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #         )
    #     else:
    #         sampler_train = torch.utils.data.DistributedSampler(
    #             dataset_train,
    #             # num_replicas=num_tasks,
    #             num_replicas=0,
    #             rank=global_rank, shuffle=True
    #         )
    #     if args.dist_eval:
    #         if len(dataset_val) % num_tasks != 0:
    #             print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
    #                   'This will slightly alter validation results as extra duplicate entries are added to achieve '
    #                   'equal num of samples per-process.')
    #         sampler_val = torch.utils.data.DistributedSampler(
    #             dataset_val,
    #             # num_replicas=num_tasks,
    #             num_replicas=0,
    #             rank=global_rank, shuffle=False)
    #     else:
    #         sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    #
    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    #
    # data_loader_val = torch.utils.data.DataLoader(
    #     dataset_val, sampler=sampler_val,
    #     batch_size=int(1.5 * args.batch_size),
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False
    # )
    #

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True)







    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    # model_without_ddp = model

    # there are bugs
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        _ = model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device='cpu' if args.model_ema_force_cpu else '',
    #         resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # teacher_model = None
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.nb_classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    # criterion = DistillationLoss(
    #     criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    # )
    criterion = DistillationLoss(
        criterion, None, 'none', 0, 0
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            if not os.path.exists(args.resume):
                checkpoint = None
                print('NOTICE:' + args.resume + ' does not exist!')
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')

        if checkpoint is not None:
            if 'model' in checkpoint:
                model_without_ddp.load_state_dict(checkpoint['model'])
            else:
                model_without_ddp.load_state_dict(checkpoint)

            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                # if args.model_ema:
                #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])

            print('resume from' + args.resume)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    # max_epoch_dp_warm_up = 100
    # if 'pvt_tiny' in args.model or 'pvt_small' in args.model:
    #     max_epoch_dp_warm_up = 0
    # if args.start_epoch < max_epoch_dp_warm_up:
    #     model_without_ddp.reset_drop_path(0.0)
    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.fp32_resume and epoch > args.start_epoch + 1:
    #         args.fp32_resume = False
    #     loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)
    #
    #     if epoch == max_epoch_dp_warm_up:
    #         model_without_ddp.reset_drop_path(args.drop_path)
    #
    #     if args.distributed:
    #         # data_loader_train.sampler.set_epoch(epoch)
    #         sampler_train.set_epoch(epoch)
    #
    #     train_stats = my_train_one_epoch(
    #         model, criterion, data_loader_train,
    #         optimizer, device, epoch, loss_scaler,
    #         args.clip_grad, model_ema, mixup_fn,
    #         set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
    #         fp32=args.fp32_resume
    #     )
    #
    #     lr_scheduler.step(epoch)
    #     if args.output_dir:
    #         checkpoint_paths = [output_dir / 'checkpoint.pth']
    #         for checkpoint_path in checkpoint_paths:
    #             utils.save_on_master({
    #                 'model': model_without_ddp.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'lr_scheduler': lr_scheduler.state_dict(),
    #                 'epoch': epoch,
    #                 # 'model_ema': get_state_dict(model_ema),
    #                 'scaler': loss_scaler.state_dict(),
    #                 'args': args,
    #             }, checkpoint_path)
    #
    #     test_stats = evaluate(data_loader_val, model, device)
    #     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #     max_accuracy = max(max_accuracy, test_stats["acc1"])
    #     print(f'Max accuracy: {max_accuracy:.2f}%')
    #
    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                  **{f'test_{k}': v for k, v in test_stats.items()},
    #                  'epoch': epoch,
    #                  'n_parameters': n_parameters}
    #
    #     if args.output_dir and utils.is_main_process():
    #         with (output_dir / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    for epoch in range(args.start_epoch, args.epochs):
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        if args.distributed:
            # data_loader_train.sampler.set_epoch(epoch)
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
