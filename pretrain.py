import torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

# *user-defined
from models import SLRCLIP, Text_Decoder
import utils as utils
from datasets import S2T_Dataset

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import wandb
import copy
from pathlib import Path
import math
import sys
from typing import Iterable, Optional
from loguru import logger

# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER

# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from timm.loss import SoftTargetCrossEntropy
from timm.optim import AdamW

# visualization
from torchvision.utils import save_image, make_grid
from PIL import Image

from hpman.m import _
import hpargparse

# global definition
from definition import *


def get_args_parser():
    parser = argparse.ArgumentParser("Visual-Keypoint-Fusion-language-Pretraining", add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument("--epochs", default=2, type=int)

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')  # 设置分布式训练的进程数，默认值为1。

    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')  # 设置分布式训练的URL，默认值为env://。
    parser.add_argument('--local_rank', default=0, type=int)  # 设置本地进程的rank，默认值为0。

    # 微调参数
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')  # 从检查点微调模型的路径，默认值为空。

    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')  # 设置优化器类型，默认值为adamw。
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')  # 设置优化器的epsilon值，默认值为1.0e-09。
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')  # 设置优化器的beta值，默认值为None，即使用优化器的默认值。
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')  # 设置梯度裁剪的范数，默认值为None，即不进行裁剪。
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')  # 设置SGD优化器的动量，默认值为0.9。
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')  # 设置权重衰减系数，默认值为0.0。

    # 学习率调度参数
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')  # 设置学习率调度器，默认值为cosine。
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')  # 设置初始学习率，默认值为1.0e-3。
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')  # 设置学习率噪声的启用和关闭的百分比，默认值为None。
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')  # 设置学习率噪声的限制百分比，默认值为0.67。
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')  # 设置学习率噪声的标准差，默认值为1.0。
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')  # 设置预热学习率，默认值为1e-6。
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')  # 设置最小学习率，默认值为1.0e-08。

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')  # 设置学习率衰减的间隔轮数，默认值为30。
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')  # 设置学习率预热的轮数，默认值为0。
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')  # 设置学习率冷却的轮数，默认值为10。
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')  # 设置Plateau学习率调度器的耐心轮数，默认值为10。
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')  # 设置学习率衰减率，默认值为0.1。

    # 基础参数
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')  # 设置保存路径，默认值为空，即不保存。
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')  # 设置用于训练/测试的设备，默认值为cuda。
    parser.add_argument('--seed', default=0, type=int)  # 设置随机种子，默认值为0。
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # 从检查点恢复训练，默认值为空。
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')  # 设置开始的轮数，默认值为0。
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')  # 设置只进行评估模式。
    parser.add_argument('--num_workers', default=4, type=int)  # 设置DataLoader的工作线程数，默认值为10。
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')  # 设置DataLoader是否固定CPU内存，以提高传输效率。
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')  # 设置不固定CPU内存。
    parser.set_defaults(pin_mem=True)  # 默认固定CPU内存。
    parser.add_argument('--config', type=str,
                        default='./configs/config_gloss_free.yaml')  # 设置配置文件路径，默认值为'./configs/config_gloss_free.yaml'。

    # 数据处理参数
    # parser.add_argument('--input-size', default=224, type=int) # 设置输入图像大小，默认值为224。
    parser.add_argument('--input-size', default=1280, type=int)
    parser.add_argument('--input-size2', default=720, type=int)
    parser.add_argument('--resize', default=256, type=int)  # 设置图像调整大小的尺寸，默认值为256。

    # wandb参数
    parser.add_argument("--log_all", action="store_true",
                        help="flag to log in all processes, otherwise only in rank0")  # 设置是否在所有进程中记录日志，默认为仅在rank0进程中记录。
    parser.add_argument("--entity", type=str,
                        help="wandb entity")  # 设置wandb的实体名称。
    parser.add_argument("--project", type=str, default='VLP',
                        help="wandb project")  # 设置wandb的项目名称，默认值为'VLP'。

    # 噪声参数
    parser.add_argument('--training-refurbish', default=True, type=bool)  # 设置是否进行训练数据的刷新，默认值为True。
    parser.add_argument('--noise-rate', default=0.15, type=float)  # 设置噪声率，默认值为0.15。
    parser.add_argument('--noise-type', default='omit_last', type=str,
                        choices=['omit', 'omit_last'])  # 设置噪声类型，默认值为'omit_last'。
    parser.add_argument('--random-shuffle', default=False, type=bool)  # 设置是否随机打乱数据，默认值为False。

    parser.add_argument('--loss-lambda', type=float, default=1.0, metavar='RATE',
                        help='lambda param')  # 设置损失函数的lambda参数，默认值为1.0。

    parser.add_argument("--random_shuffle", action='store_true', )
    return parser


def main(args, config):
    utils.init_distributed_mode(args) #分布式
    print(args)
    device = torch.device(args.device)

    # 保存训练种子，为了复现实验
    # seed = args.seed + utils.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 启用CuDNN的调优功能以提高运算性能
    cudnn.benchmark = True

    # 如果你希望禁用CuDNN的调优功能以确保结果可重复性，可以设置为False
    cudnn.benchmark = False

    print(f"模型创建")
    tokenizer = MBartTokenizer.from_pretrained(config["model"]["tokenizer"])
    train_data = S2T_Dataset(
        path=config['data']['train_label_path'],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase='train',
        training_refurbish=True
    )
    print("训练数据", train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=train_data.collate_fn,
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=args.random_shuffle,
    )

    dev_data = S2T_Dataset(
        path=config['data']['dev_label_path'],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase='dev',
        training_refurbish=True
    )
    print("验证数据", dev_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_dataloader = DataLoader(
        dev_data, batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        sampler=dev_sampler,
        pin_memory=args.pin_mem,

    )

    test_data = S2T_Dataset(
        path=config['data']['test_label_path'],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase='test',
        training_refurbish=True
    )
    print("测试数据", test_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=test_data.collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        sampler=test_sampler,
    )

    print("VK-MODEL 创建")
    model = SLRCLIP(config=config)
    model.to(device)
    print(model)

    # 何加载一个预训练模型的权重，并处理在加载过程中可能出现的键缺失或多余的情况
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model

    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    text_decoder = Text_Decoder(config).to(device)

    if args.distributed:
        text_decoder = torch.nn.parallel.DistributedDataParallel(text_decoder, device_ids=[args.gpu],find_unused_parameters=False)
    optimizer_td = AdamW(text_decoder.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98))
    lr_scheduler_td = scheduler.CosineAnnealingLR(
        optimizer=optimizer_td,
        eta_min=1e-8,
        T_max=args.epochs,
    )

    TD_train_dict = dict(
        optimizer=optimizer_td,
        lr_scheduler=lr_scheduler_td,
        text_decoder=text_decoder
    )
    # 创建损失函数实例
    criterion = utils.KLLoss()
    # 使用 KL 散度作为损失函数
    # 创建梯度缩放器实例
    loss_scaler = NativeScaler()
    # 使用原生梯度缩放器进行混合精度训练

    output_dir = Path(args.output_dir)

    # 从检查点恢复模型、优化器和学习率调度器的状态
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        if args.eval:
            if not args.resume:
                logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
            dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, args.start_epoch,
                                 UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
            print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

            test_stats = evaluate(args, test_dataloader, model, model_without_ddp, criterion, config, args.start_epoch,
                                  UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
            print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
            return

    print(f"Start training for {args.epochs} epochs")
    print("开始训练")
    start_time = time.time()
    min_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config,
                                      PAD_IDX, loss_scaler, TD_train_dict)

        lr_scheduler.step(epoch)
        TD_train_dict['lr_scheduler'].step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'text_decoder': TD_train_dict['text_decoder'].state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
                              SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)

        if min_loss > test_stats["loss"]:
            min_loss = test_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / f'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'text_decoder': TD_train_dict['text_decoder'].state_dict(),
                        'epoch': epoch,
                        # 'args': args,

                    }, checkpoint_path)
        print(f"* DEV loss {test_stats['loss']:.3f} Min DEV loss {min_loss}")
        if args.run:
            args.run.log({'epoch': epoch + 1, 'training/train_loss': train_stats['loss'],
                          'training/masked_lm_loss': train_stats['masked_lm_loss'], 'dev/dev_loss': test_stats['loss'],
                          'dev/min_loss': min_loss})
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        torch.distributed.barrier()
        checkpoint = torch.load(args.output_dir + '/best_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        # 考虑下evaluate里有没有分布式
        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
                             SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
                              SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS,
             PAD_IDX, device, TD_train_dict):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    with torch.no_grad():
        for step, (src_input, tgt_input, masked_tgt_input) in enumerate(
                metric_logger.log_every(dev_dataloader, print_freq, header)):

            logits_per_image, logits_per_text, ground_truth = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)

            lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model_without_ddp.model_txt)
            masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].cuda().view(-1))
            total_loss = (loss_imgs + loss_texts) / 2.

            metric_logger.update(loss=total_loss.item())
            metric_logger.update(masked_lm_loss=masked_lm_loss.item())

            if (step + 1) % 10 == 0 and utils.is_main_process():
                visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
                utils.visualization([visual_map, ])

    if args.run:
        args.run.log({'epoch': epoch + 1, 'epoch/dev_loss': total_loss.item()})

    metric_logger.synchronize_between_processes()
    print("* Averaged stats:", metric_logger)
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, PAD_IDX, loss_scaler, TD_train_dict, max_norm: float = 0,
                    set_training_mode=True):
    '''
    定义一个训练一个 epoch 的函数。
参数 args 是命令行参数。
参数 model 是要训练的模型。
参数 criterion 是损失函数，这里是 nn.CrossEntropyLoss。
参数 data_loader 是训练数据的迭代器。
参数 optimizer 是优化器。
参数 device 是用于训练的设备（如 GPU）。
参数 epoch 是当前的 epoch 编号。
参数 config 是配置文件（可能包含一些超参数）。
参数 PAD_IDX 是填充索引，用于忽略填充部分的损失计算。
参数 loss_scaler 是用于混合精度训练的损失缩放器。
参数 TD_train_dict 是包含优化器和文本解码器的字典。
参数 max_norm 是用于梯度裁剪的最大范数。
参数 set_training_mode 是布尔值，控制是否设置模型为训练模式。
    '''
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    for step, (src_input, tgt_input, masked_tgt_input) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text, ground_truth = model(src_input, tgt_input)
            # criterion = utils.KLLoss()
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            total_loss = (loss_imgs + loss_texts) / 2.

        print(
            f"Step {step}: loss_imgs = {loss_imgs.item()}, loss_texts = {loss_texts.item()}, total_loss = {total_loss.item()}")

        loss_scaler(total_loss, optimizer)

        if step % 5 == 0:
            TD_train_dict['optimizer'].zero_grad()
            with torch.cuda.amp.autocast():
                lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model.module.model_txt)
                # lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model.model_txt)

                masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]),
                                          tgt_input['input_ids'].cuda().view(-1)) * args.loss_lambda
            loss_scaler(masked_lm_loss, TD_train_dict['optimizer'])
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("loss 是{},停止训练".format(loss_value))
            sys.exit(1)
            # 更新记录器中的指标
        metric_logger.update(loss=loss_value)
        metric_logger.update(masked_lm_loss=masked_lm_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(td_lr=TD_train_dict['optimizer'].param_groups[0]['lr'])

        # 每个 mini-batch 处理后释放显存
        torch.cuda.empty_cache()

        # 每10步进行一次可视化
        if (step + 1) % 10 == 0 and utils.is_main_process():
            visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
            utils.visualization([visual_map, ])

    # 如果args中有run，则记录日志
    if args.run:
        args.run.log(
            {'epoch': epoch + 1, 'epoch/train_loss': loss_value, 'epoch/masked_lm_loss': masked_lm_loss.item()})

    # 从所有进程中同步统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def setup_run(args, config):
    if args.log_all:
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("training/*", step_metric="epoch")
        run.define_metric("dev/*", step_metric="epoch")
    else:
        if utils.is_main_process():
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            run.name = args.output_dir.split('/')[-1]
        run.name = args.output_dir.split('/')[-1]

    return run


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 禁用并行处理，以避免某些情况下的多线程问题。

    parser = argparse.ArgumentParser("Visual-Keypoint-Fusion-language-Pretraining", parents=[get_args_parser()])

    # 解析当前文件所在路径中的某个文件，具体文件未明确
    _.parse_file(Path(__file__).resolve().parent)

    # 绑定hpargparse参数
    hpargparse.bind(parser, _)

    # 解析命令行参数
    args = parser.parse_args()

    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args.run = setup_run(args, config)

    # 如果指定了输出目录，则创建该目录及其父目录（如果不存在）
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)
