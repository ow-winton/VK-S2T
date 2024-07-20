# *torch
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

# 参数管理，这部分管理传入模型的参数

def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', add_help=False)
    parser.add_argument('--input-size', default=1280, type=int)
    parser.add_argument('--input-size2', default=720, type=int)
    parser.add_argument('--training-refurbish', default=True, type=bool)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--batch-size', default=2, type=int)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=10, type=int)
    return parser


def main(args,config):
    # 由于我只有一个卡，所以分布式的部分暂时注释
    # utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # 固定随机数种子以确保可重复性
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False


    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])
    train_data = S2T_Dataset(
        path=config['data']['train_label_path'],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase='train',
        training_refurbish=True
    )

    print(train_data)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    # 这个也是分布式的代码，暂时不用
    train_dataloader = DataLoader(
        train_data,
        batch_size= args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        # sampler=train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True
    )


    dev_data = S2T_Dataset(
        path=config['data']['dev_label_path'],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase='val',
        training_refurbish=True
    )

    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    # 涉及分布式，所以注释
    dev_dataloader = DataLoader(
        dev_data,
        batch_size= args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        pin_memory=args.pin_mem,
        # sampler= dev_sampler
    )

    test_data = S2T_Dataset(path=config["data"]["test_label_path"],
                            tokenizer = tokenizer,
                            config= config,
                            args = args,
                            phase="test",
                            training_refurbish=True
                            )
    print(test_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle= False)
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
        # sampler=test_sampler,
        pin_memory=args.pin_mem,
    )

    print(f"Creating model:")
    model = SLRCLIP(config=config)
    model.to(device)
    print(model)
