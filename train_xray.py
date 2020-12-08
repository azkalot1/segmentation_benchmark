""" PanNule Training Script
This is intended to be a lean and easily modifiable training script that trains the model
with some of the latest networks and training techniques using catalyst.
Based on Ross Wightman (https://github.com/rwightman) ImageNet Training Script
This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)
NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)
"""

import argparse
import yaml
import os
import numpy as np
import logging
from datetime import datetime
import torch

from timm.utils import setup_default_logging
from src.utils import create_model, create_optimizer, create_criterion, create_callbacks, create_scheduler, get_outdir
from src.augmentations import get_training_trasnforms_xray, get_valid_transforms_xray
from src.datasets import ChestXRayDataset
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='PyTorch PanNuke Training using Catalyst')
# Dataset / Model parameters
parser.add_argument('--model', default='unet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "unet"')
parser.add_argument('--encoder', default='resnet18', type=str, metavar='MODEL',
                    help='Name of encoder (default: "resnet18"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1, metavar='N',
                    help='number of label classes (default: 6)')
parser.add_argument('--images-dir', default='data/ChestXray/images', type=str,
                    help='Path to images')
parser.add_argument('--masks-dir', default='data/ChestXray/masks', type=str,
                    help='Path to masks')
parser.add_argument('--train-fold', default='1', type=str,
                    help='train fold idx')
parser.add_argument('--valid-fold', default='2', type=str,
                    help='valid fold idx')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
# Criterion parametrs
parser.add_argument('--criterion', default='dice_bce', type=str,
                    help='Criterion (default: DICE + BCE)')
# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='plateau', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "plateau"')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--ev', type=float, default=0.1, metavar='LR',
                    help='multiplier for encoder learning rate (default: 0.1)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-scheduler', type=int, default=5, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 5)')
parser.add_argument('--factor-scheduler', '--fs', type=float, default=0.1, metavar='N',
                    help='LR decay for Plateau LR scheduler (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--aug-type', type=str, default='light',
                    help='Augs type: light, medium, heavy')
# TO DO: add mixup\cutmix
# parser.add_argument('--mixup', type=float, default=0.0,
#                     help='mixup alpha, mixup enabled if > 0. (default: 0.)')
# parser.add_argument('--cutmix', type=float, default=0.0,
#                     help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
# parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
#                     help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
# parser.add_argument('--mixup-prob', type=float, default=1.0,
#                     help='Probability of performing mixup or cutmix when either/both is enabled')
# parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
#                     help='Probability of switching to cutmix when both mixup and cutmix enabled')
# parser.add_argument('--mixup-mode', type=str, default='batch',
#                     help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
# parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
#                     help='Turn off mixup after this epoch, disabled if 0 (default: 0)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('-j', '--workers', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 16)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use mixed precision')
parser.add_argument('--eval-metric', default='loss', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "loss")')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
# Catalyst
parser.add_argument('--input-target-key', default='mask', type=str,
                    help='Runner input target key (default: "mask")')
parser.add_argument('--input-key', default='features', type=str,
                    help='Runner input key (default: "features")')
parser.add_argument('--patience', default=10, type=int,
                    help='Patience for early stopping (default: 10)')
parser.add_argument('--save-n-best', default=5, type=int,
                    help='N best epochs to save (default: 5)')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)
    torch.manual_seed(args.seed + args.rank)

    # prepare model
    model = create_model(
        args.model,
        args.encoder,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=1,
        checkpoint_path=args.initial_checkpoint)

    # prepare optimizer
    optimizer = create_optimizer(args, model)

    # prepare scheduler
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # prepare dataset
    images = os.listdir(args.images_dir)
    masks = np.array([args.masks_dir+image_path for image_path in images])
    images = np.array([args.images_dir+image_path for image_path in images])
    images_train, images_valid, masks_train, masks_valid = train_test_split(
        images,
        masks,
        test_size=0.25,
        random_state=42
        )

    if args.no_aug:
        train_dataset = ChestXRayDataset(images_train, masks_train, get_valid_transforms_xray())
    else:
        train_dataset = ChestXRayDataset(images_train, masks_train, get_training_trasnforms_xray(args.aug_type))
    val_dataset = ChestXRayDataset(images_valid, masks_valid, get_valid_transforms_xray())

    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=True),
        'valid': DataLoader(
            val_dataset,
            batch_size=args.batch_size*args.validation_batch_size_multiplier,
            num_workers=args.workers,
            pin_memory=True,
            shuffle=False)
    }

    # save config
    output_dir = ''
    output_base = args.output if args.output else './logs'
    exp_name = '-'.join([
        'ChestXray',
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        args.encoder,
        args.aug_type,
        args.opt.lower()
    ])
    output_dir = get_outdir(output_base, 'train', exp_name)

    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    criterion, criterion_names = create_criterion(args)
    callbacks = create_callbacks(args, criterion_names)
    eval_metric = args.eval_metric
    minimize_metric = True if eval_metric == 'loss' else False
    runner = SupervisedRunner(input_key=args.input_key, input_target_key=args.input_target_key)
    # set fp16
    if args.fp16:
        fp16_params = dict(opt_level="O1")  # params for FP16
        _logger.info('Using fp16 O1')
    else:
        fp16_params = None
        _logger.info('Not using fp16 O1')
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=output_dir,
        num_epochs=num_epochs,
        main_metric=eval_metric,
        minimize_metric=minimize_metric,
        verbose=True,
        fp16=fp16_params,
    )


if __name__ == '__main__':
    main()
