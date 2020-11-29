import segmentation_models_pytorch as smp
import torch
import os
import logging
from catalyst.contrib.nn import Adam, AdamW, RAdam, Lookahead, SGD
from catalyst.utils import process_model_params
from pytorch_toolbelt.losses import DiceLoss
import torch.nn as nn
from catalyst.dl import CriterionCallback, MetricAggregationCallback
from pytorch_toolbelt.utils.catalyst import IoUMetricsCallback
_logger = logging.getLogger(__name__)


def create_model(
        model_name,
        encoder_name,
        pretrained=False,
        num_classes=6,
        in_chans=3,
        checkpoint_path='',
        **kwargs):
    """Create a model
    Args:
        model_name (str): name of model to instantiate
        encoder_name (str): name of encoder to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final layer (default 6)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized
    Keyword Args:
        **: other kwargs are model specific
    """
    # I should probably rewrite it
    if pretrained:
        weights = 'imagenet'
    else:
        weights = None

    if model_name == 'unetpluspluts':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=weights,
            classes=num_classes,
            in_channels=in_chans,
            **kwargs)
    elif model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            classes=num_classes,
            in_channels=in_chans,
            **kwargs)
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=weights,
            classes=num_classes,
            in_channels=in_chans,
            **kwargs)
    elif model_name == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            classes=num_classes,
            in_channels=in_chans,
            **kwargs)
    elif model_name == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            classes=num_classes,
            in_channels=in_chans,
            **kwargs)
    else:
        raise NotImplementedError()

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model


def load_state_dict(checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'model_state_dict'
        if state_dict_key in checkpoint:
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict, strict=strict)


def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    layerwise_params = {"encoder*": dict(lr=args.lr*args.ev, weight_decay=args.weight_decay)}
    parameters = process_model_params(model, layerwise_params=layerwise_params)

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = AdamW(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def create_criterion(args):
    criterion_lower = args.criterion.lower()
    criterion_names = criterion_lower.split('_')

    def get_criterion_by_name(criterion_name):
        if criterion_name == 'ce':
            return nn.CrossEntropyLoss()
        elif criterion_name == 'dice':
            return DiceLoss(mode='multiclass')
        else:
            raise NotImplementedError

    criterion = {}
    for cn in criterion_names:
        criterion[cn] = get_criterion_by_name(cn)

    return criterion, criterion_names


def create_callbacks(args, criterion_names):
    callbacks = [
        IoUMetricsCallback(
            mode='multiclass',
            input_key=args.input_target_key,
            class_names=args.class_names.split(',') if args.class_names else None
        )
    ]
    metrics_weights = {}
    for cn in criterion_names:
        callbacks.append(CriterionCallback(input_key=args.input_target_key, prefix=f"loss_{cn}", criterion_key=cn))
        metrics_weights[f'loss_{cn}'] = 1.0
    callbacks.append(MetricAggregationCallback(prefix="loss", mode="weighted_sum", metrics=metrics_weights))
    return callbacks
