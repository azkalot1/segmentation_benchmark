#!/bin/bash
# train all Unets++
#  RegNet
python train_pannuke.py --model unetplusplus -b 64 --encoder timm-regnetx_002 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 64 --encoder timm-regnety_002 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 64 --encoder timm-regnetx_004 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 64 --encoder timm-regnety_004 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 64 --encoder timm-regnetx_006 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 64 --encoder timm-regnety_006 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 64 --encoder timm-regnetx_008 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 64 --encoder timm-regnety_008 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-regnetx_016 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-regnety_016 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-regnetx_032 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-regnety_032 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-regnetx_040 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-regnety_040 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-regnetx_064 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-regnety_064 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-regnetx_120 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-regnety_120 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder timm-regnety_160 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder timm-regnetx_160 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 8 --encoder timm-regnety_320 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 8 --encoder timm-regnetx_320 --opt lookahead_radam --pretrained --fp16
#  ResNest
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-resnest14d --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-resnest26d --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-resnest50d --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder timm-resnest101e --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 8 --encoder timm-resnest200e --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 8 --encoder timm-resnest269e --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-resnest50d_4s2x40d --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-resnest50d_1s4x24d --opt lookahead_radam --pretrained --fp16
#  Res2Net
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-res2net50_26w_4s --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-res2net50_48w_2s --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-res2net50_14w_8s --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder timm-res2net50_26w_6s --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder timm-res2net50_26w_8s --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder timm-res2net101_26w_4s --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder timm-res2next50 --opt lookahead_radam --pretrained --fp16
#  SkNet
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-skresnet18 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-skresnet34 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-skresnext50_32x4d --opt lookahead_radam --pretrained --fp16
#  ResNet
python train_pannuke.py --model unetplusplus -b 48 --encoder resnet18 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 48 --encoder resnet34 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder resnet50 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder resnet101 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder resnet152 --opt lookahead_radam --pretrained --fp16
#  EffNet
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-efficientnet-b0 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-efficientnet-b1 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-efficientnet-b2 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 32 --encoder timm-efficientnet-b3 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 24 --encoder timm-efficientnet-b4 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 16 --encoder timm-efficientnet-b5 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 12 --encoder timm-efficientnet-b6 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 8 --encoder timm-efficientnet-b7 --opt lookahead_radam --pretrained --fp16
python train_pannuke.py --model unetplusplus -b 8 --encoder timm-efficientnet-b8 --opt lookahead_radam --pretrained --fp16