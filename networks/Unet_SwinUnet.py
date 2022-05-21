# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .Unet import UNet
from .vision_transformer import *
import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class Unet_SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,
                 in_channels=1, dock_channels=128):
        super(Unet_SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.in_channels = in_channels
        self.dock_channels = dock_channels
        self.unet = UNet(in_channels, dock_channels)
        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=dock_channels,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        x = self.unet(x)
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits


