import torch
from torch import nn
from torch.nn import functional as F
from .dock_swinunet import dock_swinunet

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode="reflect"),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1, 1),
            nn.Conv2d(channel//2, channel//2, 1, 1)
        )


    def forward(self, x, feature_map=None):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.layer(up)
        if feature_map==None:
            return up
        else:
            return torch.cat((out, feature_map), dim=1)



class Unet_using_SwinUnet_as_Bottleneck(nn.Module):
    def __init__(self, config, in_channel=1, out_channel=2, img_size=1792, dock_channel=256):
        super(Unet_using_SwinUnet_as_Bottleneck, self).__init__()
        self.config = config
        self.num_classes = out_channel
        self.dock_channel = dock_channel


        self.c1 = DoubleConv(in_channel, 64) #448x448
        self.d1 = DownSample(64) #224x224
        self.c2 = DoubleConv(64, 128)
        self.d2 = DownSample(128) #112x112
        self.c3 = DoubleConv(128, 256)
        self.d3 = DownSample(256) #56x56
        self.c4 = DoubleConv(256, self.dock_channel)
        self.u1_bottleneck = UpSample(self.dock_channel) #112x112
        self.u2_bottleneck = UpSample(self.dock_channel) #224x224
        self.dock_swinunet = dock_swinunet(img_size=224,  # 对接尺寸为224
                                           patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                           in_chans=dock_channel,
                                           num_classes=dock_channel,
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
        self.d1_bottleneck = DownSample(self.dock_channel)#112x112
        self.d2_bottleneck = DownSample(self.dock_channel)#56x56
        # self.d4 = DownSample(512)
        # self.c5 = DoubleConv(512, 1024)
        # self.u1 = UpSample(1024)
        self.c6 = DoubleConv(self.dock_channel+self.dock_channel, 512)
        self.u2 = UpSample(512)
        self.c7 = DoubleConv(512, 256)
        self.u3 = UpSample(256)
        self.c8 = DoubleConv(256, 128)
        self.u4 = UpSample(128)
        self.c9 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channel, 3, 1, 1)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        #encoder区（下采样区）
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))
        # r5 = self.c5(self.d4(r4))

        #Bottleneck:swinunet
        x_bottleneck = self.u1_bottleneck(r4)  # 112x112
        x_bottleneck = self.u2_bottleneck(x_bottleneck)  # 224x224
        x_bottleneck = self.dock_swinunet(x_bottleneck)
        x_bottleneck = self.d1_bottleneck(x_bottleneck)
        x_bottleneck = self.d2_bottleneck(x_bottleneck)

        #decoder区
        o1 = self.c6(torch.cat((x_bottleneck, r4), dim=1))
        o2 = self.c7(self.u2(o1, r3))
        o3 = self.c8(self.u3(o2, r2))
        o4 = self.c9(self.u4(o3, r1))

        return self.out(o4)



