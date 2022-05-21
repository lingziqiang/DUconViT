import torch
from torch import nn
from torch.nn import functional as F

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


    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        self.c1 = DoubleConv(in_channel, 64)
        self.d1 = DownSample(64)
        self.c2 = DoubleConv(64, 128)
        self.d2 = DownSample(128)
        self.c3 = DoubleConv(128, 256)
        self.d3 = DownSample(256)
        self.c4 = DoubleConv(256, 512)
        self.d4 = DownSample(512)
        self.c5 = DoubleConv(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = DoubleConv(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = DoubleConv(512, 256)
        self.u3 = UpSample(256)
        self.c8 = DoubleConv(256, 128)
        self.u4 = UpSample(128)
        self.c9 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channel, 3, 1, 1)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        #encoder
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))
        r5 = self.c5(self.d4(r4))

        #decoder
        o1 = self.c6(self.u1(r5, r4))
        o2 = self.c7(self.u2(o1, r3))
        o3 = self.c8(self.u3(o2, r2))
        o4 = self.c9(self.u4(o3, r1))

        # return self.act(self.out(o4))
        return self.out(o4)

if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = UNet(3, 1)
    print(net(x).shape)


