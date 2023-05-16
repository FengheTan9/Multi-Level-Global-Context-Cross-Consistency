import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import numpy as np


class MSAG(nn.Module):
    """
    Multi-scale attention gate
    Arxiv: https://arxiv.org/abs/2210.13012
    """
    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):
    def __init__(self, dim=1024, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )

    def forward(self, x):
        x = self.block(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class Decoder(nn.Module):

    def __init__(self, dim_mult=4, with_masg=True):
        super(Decoder, self).__init__()
        print(with_masg)
        self.with_masg = with_masg
        self.Up5 = up_conv(ch_in=256 * dim_mult, ch_out=128 * dim_mult)
        self.Up_conv5 = conv_block(ch_in=128 * 2 * dim_mult, ch_out=128 * dim_mult)
        self.Up4 = up_conv(ch_in=128 * dim_mult, ch_out=64 * dim_mult)
        self.Up_conv4 = conv_block(ch_in=64 * 2 * dim_mult, ch_out=64 * dim_mult)
        self.Up3 = up_conv(ch_in=64 * dim_mult, ch_out=32 * dim_mult)
        self.Up_conv3 = conv_block(ch_in=32 * 2 * dim_mult, ch_out=32 * dim_mult)
        self.Up2 = up_conv(ch_in=32 * dim_mult, ch_out=16 * dim_mult)
        self.Up_conv2 = conv_block(ch_in=16 * 2 * dim_mult, ch_out=16 * dim_mult)
        self.Conv_1x1 = nn.Conv2d(16 * dim_mult, 1, kernel_size=1, stride=1, padding=0)

        self.msag4 = MSAG(128 * dim_mult)
        self.msag3 = MSAG(64 * dim_mult)
        self.msag2 = MSAG(32 * dim_mult)
        self.msag1 = MSAG(16 * dim_mult)

    def forward(self, feature):
        x1, x2, x3, x4, x5 = feature
        if self.with_masg:
            x4 = self.msag4(x4)
            x3 = self.msag3(x3)
            x2 = self.msag2(x2)
            x1 = self.msag1(x1)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1


class MGCC(nn.Module):
    def __init__(self, img_ch=3, length=(3, 3, 3), k=7, dim_mult=4):
        """
        Multi-Level Global Context Cross Consistency Model
        Args:
            img_ch : input channel.
            output_ch: output channel.
            length: number of convMixer layers
            k: kernal size of convMixer

        """
        super(MGCC, self).__init__()

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16 * dim_mult)
        self.Conv2 = conv_block(ch_in=16 * dim_mult, ch_out=32 * dim_mult)
        self.Conv3 = conv_block(ch_in=32 * dim_mult, ch_out=64 * dim_mult)
        self.Conv4 = conv_block(ch_in=64 * dim_mult, ch_out=128 * dim_mult)
        self.Conv5 = conv_block(ch_in=128 * dim_mult, ch_out=256 * dim_mult)
        self.ConvMixer1 = ConvMixerBlock(dim=256 * dim_mult, depth=length[0], k=k)
        self.ConvMixer2 = ConvMixerBlock(dim=256 * dim_mult, depth=length[1], k=k)
        self.ConvMixer3 = ConvMixerBlock(dim=256 * dim_mult, depth=length[2], k=k)
        # main Decoder
        self.main_decoder = Decoder(dim_mult=dim_mult, with_masg=True)
        # aux Decoder
        self.aux_decoder1 = Decoder(dim_mult=dim_mult, with_masg=True)
        self.aux_decoder2 = Decoder(dim_mult=dim_mult, with_masg=True)
        self.aux_decoder3 = Decoder(dim_mult=dim_mult, with_masg=True)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        if not self.training:
            x5 = self.ConvMixer1(x5)
            x5 = self.ConvMixer2(x5)
            x5 = self.ConvMixer3(x5)
            feature = [x1, x2, x3, x4, x5]
            main_seg = self.main_decoder(feature)
            return main_seg

        # FeatureNoise
        feature = [x1, x2, x3, x4, x5]
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)

        x5 = self.ConvMixer1(x5)
        feature = [x1, x2, x3, x4, x5]
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)

        x5 = self.ConvMixer2(x5)
        feature = [x1, x2, x3, x4, x5]
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)

        # main decoder
        x5 = self.ConvMixer3(x5)
        feature = [x1, x2, x3, x4, x5]
        main_seg = self.main_decoder(feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3
