import torch
import torch.nn as nn
from models.ConvBlock import ConvBlock
from models.UDCUnet.JND_Attention import JNDNet
from .adaptive_addtionParams import AdaptivelyAddition
from models.UDCUnet.arch_util import ResidualBlockNoBN, ResBlock_with_SFT


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, 1, 1),
            ResidualBlockNoBN(out_ch)
        )
        self.downsample = nn.Sequential(
            ConvBlock(out_ch, out_ch, 3, 2, 1),
        )

    def forward(self, x):
        try:
            """
            :param x: 一个 tensor
            :return: out输出到深层，out_2输入到下一层，
            """
            out = self.Conv_BN_ReLU_2(x)
            out_2 = self.downsample(out)
        except:
            """
            :param x: 一个列表
            :return: out输出到深层，out_2输入到下一层，
            """
            x_ori = x[0]
            x_back = x[1]
            out = self.Conv_BN_ReLU_2(x_ori)
            out_2 = self.downsample(out + x_back)

        return out, out_2


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            ConvBlock(in_ch, out_ch * 2, 3, 1, 1),
            ResidualBlockNoBN(out_ch*2)
        )
        self.SFT = ResBlock_with_SFT(64, out_ch * 2)
        self.upsample = nn.Sequential(
            ConvBlock(out_ch * 2, out_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
        )
        # self.addtion_factor = AdaptivelyAddition(out_ch, 4)

    def forward(self, x, cond):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''

        x_out = self.Conv_BN_ReLU_2(x)
        x_out, _ = self.SFT([x_out, cond])
        x_out = self.upsample(x_out)
        # factor1, factor2 = self.addtion_factor(out, x_out)
        # cat_out = factor1 * x_out + factor2 * out
        return x_out


class CoverDown(nn.Module):
    def __init__(self, inc, nf=64):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(CoverDown, self).__init__()
        self.down1 = ConvBlock(inc, nf, 3, 2, 1)
        self.down2 = ConvBlock(nf, nf, 3, 2, 1)
        self.down3 = ConvBlock(nf, nf, 3, 2, 1)
        self.down4 = ConvBlock(nf, nf, 3, 2, 1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        return (down1, down2, down3, down4)


class UNet(nn.Module):
    def __init__(self, inc=3, outc=3, nf=32):
        super(UNet, self).__init__()
        out_channels = [nf * (2 ** (i + 1)) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 下采样
        self.downC = CoverDown(nf//2, out_channels[0])
        self.pre_layer = nn.Sequential(
            ConvBlock(inc, out_channels[0], 3, 1, 1),
            ConvBlock(out_channels[0], out_channels[0], 3, 1, 1)
        )
        self.d1 = DownsampleLayer(out_channels[0], out_channels[0])  # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
        # self.d1JND = JNDNet(out_channels[0], out_channels[0], Wnet(SEP2Embedwater)+SEDecoder_COCO)
        # self.d2JND = JNDNet(out_channels[Wnet(SEP2Embedwater)+SEDecoder_COCO], out_channels[Wnet(SEP2Embedwater)+SEDecoder_COCO], Wnet(SEP2Embedwater)+SEDecoder_COCO)
        # self.d3JND = JNDNet(out_channels[2], out_channels[2], Wnet(SEP2Embedwater)+SEDecoder_COCO)
        # self.d4JND = JNDNet(out_channels[3], out_channels[3], Wnet(SEP2Embedwater)+SEDecoder_COCO)
        # 上采样
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[3], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[2], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[1], out_channels[0])  # 256-128-64
        # 输出
        self.out = nn.Sequential(
            ResidualBlockNoBN(out_channels[0]),
            ConvBlock(out_channels[0], outc, 3, 1, 1),
        )
        self.AdaptLayer1 = AdaptivelyAddition(out_channels[0], 16)
        self.AdaptLayer2 = AdaptivelyAddition(out_channels[1], 16)
        self.AdaptLayer3 = AdaptivelyAddition(out_channels[2], 16)
        self.AdaptLayer4 = AdaptivelyAddition(out_channels[3], 16)

    def forward(self, x):
        cover = x[0]
        water = x[1]
        coverDown = self.downC(cover)
        concat = torch.cat((cover, water), dim=1)
        x_pre = self.pre_layer(concat)
        out_1, out1 = self.d1(x_pre)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, coverDown[3])
        factor1, factor2 = self.AdaptLayer4(out5, out_4)
        out5 = factor1 * out5 + factor2 * out_4
        out6 = self.u2(out5, coverDown[2])
        factor1, factor2 = self.AdaptLayer3(out6, out_3)
        out6 = factor1 * out6 + factor2 * out_3
        out7 = self.u3(out6, coverDown[1])
        factor1, factor2 = self.AdaptLayer2(out7, out_2)
        out7 = factor1 * out7 + factor2 * out_2
        out8 = self.u4(out7, coverDown[0])
        factor1, factor2 = self.AdaptLayer1(out8, out_1)
        out8 = factor1 * out8 + factor2 * out_1
        out = self.out(out8)

        return out


if __name__ == '__main__':
    import time

    a = torch.rand(16, 16, 128, 128).cuda()
    model = UNet(32).cuda()
    start = time.time()
    model([a, a])
    print(time.time()-start)
