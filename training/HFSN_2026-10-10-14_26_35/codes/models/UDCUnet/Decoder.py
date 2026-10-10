import torch
import torch.nn as nn
from models.ConvBlock import ConvBlock
from models.UDCUnet.fcanet import MultiSpectralAttentionLayer
from models.UDCUnet.arch_util import ResidualBlockNoBN
from models.UDCUnet.STN import STN_Net
from models.UDCUnet.JND_Attention import JNDNet


class Down(nn.Module):
    def __init__(self, in_nc, times):
        super(Down, self).__init__()
        self.donwlayers = []
        for i in range(times):
            self.donwlayer = ConvBlock(in_nc, in_nc * 2, 3, 2, 1)
            # self.attention = MultiSpectralAttentionLayer(in_nc*2, 8, 8)
            self.resblock = ResidualBlockNoBN(in_nc * 2)
            # self.jnd = JNDNet(in_nc * 2, in_nc * 2, Wnet(SEP2Embedwater)+SEDecoder_COCO)
            self.donwlayers.append(self.donwlayer)
            self.donwlayers.append(self.resblock)
            # self.donwlayers.append(self.jnd)

            in_nc = in_nc*2
        self.down = nn.Sequential(*self.donwlayers)

    def forward(self, x):
        return self.down(x)


class Decoder(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Decoder, self).__init__()
        self.startlayer = ConvBlock(in_nc, nf, 3, 1, 1)
        self.donwlayer1 = Down(nf, 1)
        self.backlayer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(nf * 2, nf, 3, 1, 1),
        )
        self.donwlayer2 = Down(nf * 2, 1)
        self.backlayer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(nf * 4, nf * 2, 3, 1, 1),
        )
        self.donwlayer3 = Down(nf * 4, 1)
        self.backlayer3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(nf * 8, nf * 4, 3, 1, 1),
        )
        self.donwlayer4 = Down(nf * 8, 1)
        self.backlayer4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(nf * 16, nf * 8, 3, 1, 1),
        )
        self.out = nn.Sequential(
            MultiSpectralAttentionLayer(nf * 16, 8, 8),
            ConvBlock(nf * 16, nf, 1, 1, 0),
            ConvBlock(nf, 1, 3, 1, 1),)
        self.stn = STN_Net()

    def forward(self, x):
        x = self.stn(x)
        head = self.startlayer(x)
        down1 = self.donwlayer1(head)
        down2 = self.donwlayer2(down1)
        down3 = self.donwlayer3(down2)
        down4 = self.donwlayer4(down3)
        down11 = self.donwlayer1(head + self.backlayer1(down1))
        # down11 = self.downFCA1(down11)
        down22 = self.donwlayer2(down11 + self.backlayer2(down2))
        # down22 = self.downFCA2(down22)
        down33 = self.donwlayer3(down22 + self.backlayer3(down3))
        # down33 = self.downFCA3(down33)
        down44 = self.donwlayer4(down33 + self.backlayer4(down4))
        # down44 = self.downFCA4(down44)
        out = self.out(down44)
        message_decoder = out.view(x.shape[0], -1)
        return message_decoder


if __name__ == '__main__':
    import time
    start_time = time.time()
    a = torch.rand(16, 3, 128, 128).cuda()
    model = Decoder().cuda()
    out = model(a)
    print(time.time()-start_time)
    print(out.shape)








