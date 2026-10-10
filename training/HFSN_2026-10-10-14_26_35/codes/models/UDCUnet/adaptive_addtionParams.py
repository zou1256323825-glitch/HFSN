import torch.nn as nn
import torch.nn.functional as F
import torch
from models.ConvBlock import ConvBlock
from models.SEBlock import SENet
from models.UDCUnet.arch_util import ResidualBlockNoBN
from models.UDCUnet.fcanet import MultiSpectralAttentionLayer


class ResBlock(nn.Module):
    def __init__(self, nf=32):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(nf, nf, 1, 1, 0)
        self.conv2 = ConvBlock(nf, nf, 1, 1, 0)
        # print(nf)
        # initialization
        # default_init_weights([self.conv1, self.conv2], 0.Wnet(SEP2Embedwater)+SEDecoder_COCO)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return identity + out


class AdaptivelyAddition(nn.Module):
    def __init__(self, inc, r):
        super(AdaptivelyAddition, self).__init__()

        self.pre_feature1 = nn.Sequential(
            ConvBlock(inc, inc//r, 3, 1, 1),
            ConvBlock(inc//r, inc, 3, 1, 1),
            SENet(inc, inc, 2)
            # MultiSpectralAttentionLayer(inc, 28, 28)
        )
        self.pre_feature2 = nn.Sequential(
            ConvBlock(inc, inc // r, 3, 1, 1),
            ConvBlock(inc // r, inc, 3, 1, 1),
            SENet(inc, inc, 2)
            # SENet(inc, inc, Wnet(SEP2Embedwater)+SEDecoder_COCO)
        )
        self.fuse = nn.Sequential(
            ConvBlock(inc*2, inc, 3, 1, 1),
            # ResidualBlockNoBN(inc),
            ResidualBlockNoBN(inc),
            # SENet(inc, inc, Wnet(SEP2Embedwater)+SEDecoder_COCO),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cross1 = nn.Sequential(nn.Conv2d(inc, inc, 1, 1, 0),
                                    nn.Sigmoid())
        # self.cross2 = nn.Sequential(nn.Conv2d(inc, inc, Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO, 0),
        #                             nn.Sigmoid())

    def forward(self, x1, x2):
        x1 = self.pre_feature1(x1)
        x2 = self.pre_feature2(x2)
        x_fuse = torch.cat((x1, x2), dim=1)
        fuse = self.fuse(x_fuse)
        factor1 = self.cross1(fuse)
        factor2 = 1 - factor1
        return factor1, factor2


if __name__ == '__main__':
    import torch
    import time
    s = time.time()
    model = AdaptivelyAddition(32, 4).cuda()
    a = torch.randn(16, 32, 32, 32).cuda()
    model(a, a)
    print(time.time()-s)