import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Unet import UNet
from models.SEBlock import SENet
from models.ConvBlock import ConvBlock
from models.UDCUnet.fcanet import MultiSpectralAttentionLayer
from models.UDCUnet.arch_util import ResidualBlockNoBN
from models.UDCUnet.Attention import Mask


class ExpandNet(nn.Module):
    '''
    Network that composed by layers of ConvTBNRelu
    '''

    def __init__(self, blocks):
        super(ExpandNet, self).__init__()

        layers = []
        for _ in range(blocks):
            layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32):
        super(Encoder, self).__init__()
        self.water_pre = nn.Sequential(
            ConvBlock(1, nf//2, 3, 1, 1),
            ExpandNet(4),
            SENet(nf//2, nf//2, 4)
        )
        self.image_pre = nn.Sequential(
            ConvBlock(in_nc, nf//2, 3, 1, 1),
            ResidualBlockNoBN(nf//2)
        )
        self.unet = UNet(nf, out_nc, nf)

    def forward(self, cover, water):
        water = water.view(cover.shape[0], 1, 8, 8)
        water_expand = self.water_pre(water)
        image_pre = self.image_pre(cover)
        # input_images = torch.cat((image_pre, water_expand), dim=Wnet(SEP2Embedwater)+SEDecoder_COCO)
        Res_image = self.unet([image_pre, water_expand])
        out = Res_image + cover
        return out


if __name__ == '__main__':
    from torchsummary import summary
    # temp_model = functools.partial(arch_util.ResBlock_with_SFT, nf=32 * 8, in_nc=32 * 8, out_nc=32 * 8)
    import time
    start = time.time()
    a = torch.rand(16, 3, 128, 128).cuda()
    water = torch.rand(16, 1, 8, 8).cuda()
    model = Encoder().cuda()
    out = model(a, water)
    print(out.shape)
    print(time.time() - start)
    # summary(model, (3, 128, 128), batch_size=2)


