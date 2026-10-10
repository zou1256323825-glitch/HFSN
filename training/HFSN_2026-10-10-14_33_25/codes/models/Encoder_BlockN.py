import functools
import numpy as np
import torch
import torch.nn as nn
from .DoublePathUnetV1_BlockN import UNet, PositiveEvidence, NegativeEvidence, AdaptiveScoringMask
from .ConvBlock import ConvBlock


class ExpandNet(nn.Module):
    '''
    Network that composed by layers of ConvTBNRelu
    '''

    def __init__(self, blocks, nf=32):
        super(ExpandNet, self).__init__()

        layers = []
        for _ in range(blocks):
            layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, Blockn=8, factor=1):
        super(Encoder, self).__init__()
        self.factor = factor

        self.water_pre = nn.Sequential(
            ConvBlock(1, nf//2, 3, 1, 1),
            ExpandNet(4),
        )
        self.image_pre = nn.Sequential(
            ConvBlock(in_nc, nf//2, 3, 1, 1),
        )
        self.unet = UNet(nf, out_nc, nf, Blockn)
        self.pos = PositiveEvidence()
        self.neg = NegativeEvidence()
        self.asm = AdaptiveScoringMask(3, 1)

    def forward(self, cover, water):
        length = int(np.sqrt(water.shape[-1]))
        water = water.view(cover.shape[0], 1, length, length)
        water = self.water_pre(water)
        image_pre = self.image_pre(cover)
        Res_image = self.unet(torch.cat((image_pre, water), dim=1))
        Pmask = self.asm.mask_gen(cover) * Res_image.detach()
        out = Res_image * self.factor + cover
        P = self.pos(out.detach(), cover.detach())
        N = self.neg(Pmask)
        Fout = self.asm(cover, out, P, N)
        # Fout = self.asm.mask_gen(cover) * Res_image + cover
        return Fout


if __name__ == '__main__':
    from torchsummary import summary
    # temp_model = functools.partial(arch_util.ResBlock_with_SFT, nf=32 * 8, in_nc=32 * 8, out_nc=32 * 8)
    import time

    start = time.time()
    a = torch.rand(16, 3, 128, 128).cuda()
    water = torch.rand(16, 64).cuda()
    model = Encoder().cuda()
    # from thop import profile
    # flopsd, paramd = profile(model, inputs=(a, water))
    # print('FLOPs = ' + str(flopsd / 1000 ** 3) + 'G')
    # print('Params = ' + str(paramd / 1000 ** 2) + 'M')
    out = model(a, water)
    # print(out.shape)
    # print(time.time() - start)
    # summary(model, (3, 128, 128), batch_size=2)
