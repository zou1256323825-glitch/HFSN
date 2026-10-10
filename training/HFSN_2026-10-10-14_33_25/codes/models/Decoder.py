import torch.nn as nn
from models.ConvBlock import ConvBlock
from models.SEBlock import SENet


class Down(nn.Module):
    def __init__(self, in_nc, times):
        super(Down, self).__init__()
        self.donwlayers = []
        for i in range(times):
            self.donwlayer = ConvBlock(in_nc, in_nc * 2, 3, 2, 1)
            # self.attention = MultiSpectralAttentionLayer(in_nc*2, 8, 8)
            self.resblock = SENet(in_nc * 2, in_nc * 2, 1)
            # self.jnd = JNDNet(in_nc * 2, in_nc * 2, Wnet(SEP2Embedwater)+SEDecoder_COCO)
            self.donwlayers.append(self.donwlayer)
            self.donwlayers.append(self.resblock)

            in_nc = in_nc*2
        self.down = nn.Sequential(*self.donwlayers)

    def forward(self, x):
        return self.down(x)


class Decoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(in_ch, 32, 3, 1, 1), Down(32, 1), Down(64, 1),
                                  Down(128, 1), Down(256, 1))
        self.out = nn.Sequential(
            ConvBlock(512, 64, 3, 1, 1),
            ConvBlock(64, out_ch, 3, 1, 1),
        )
        # self.fc = nn.Linear(64, 30)

    def forward(self, img):
        f = self.conv(img)
        z = self.out(f)
        out = z.view(f.shape[0], -1)
        # out = self.fc(out)
        return out





















