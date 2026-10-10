import torch
import torch.nn as nn
from models.UDCUnet.fcanet import MultiSpectralAttentionLayer


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 2, 4, 8]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        out = torch.cat((x1, x2, x3, x4), dim=1)
        return out

class Mask(nn.Module):
    def conv_1(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=7, padding=3)

    def conv_2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 2, 4, 8]):
        super(Mask, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        self.first_layer = nn.Sequential(
            self.conv_2(3, inplanes)

        )
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=Wnet(SEP2Embedwater)+SEDecoder_COCO, bias=False)
        self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # self.att = MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes], reduction=reduction,
        #                                        freq_sel_method='top16')

        self.last_layer = nn.Sequential(
            nn.BatchNorm2d(planes*4),
            nn.ReLU(inplace=True),
            self.conv_2(planes*4, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            self.conv_2(planes,planes),
            nn.Softmax(dim=1)
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.first_layer(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.att(out)

        out = self.last_layer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)

        return out

if __name__ == '__main__':
    a = torch.rand(2, 3, 256, 256)
    model =  Mask(16, 32)
    print(model(a).shape)