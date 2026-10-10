# @Time: 2023/7/18 15:40
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kcore, stride, padding=0, relutype='Relu', neoconv=1, Dconv=False):
        super(ConvBlock, self).__init__()
        self.Dconv = Dconv
        if Dconv:
            conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kcore, stride=stride, padding=padding)
            conv1 = nn.ConvTranspose2d(output_nc, output_nc, kernel_size=kcore, stride=stride, padding=padding)
        else:
            conv = nn.Conv2d(input_nc, output_nc, kernel_size=kcore, stride=stride, padding=padding)
            conv1 = nn.Conv2d(output_nc, output_nc, kernel_size=kcore, stride=stride, padding=padding)
        norm = nn.BatchNorm2d(output_nc)
        relu = None
        if relutype == 'Relu':
            relu = nn.ReLU(True)
        elif relutype == 'Sigmoid':
            relu = nn.Sigmoid()
        elif relutype == 'Softmax':
            relu = nn.Softmax()
        if neoconv == 1:
            model = [conv, norm, relu]
        elif neoconv == 2:
            model = [conv, conv1, norm, relu]
        elif neoconv == 3:
            model = [conv, conv1, conv1, norm, relu]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResBlock1(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(ResBlock1, self).__init__()
        self.stride = stride
        self.model = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            ConvBlock(in_channels, in_channels, 1, 1, 0),
            ConvBlock(in_channels, in_channels, 3, 1, 1),
        )
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        identity = self.conv1(x)
        out = self.model(x)
        # 如果输入和输出的维度不一致，进行投影映射
        if self.stride != 1 or identity.shape[1] != out.shape[1]:
            identity = nn.Conv2d(identity.shape[1], out.shape[1], kernel_size=1, stride=self.stride, bias=False)(identity)
        out = out + identity
        # out = nn.Sigmoid()(out)
        return nn.Sigmoid()(out)


class ResBlock2(nn.Module):
    def __init__(self, in_channels, stride=1):
        self.stride=stride
        super(ResBlock2, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            ConvBlock(in_channels, in_channels*4, 1, 1, 0),
            nn.PixelShuffle(2),
            ConvBlock(in_channels, in_channels, 3, 1, 1),
        )
        self.conv1 = ConvBlock(in_channels, in_channels, 4, 2, 1, Dconv=True)

    def forward(self, x):
        identity = self.conv1(x)
        out = self.model(x)
        if self.stride != 1 or identity.shape[1] != out.shape[1]:
            identity = nn.Conv2d(identity.shape[1], out.shape[1], kernel_size=1, stride=self.stride, bias=False)(
                identity)
        out = out + identity
        # out = nn.Sigmoid()(out)
        return nn.Sigmoid()(out)


class BlockB(nn.Module):
    def __init__(self, in_channels, outchannels, level):
        super(BlockB, self).__init__()
        # self.step1 = ConvBlock(in_channels, in_channels, Wnet(SEP2Embedwater)+SEDecoder_COCO+2*level, Wnet(SEP2Embedwater)+SEDecoder_COCO, 0+level)
        self.step2 = ConvBlock(in_channels, in_channels, 1, 1, 0)
        self.step3 = ConvBlock(in_channels, in_channels, 3, 1, 1)
        self.step4 = ConvBlock(in_channels, in_channels, 5, 1, 2)
        # self.se1 = SE_Block(in_channels*4)
        self.step5 = ConvBlock(in_channels*3, outchannels, 1, 1, 0)
        # self.se2 = SE_Block(in_channels)

    def forward(self, x):
        # step1 = self.step1(x)
        step2 = self.step2(x)
        step3 = self.step3(x)
        step4 = self.step4(x)
        concat = torch.cat((step2, step3, step4), dim=1)
        step5 = self.step5(concat)
        output = step5 + x

        return output


class Dense_level(nn.Module):
    def __init__(self, in_channels, level):
        super(Dense_level, self).__init__()
        self.step1 = ConvBlock(in_channels, in_channels, 7 + 2 * level, 1, 3 + level)
        self.step2 = ConvBlock(in_channels*2, in_channels, 5 + 2 * level, 1, 2 + level)
        self.step3 = ConvBlock(in_channels*3, in_channels, 3 + 2 * level, 1, 1 + level)

    def forward(self, x):
        step1 = self.step1(x)
        step2 = self.step2(torch.cat((step1, x), dim=1))
        step3 = self.step3(torch.cat((step2, step1, x), dim=1))
        return step3


class Conv_BlockB(nn.Module):
    def __init__(self, in_channels, level):
        super(Conv_BlockB, self).__init__()
        self.conv1 = Dense_level(in_channels, level)
        self.conv2 = Dense_level(in_channels, level)
        self.conv_out = MutiConv(in_channels, in_channels, level)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv2(self.conv1(x))
        out = self.conv_out(x1)
        return self.sigmoid(out)


class MutiConv(nn.Module):
    def __init__(self, init_C, out_channels, level, drop=False):
        super(MutiConv, self).__init__()
        self.init_C = init_C//3
        self.conv11 = nn.Sequential(
            ConvBlock(init_C, self.init_C, 1, 1, 0),
            nn.PixelShuffle(2),
            ConvBlock(self.init_C // 4, self.init_C // 4, 5+2*level, 1, 2+level),
            nn.PixelUnshuffle(2),
        )
        self.conv2 = ConvBlock(init_C, self.init_C, 1, 1, 0)
        self.conv22 = nn.Sequential(
            ConvBlock(self.init_C*2, self.init_C*2, 3+2*level, 1, 1+level),
            nn.PixelUnshuffle(2),
                                   )
        self.conv3 = nn.Sequential(
            ConvBlock(init_C, self.init_C, 1, 1, 0),
            nn.PixelUnshuffle(2),
        )
        self.conv33 = nn.Sequential(
            ConvBlock(self.init_C*12, self.init_C*12, 1+2*level, 1, 0+level),
            nn.PixelShuffle(2),

        )
        self.out_conv = ConvBlock(init_C, out_channels, 1, 1, 0)

    def forward(self, x):
        x11 = self.conv11(x)
        x2 = self.conv2(x)
        x22 = self.conv22(torch.cat((x11, x2), dim=1))
        x3 = self.conv3(x)
        x33 = self.conv33(torch.cat((x22, x3), dim=1))
        out = self.out_conv(x33+x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, nf):
        super(DilatedConv, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels, nf, kernel_size=3, padding=1)

        self.res_block1 = ResidualBlock(nf, nf, dilation=16)
        self.res_block2 = ResidualBlock(nf, nf, dilation=8)
        self.res_block3 = ResidualBlock(nf, nf, dilation=4)
        self.res_block4 = ResidualBlock(nf, nf, dilation=2)

        self.final_conv = ConvBlock(nf, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.final_conv(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nf):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf*2, nf, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf*3, nf, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(nf*4, nf, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.conv5 = nn.Conv2d(nf*5, out_channels, 3, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((conv1, x), dim=1))
        conv3 = self.conv3(torch.cat((conv2, conv1, x), dim=1))
        conv4 = self.conv4(torch.cat((conv3, conv2, conv1, x), dim=1))
        conv5 = self.conv5(torch.cat((conv4, conv3, conv2, conv1, x), dim=1))

        return conv5


class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.DB1 = DenseBlock(in_channels, in_channels, in_channels)
        self.DB2 = DenseBlock(in_channels, in_channels, in_channels)
        self.DB3 = DenseBlock(in_channels, in_channels, in_channels)

    def forward(self, x):
        db1 = self.DB1(x)
        db1 = x + db1
        db2 = self.DB2(db1)
        db2 = db1 + db2
        db3 = self.DB3(db2)
        db3 = db2 + db3
        out = db3 + x
        return out


if __name__ == '__main__':
    from torchsummary import summary
    model = Conv_BlockB(32, 4).cuda()
    model.eval()
    out = summary(model, (32, 256, 256), batch_size=4)



















