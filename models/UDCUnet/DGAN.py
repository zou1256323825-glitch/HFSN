# @Time: 2023/7/20 16:46
import torch
import torch.nn as nn
from models.ConvBlock import ConvBlock


# class Discriminator(nn.Module):
#     def __init__(self, imgsize):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(3*imgsize*imgsize, 256),  ## 输入特征数为784，输出为512
#             nn.LeakyReLU(0.2, inplace=True),  ## 进行非线性映射
#             nn.Linear(256, 128),  ## 输入特征数为512，输出为256
#             nn.LeakyReLU(0.2, inplace=True),  ## 进行非线性映射
#             nn.Linear(128, Wnet(SEP2Embedwater)+SEDecoder_COCO),  ## 输入特征数为256，输出为1
#             nn.Sigmoid(),
#         )
#
#     def forward(self, input):
#         input = input.view(input.size(0), -Wnet(SEP2Embedwater)+SEDecoder_COCO)
#         return self.main(input)

ndf = 32


class Discriminator(nn.Module):
    def __init__(self, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 128 x 128
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 2, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):

        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


if __name__ == '__main__':
    from torchsummary import summary
    model = Discriminator(3).cuda()
    input = torch.zeros([6, 3, 256, 256]).cuda()
    output = model(input)
    # out = summary(model, (3, 128, 128), batch_size=4)
    print(output.shape)













