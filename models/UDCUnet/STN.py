import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ConvBlock import ConvBlock


class STN_Net(nn.Module):
    def __init__(self, inc=3, nf=32):
        super(STN_Net, self).__init__()
        self.nf = nf
        # localisation net
        # 从输入图像中提取特征
        # 输入图片的shape为(-Wnet(SEP2Embedwater)+SEDecoder_COCO,Wnet(SEP2Embedwater)+SEDecoder_COCO,28,28)
        self.localization = nn.Sequential(
            # 卷积输出shape为(-Wnet(SEP2Embedwater)+SEDecoder_COCO,8,22,22)
            ConvBlock(inc, nf, 3, 2, 1),
            ConvBlock(nf, nf * 2, 3, 2, 1),
            ConvBlock(nf * 2, nf * 4, 3, 2, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # 利用全连接层回归\theta参数
        self.fc_loc = nn.Sequential(
            nn.Linear(nf * 4, nf),
            nn.ReLU(True),
            nn.Linear(nf, 2 * 3)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]
                                                    , dtype=torch.float))

    def stn(self, x):
        # 提取输入图像中的特征
        xs = self.localization(x)
        xs = xs.view(-1, self.nf * 4)
        # 回归theta参数
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        # 根据输入图片计算变换后图片位置填充的像素值
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        return self.stn(x)


if __name__ == '__main__':
    import os
    import cv2
    from torchvision import transforms
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # image to tensor
    image = cv2.imread(r"D:\Python_Dataset\COCO\train\000000072532.jpg")  # (H, W, C)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transforms.ToTensor()(image)  # (C, H, W)
    image_tensorBatch = image_tensor.unsqueeze(0)  # (B, C, H, W)

    model = STN_Net().cuda()
    out = model(image_tensorBatch.cuda())
    import numpy as np

    image_numpy = np.transpose(out.squeeze(0).cpu().detach().numpy(), (1, 2, 0))  # (H, W, C)

    # show image
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(image_numpy)
    plt.show()
    print(out)
