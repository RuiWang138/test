import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

pretrain_model = models.vgg16(pretrained=True)  # 可以考虑vgg16_bn，感觉更有效果


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class FCN_VGG(nn.Module):
    def __init__(self, num):
        super(FCN_VGG, self).__init__()
        net1 = list(pretrain_model.children())[0]
        self.stage1 = nn.Sequential(*net1[0:16])
        self.stage2 = nn.Sequential(*net1[17:24])
        temp_net = []
        for i in range(24, 31):
            temp_net.append(net1[i])
        temp_net.append(nn.Conv2d(512, 4096, 1, 1, 0))
        temp_net.append(nn.Conv2d(4096, 4096, 1, 1, 0))
        self.stage3 = nn.Sequential(*temp_net)

        self.score1 = nn.Conv2d(256, num, 1, 1, 0)
        self.score2 = nn.Conv2d(512, num, 1, 1, 0)
        self.score3 = nn.Conv2d(4096, num, 1, 1, 0)

        self.up_sample1 = nn.ConvTranspose2d(num, num, 8, 4, 2)
        self.up_sample1.weight.data = bilinear_kernel(num, num, 8)
        self.up_sample2 = nn.ConvTranspose2d(num, num, 4, 2, 1)
        self.up_sample2.weight.data = bilinear_kernel(num, num, 4)
        self.up_sample3 = nn.ConvTranspose2d(num, num, 4, 2, 1)
        self.up_sample3.weight.data = bilinear_kernel(num, num, 4)

    def forward(self, x):
        x = self.stage1(x)
        s1 = self.score1(x)
        x = self.stage2(x)
        s2 = self.score2(x)
        x = self.stage3(x)
        s3 = self.score3(x)
        # print(s1.size())
        # print(s2.size())
        # print(s3.size())

        # s3 = self.up_sample3(s3)
        # s2 = self.up_sample2(s2+s3)
        # s1 = self.up_sample1(s1+s2)

        s3 = self.up_sample3(s3)
        s2 = s2+s3
        s2 = self.up_sample2(s2)
        s1 = s1+s2
        s1 = self.up_sample1(s1)

        return s1
