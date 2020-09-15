import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# input_size: 1280*720

def block(in_ch, out_ch, is_first):
    temp_net = []

    if not is_first:
        temp_net.append(nn.MaxPool2d(2, 2))

    temp_net.append(nn.Conv2d(in_ch, in_ch, 3, 1, 1, dilation=1, groups=in_ch))
    # temp_net.append(nn.BatchNorm2d(in_ch, affine=True))
    temp_net.append(nn.ReLU())
    temp_net.append(nn.Conv2d(in_ch, out_ch, 1, 1, 0, dilation=1, groups=1))  # 不知道在这里改变通道数行不行
    # temp_net.append(nn.BatchNorm2d(out_ch, affine=True))
    temp_net.append(nn.ReLU())

    temp_net.append(nn.Conv2d(out_ch, out_ch, 3, 1, 2, dilation=2, groups=out_ch))  # 这样卷积后尺寸不变
    # temp_net.append(nn.BatchNorm2d(out_ch, affine=True))
    temp_net.append(nn.ReLU())
    temp_net.append(nn.Conv2d(out_ch, out_ch, 1, 1, 0, dilation=1, groups=1))
    # temp_net.append(nn.BatchNorm2d(out_ch, affine=True))
    temp_net.append(nn.ReLU())

    temp_net.append(nn.Conv2d(out_ch, out_ch, 3, 1, 4, dilation=4, groups=out_ch))  # 这样卷积后尺寸不变
    # temp_net.append(nn.BatchNorm2d(out_ch, affine=True))
    temp_net.append(nn.ReLU())
    temp_net.append(nn.Conv2d(out_ch, out_ch, 1, 1, 0, dilation=1, groups=1))
    # temp_net.append(nn.BatchNorm2d(out_ch, affine=True))
    temp_net.append(nn.ReLU())

    return temp_net


class encoder_decoder(nn.Module):
    def __init__(self, num):
        super(encoder_decoder, self).__init__()

        block1 = block(3, 64, True)  # 输入尺寸：1*3*1280*720  输出尺寸：1*64*1280*720
        block2 = block(64, 128, False)  # 输入尺寸：1*64*1280*720  输出尺寸：1*128*640*360
        block3 = block(128, 256, False)  # 输出尺寸：1*256*320*180
        block4 = block(256, 512, False)  # 输出尺寸：1*512*160*90
        block5 = block(512, 4096, False)  # 输出尺寸：1*4096*80*45
        self.stage1 = nn.Sequential(*block1)
        self.stage2 = nn.Sequential(*block2)
        self.stage3 = nn.Sequential(*block3)
        self.stage4 = nn.Sequential(*block4)
        self.stage5 = nn.Sequential(*block5)

        self.stage6 = nn.PixelShuffle(4)  # 输出尺寸：1*(4096/16)*(80*4)*(45*4)
        self.stage7 = nn.Conv2d(512, 1024, 1, 1, 0)  # 输入尺寸：1*(256+256)*(80*4)*(45*4)  输出尺寸：1*1024*(80*4)*(45*4)
        self.stage8 = nn.PixelShuffle(4)  # 输入尺寸：1*1*1024*(80*4)*(45*4)  输出尺寸：1*64*1280*720
        self.stage9 = nn.Conv2d(128, num, 1, 1, 0)  # 二分类问题

    def forward(self, x):
        x = self.stage1(x)
        s1 = x
        x = self.stage2(x)
        x = self.stage3(x)
        s3 = x
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.stage6(x)
        x = self.stage7(torch.cat((s3, x), dim=1))  # [1, 256, 320, 90], dim=1表明选择四个维度里的1号位，其他还有0、2、3号位
        x = self.stage8(x)
        x = self.stage9(torch.cat((s1, x), dim=1))

        return x










