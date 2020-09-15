import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
# port torchsummary as summary
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from model import *
import numpy as np
import cv2
from PIL import Image

binary_colormap = [0, 255]

tusimple_path = '/home/gong/Datasets/TuSimple'
our_batch_size = 1
num_classes = 2
epochs = 10


# 读取所有数据的地址
def read_images(path, is_train=True):
    file = path + ('/train/train.txt' if is_train else '/test/test.txt')
    with open(file) as f:
        imgs = f.read().split()

    datas = []
    labels = []
    for idx, img in enumerate(imgs):
        if idx % 3 == 0:
            datas.append(img)
        if idx % 3 == 1:
            labels.append(img)

    return datas, labels


# 对单张图像对应的 原始图+标签图，返回各自的tensor
def transform(img, label):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)

    img2tensor = torch.from_numpy(img.transpose((2, 0, 1)))
    img2tensor = img2tensor.float().div(255)

    # 使标签中的不同颜色对应 数字0和1
    for i in range(2):
        label[label == binary_colormap[i]] = i
    label[label == 224] = 0
    label2tensor = torch.from_numpy(label)

    return img2tensor, label2tensor


#   自己写一个继承了Dataset类，负责将数据处理成”适配DataLoader函数“的类
#   原本是 先把所有img的tensor存入一个list， 再把所有label的tensor存入另一个list，
# 然后把这两个list作为TensorDataset的输入,最后把TensorDataset的输出作为DataLoader的输入
#   https://blog.csdn.net/zw__chen/article/details/82806900


class DealDataset(Dataset):
    def __init__(self, train):
        super(DealDataset, self).__init__()
        self.img_path, self.label_path = read_images(tusimple_path, train)
        self.len = len(self.img_path)

    # 看起来很简单，写好下面三行后，似乎继承的类里有类似for循环一样的操作？总共执行了self.len次？
    def __getitem__(self, idx):
        img, label = transform(self.img_path[idx], self.label_path[idx])
        # print(label.size())
        return img, label

    def __len__(self):
        return self.len


# 设备
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.enabled = False  # 死马当做活马医 # 试验后，发现没用，应该是label的值中有不是0~20的原因，导致criterion相关函数报错

# 数据
train_data = DealDataset(train=True)
val_data = DealDataset(train=False)
train_loader = DataLoader(dataset=train_data, batch_size=our_batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=our_batch_size)

# 模型
our_model = encoder_decoder(num=num_classes)  # 训练时
# our_model = FCN_VGG() # 测试时(以下三行)
# our_model.load_state_dict(torch.load('./model/fcn_vgg16_1.pkl'))
# model.eval()

# our_model.to(device)  # 指定模型加载于GPU上
our_model = our_model.to(device)

# config, 配置
learning_rate = 0.01
criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(our_model.parameters(), lr=learning_rate, weight_decay=5*1e-4, momentum=0.9)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = 0.01 * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:  # 参考https://blog.csdn.net/bc521bc/article/details/85864555
        print('lr : ', param_group['lr'])
        param_group['lr'] = lr
        print('reduce to ', param_group['lr'])


# 单个epoch的训练
def train(our_model, device, train_loader, optimizer, epoch):
    our_model.train()
    adjust_learning_rate(optimizer, epoch)
    for batch_idx, (data, label) in enumerate(train_loader):
        # data.to(device)
        # label.to(device)
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()

        output = our_model(data)
        # print(output.size())
        output = F.log_softmax(output, dim=1)  # dim=1 代表对输入tensor的每一行做“softmax，再做log”
        # print(output.size())
        loss = criterion(output, label.long())
        # loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        if batch_idx % 30 == 0:
            print('train {} epoch : {}/{} \t loss : {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))


for epoch in range(epochs):
    print(epoch)
    train(our_model, device, train_loader, optimizer, epoch)
    torch.save({
        "epoch": epoch,
        "model_state_dict": our_model.state_dict(),
        "loss_state_dict": criterion.state_dict(),   # 这个没什么意义吧！！！
        "optimizer_state_dict": optimizer.state_dict()
        }, './checkpoint/encoder_decoder_1.pkl')
