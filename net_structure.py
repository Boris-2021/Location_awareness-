# ==================================
# !/usr/bin/python3
# --coding:utf-8--
# Author : time-无产者
# @time : 2021/8/24 10:04
# ==================================

import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb


# 网络结构
# 基本定义__init__, 前向传播forward
class LeNet(nn.Module):
    def __init__(self, classes):
        # 初始化函数中，定义每层
        super(LeNet, self).__init__()
        # 输入的通道数3，输出的通道数6，卷积核的宽和高都是5
        # 卷积核：6*3*5*5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积核 16*6*5*5
        self.fc1 = nn.Linear(16*13*13, 120)  # 全连接
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # X: B*C*H*W
        # X: 1*3*64*64
        out = F.relu(self.conv1(x))  # 1*6*64*64
        out = F.max_pool2d(out, 2)  # 核的大小2*2； 1*6*60*60
        out = F.relu(self.conv2(out))  # 1* 16*30*30
        out = F.max_pool2d(out, 2)  # 核的大小2*2；1*16*26*26
        out = out.view(out.size(0), -1)  # 展平(1, 16*13*13)==(1, 2704)
        out = F.relu(self.fc1(out))  # full connect  (1, 120)
        out = F.relu(self.fc2(out))  # (1, 84)
        out = self.fc3(out)  # (1, 类别数)
        return out


if __name__ == '__main__':
    model = LeNet(classes=5)
    img = torch.randn(1, 3, 64, 64)
    print(model(img))
