# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 16:33
# @Author  : Jimmy
# @Email   : 2362763425@qq.com
# @File    : net.py


import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(  # 输入1*64*64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32*64*64
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 32*32*32
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*32*32
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=3),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 64*16*16
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*16*16
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*16*16
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*16*16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 256*7*7
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(),  # 默认0.5的概率归零，防止过拟合
            nn.Linear(256 * 7 * 7, 1024),  # 全连接1
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(),  # 默认0.5的概率归零，防止过拟合
            nn.Linear(1024, 512),  # 全连接2
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 15)  # 全连接3
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
