# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 20:12
# @Author  : Jimmy
# @Email   : 2362763425@qq.com
# @File    : mydataset.py

from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    """
    将传入的数据集，转成Dataset类，方面后续转入Dataloader类
    注意定义时传入的images,targets必须为numpy数组
    """

    # 使用__init__()初始化一些需要传入的参数及数据集的调用
    def __init__(self, images, targets):
        self.len = len(images)
        self.image = torch.FloatTensor(images)  # 转换成tensor类型
        self.target = torch.FloatTensor(targets)  # 转换成tensor类型

    def __getitem__(self, index):
        return self.image[index], self.target[index]

    def __len__(self):
        return self.len
