# -*- coding: utf-8 -*-
# @Time    : 2022/12/24 22:08
# @Author  : Jimmy
# @Email   : 2362763425@qq.com
# @File    : load_data.py

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from mydataset import MyDataset
from torch.utils.data import DataLoader


def load_data(batch_size, test_size):
    with open("chn_mnist", "rb") as f:
        data = pickle.load(f)
    images = data["images"]
    images = np.stack((images,) * 1, axis=1)  # 加一维
    target = data["targets"].astype(np.int32)  # 转换类型
    targets = []
    for i in target:  # 直接训练会造成数组越界的报错，因此进行转换
        if i <= 10:
            i = i
        elif i == 100:
            i = 11
        elif i == 1000:
            i = 12
        elif i == 10000:
            i = 13
        elif i == 100000000:
            i = 14
        targets.append(i)
    targets = np.array(targets)

    x_train, x_test, y_train, y_test = train_test_split(images, targets, test_size=test_size)  # 将数据随机分为测试集和训练集

    train_dataset = MyDataset(x_train, y_train)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataset = MyDataset(x_test, y_test)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size)

    return train_dataloader, test_dataloader
