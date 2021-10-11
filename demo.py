# ==================================
# !/usr/bin/python3
# --coding:utf-8--
# Author : time-无产者
# @time : 2021/8/24 10:04
# ==================================
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms

# class Animal:
#     def __init__(self, animal_list):
#         self.animals_name = animal_list
#
#     def __getitem__(self, index):
#         return self.animals_name[index]
#
#
# animals = Animal(["dog","cat","fish"])
# for animal in animals:
#     print(animal)


# 计算数据集的均值和标准差
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    # train_dataset = ImageFolder(root=r'dataset/train', transform=transforms.ToTensor())
    # print(getStat(train_dataset))
    print(transition_dict[1])
