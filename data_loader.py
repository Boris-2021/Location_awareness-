# ==================================
# !/usr/bin/python3
# --coding:utf-8--
# Author : time-无产者
# @time : 2021/8/24 10:04
# ==================================


from torch.utils.data import Dataset
import os
from PIL import Image

"""
自定义的数据类 都必须 继承 于Dataset
__getitem__ 和 __len__ 是必须写的. 通常为规范写__init__函数
"""


class RMB_dataset(Dataset):
    def __init__(self, path="", transform=None):
        # 输入：图像的存储位置
        # 负责：根据指定的路径 将训练集 or 测试集 or 验证集 组织成如下形式
        # train_data = [[图像名称1, 类别1],[图像名称2, 类别2],....[图像名称n, 类别n]]
        self.img_info = []
        for root, dir, files in os.walk(path):
            for file in files:
                file_name = os.path.join(root, file)
                label = int(file_name.split('\\')[-2])
                # print(file_name)
                # 0: '办公室B区门前廊道', 1: '会议室门前廊道', 2: '男厕所门前', 3: '正门口前台前廊道', 4: '资料室门前廊道'
                self.img_info.append([file_name, label])
        self.transfrom = transform

    # def __getitem__(self, item):

    def __getitem__(self, index):
        # 逐张读取图像
        # 之后进行图像处理：旋转、缩放、模糊，翻转
        # index 是个下标。范围【0,训练集的长度）
        img_name, label = self.img_info[index]  # 读图像

        img = Image.open(img_name)  # image mode=RGB size=3468x4624

        if self.transfrom is not None:
            img = self.transfrom(img)

        return img, label

    def __len__(self):
        # 返回值是整个数据集的长度
        return len(self.img_info)


if __name__ == '__main__':
    train_path = "dataset/train"
    train_dataset = RMB_dataset(path=train_path)
    # for i in train_dataset:
    #     print(i)
    # # print(img_info)
    # print(len(train_dataset))
