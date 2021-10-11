# ==================================
# !/usr/bin/python3
# --coding:utf-8--
# Author : time-无产者
# @time : 2021/8/24 10:04
# ==================================
import torch
from net_structure import LeNet
from PIL import Image
from torchvision.transforms import functional as F
import cv2
import matplotlib.pyplot as plt

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置字体
plt.rcParams["font.sans-serif"] = "SimHei"
# 默认可以显示负号，增加字体显示后。需对负号正常显示进行设置
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LeNet(classes=5).to(device)
model_path = 'model/location.pth'

# 加载模型
state_dict = torch.load(model_path)  # 获取对应的参数
model.load_state_dict(state_dict)  # 将对应的参数 放入 对应的 卷积、全连接层中


def predicted(img_name):
    img = Image.open(img_name).convert('RGB')
    norm_mean = [0.472, 0.456, 0.425]
    norm_std = [0.211, 0.223, 0.237]

    img = F.resize(img, size=(64, 64))  # 缩放
    img = F.to_tensor(img).to(device)
    img = F.normalize(img, norm_mean, norm_std)
    # print(img)
    # print(img.shape)
    # 将图片扩展为四维

    img = img.expand(1, 3, 64, 64)
    # print(img)
    # print(img.shape)

    output = model(img)
    _, y_pred = torch.max(output, dim=1)
    y_pred = y_pred.data.cpu().numpy()[0]
    transition_dict = {0: '办公室B区门前廊道', 1: '会议室门前廊道', 2: '男厕所门前', 3: '正门口前台前廊道', 4: '资料室门前廊道'}
    pred_location = transition_dict[y_pred]  # 转化为位置信息

    org_img = Image.open(img_name)
    plt.imshow(org_img)
    plt.title(pred_location)
    plt.show()


if __name__ == '__main__':
    predict_path = 'dataset/imgs/微信图片_20211009170135.jpg'
    predicted(predict_path)