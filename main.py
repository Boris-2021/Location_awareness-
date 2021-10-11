# ==================================
# !/usr/bin/python3
# --coding:utf-8--
# Author : time-无产者
# @time : 2021/8/24 10:04
# ==================================
import numpy as np
from data_loader import RMB_dataset
from torch.utils.data import DataLoader  # 数据分批
from torchvision import transforms
from net_structure import LeNet
import torch
import torch.nn as nn
from torch.optim import SGD, LBFGS, Adam
import matplotlib.pyplot as plt
# 设置字体
plt.rcParams["font.sans-serif"] = "SimHei"
# 默认可以显示负号，增加字体显示后。需对负号正常显示进行设置
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -----------------------------------------超参数--------------------------------
max_epochs = 30
LR = 0.01
Batch_size = 20
Image_size = 64

# ---------------------------------------数据模块---------------------------------
# 计算数据集， 均值的标准差做统计
# [0.47238073, 0.456102, 0.42552894], [0.21099155, 0.22330602, 0.23665123]

norm_mean = [0.472, 0.456, 0.425]
norm_std = [0.211, 0.223, 0.237]

# 仅在训练集中增加大量的图像处理，灰度处理
trans_train = transforms.Compose([
    transforms.Resize((Image_size, Image_size,)),  # 缩放图像
    transforms.RandomGrayscale(p=0.9),  # 90%的数据灰度化
    transforms.ToTensor(),  # 将图像转换为tensor, 除255
    transforms.Normalize(norm_mean, norm_std)  # 归一化
])

# 针对训练集
train_path = "dataset/train"
train_data = RMB_dataset(path=train_path, transform=trans_train)

train_loader = DataLoader(
    dataset=train_data,  # 数据类的对象
    batch_size=Batch_size,
    shuffle=True,
    # num_workers=1
)

# 验证集操作：

# -----------------------------------选择/设计网络结构-----------------------------

model = LeNet(classes=5).to(device)

# -----------------------------------选择/设计损失函数-----------------------------

loss_fun = nn.CrossEntropyLoss()

# --------------------------------------优化器------------------------------------

optimizer = Adam(model.parameters(), lr=LR)

# -------------------------------------训练过程-----------------------------------
acc_rate_list = []
loss_list = []
for epoch in range(max_epochs):
    acc = 0  # 正确的数目
    batch_loss = []
    for batch in train_loader:
        # batch 由 __getitem__的返回值决定
        # 对每一个批次进行遍历
        # img [20, 3, 320, 320] 20个样本，3 个通道 64*64

        img, label = batch
        img, label = img.to(device), label.to(device)
        # print(img)
        output = model(img)

        # 计算损失值
        # 交叉熵损失cost
        # 参数1：输出结果；参数2：真实结果
        loss_val = loss_fun(output, label)
        # print(loss_val.data.cpu().numpy())
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # 获取预测结果的类别
        # print(output)
        yy, y_pred = torch.max(output, dim=1)
        # print(y_pred, '==>', label)
        acc_num = (y_pred == label).sum().cpu().numpy()
        acc += acc_num
        batch_loss.append(loss_val.data.cpu().numpy())
        print('loss值：', batch_loss[-1], '正确数：', acc_num)
    acc_rate = acc*100/(len(train_loader)*Batch_size)
    loss_batch = np.mean(batch_loss)
    acc_rate_list.append(acc_rate)
    loss_list.append(loss_batch)
    print("第{}次迭代训练集的准确率{:.2f}%, loss值{:.2f}".format(epoch, acc_rate, loss_batch))

    if len(acc_rate_list) > 2 and acc_rate > acc_rate_list[-2]:
        # 保存模型
        torch.save(model.state_dict(), 'model/location.pth')

print('损失的变化量', loss_list)
print('准确度的变化', acc_rate_list)