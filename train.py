import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
train_dataset = torchvision.datasets.CIFAR10('./data', train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)
# 获取数据集长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print('训练数据集的长度为：{}'.format(train_data_size))
print('测试数据集的长度为：{}'.format(test_data_size))

# 加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)

# 搭建神经网络---10分类的网络：规范化的写法是单独放在一个model.py文件中

# 创建网络模型
m = module()

# 创建损失函数：在nn工具箱中
cross = nn.CrossEntropyLoss()

# 定义优化器：传入网络参数 --> torch内置有了
# 1e-2 = 1 x (10)^(-2)
learning_rate = 1e-2
optim = torch.optim.SGD(m.parameters(), lr=learning_rate)

# 设置训练网络的参数 -----------------?????????
# 记录训练的次数
train_times = 0
# 记录测试的次数
test_times = 0
# 训练的轮数
epoch = 10

# 添加tensorboard进行可视化
writer = SummaryWriter('./logs')

for i in range(epoch):
    print('------------------第 {} 轮训练开始-----------------'.format(i+1))
    epoch_loss = 0
    # 训练步骤开始：取数据，调用训练函数
    for data in train_dataloader:
        imgs, targets = data
        output = m(imgs)
        loss = cross(output, targets)
        # 优化器更新参数：先清零，再算梯度，最后更新
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_times = train_times + 1
        if train_times % 100 == 0:      # 使得后台信息更简洁
            print('第 {} 次训练的损失为：{}'.format(train_times, loss.item()))       # 只有一个值时，.item()方法可以将tensor数据类型转化为数字
            writer.add_scalar('train_loss', loss.item(), train_times)

    # 测试步骤开始：用测试数据集进行模型测试
    # with torch.no_grad(): ----> 其中包含的代码块是没有梯度的，不会影响调优
    test_loss = 0
    test_accuracy = 0       # 图像分类问题中常常需要计算正确率
    with torch.no_grad():           # 保证网络模型中没有任何梯度
        for data in test_dataloader:
            imgs, targets = data
            output = m(imgs)
            accuracy = (output.argmax(1) == targets).sum()
            test_accuracy = test_accuracy + accuracy
            loss = cross(output, targets)
            test_loss = test_loss + loss
    print('--------------测试集的损失为：{}---------------'.format(test_loss.item()))
    print('--------------测试集正确率为：{}---------------'.format(test_accuracy/test_data_size))
    writer.add_scalar('test_loss', test_loss, test_times)
    writer.add_scalar('test_accracy', test_accuracy/test_data_size, test_times)
    test_times = test_times + 1

    # 保存每一轮的模型
    torch.save(m, './model/model_{}.pth'.format(i))
    # 保存方式2 ---- torch.save(m.state_dict(), 'model_{}.pth')
    print('模型已保存')

writer.close()

"""
m.train():只对于几个特定的层有作用，Dropout、normal层等
m.eval()的作用:只对于几个特定的层有作用，Dropout、normal层等，=m.train(False)
有这些特定的层时，开始训练或测试之前一定需要写
"""


