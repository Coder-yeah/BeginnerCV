import torch.optim
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=32)


class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)        # padding和stride通过公式计算出来->kernelsize为5时，padding为2保持尺寸
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

        self.model1 = Sequential(           # nn工具对象的集合，输入会在sequential组合里面顺序传播
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = CrossEntropyLoss()
m = module()
optim = torch.optim.SGD(m.parameters(), lr=0.01)        # 定义优化器，params模型可调节的参数，lr学习速率
for epoch in range(20):                 # 对数据集进行20次学习
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data    # targets的尺寸为N
        output = m(imgs)
        # print(output)         # 最终会输出每个类别的概率列表，尺寸为N*C
        result_loss = loss(output, targets)
        # print(result_loss)      # 计算每个batchsize的平均损失

        # 反向传播(grad)根据loss求出各个参数的梯度，之后优化器通过梯度反馈对神经网络的参数进行更新
        optim.zero_grad()       # 首先对梯度进行清零
        result_loss.backward()          # 调用损失函数中的反向传播方法求梯度
        optim.step()            # 优化器开始进行参数更新
        running_loss = running_loss + result_loss
    print(epoch, running_loss)
