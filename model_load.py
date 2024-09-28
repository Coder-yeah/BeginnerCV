import torch
import torchvision.models
from model_save import *            # 解决方法2
from torch import nn


# 加载方式1->对应保存方式1
model1 = torch.load('vgg16_method1.pth')
# print(model)        # 不仅加载了模型结构，也加载了模型参数


# 加载方式2 -> 对应保存方式2
model2 = torch.load('vgg16_method2.pth')
# print(model)        # 返回的是模型参数的字典形式
vgg16 = torchvision.models.vgg16(pretrained=False)      # 加载模型
vgg16.load_state_dict(model2)        # 填充模型参数
# print(vgg16)


# 陷阱
# 解决方法1：加入定义模型（类）的代码，先声明模型结构
# 解决方法2：将自己的模型代码打包，直接使用import导入

# class module(nn.Module):
#     def __init__(self):
#         super(module, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x


model3 = torch.load('test.pth')
print(model3)           # 直接加载自己的模型时会报错
