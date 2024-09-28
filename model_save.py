import torch
import torchvision.models
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1：
torch.save(vgg16, 'vgg16_method1.pth')          # 不仅保存了模型结构，还保存了模型参数

# 保存方式2：(官方推荐，占用空间更小)
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')         # 只保留模型的参数（以字典的形式存储），不保存结构


# 陷阱
class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x


m = module()
torch.save(m, 'test.pth')           # 保存了模型结构和初始化的参数
