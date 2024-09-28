import torch
from torch import nn


# 定义网络
class Mymodule(nn.Module):      # 需要继承Module类作为模板，之后重写方法
    def __init__(self):        # self必备，指这个类的
        super(Mymodule, self).__init__()        # 对模板调用父类的初始化函数

    def forward(self, input):       # 该函数定义了从输入到输出的计算过程
        output = input + 1
        return output


module = Mymodule()     # 创建模型实例时，首先调用model的初始化函数super进行参数初始化，完成后才进行下一句主程序
x = torch.tensor(1.0)
output = module.forward(x)          # 调用方法之后，才开始执行forward函数
print(output)
