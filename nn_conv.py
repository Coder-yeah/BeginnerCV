import torch
import torch.nn.functional as F     # funtional是torch.nn的组件，是一个包含关系，nn将其进行了封装

# 输入数据的尺寸为[5,5]，是一个二维的张量
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 1],
                      [5, 2, 3, 1, 1],
                      [0, 2, 3, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 卷积运算输入输出要求的尺寸都是四个数字，即卷积运算的输入应该为一个四维矩阵
# 例如，在输入中，四个数字分别表示：batchsize、输入通道数、高、宽
# 因此，需要进行形状变换。根据图片特征，可知样本数和通道数都应该为1
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

# weight参数实际上就是卷积核
output = F.conv2d(input, kernel, stride=1)          #进行卷积运算
print(output)

# dilation用于空洞卷积（也就是每个卷积格子之间相差多少），group用于分组卷积
