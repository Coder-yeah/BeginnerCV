from torch.utils.tensorboard import SummaryWriter       # 导入类
from PIL import Image
import numpy as np      # 利用numpy.array()对PIL图片进行转换


path = "datasets/三沙/5.png"
img_PIL = Image.open(path)
img_array = np.array(img_PIL)

writer = SummaryWriter('logs')      # 创建实例
# 绘制图像，常用三个参数：名字、图像（torch.tensor\np.array\string形式的）、步骤，注意np的格式顺序(可能需要指定shape中每一个数字的含义）
writer.add_image('train', img_array, 1, dataformats='HWC')
# y = x
for i in range(100):
    writer.add_scalar('y=2x', 2*i, i)     # 绘制图表时的数值,常用三个参数：表格名、y值、x值（注意：不换名字时，会产生重叠）
# >>tensorboard --logdir=logs --port=6007打开tensorboard文件，默认端口是6006
writer.close()
