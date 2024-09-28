# 将训练好的模型用到实际图片中
import torch
import torchvision.transforms
from PIL import Image
from model import *

img_path = './datasets/dog.png'
image = Image.open(img_path)        # 以PIL形式导入，后续转换成tensor格式   ----RGBA
image = image.convert('RGB')        # png格式的图片有四个通道，有一个是透明度通道，需要转换成颜色的三通道    ----RGB
print(image)

# 根据模型调整输入图片尺寸以及格式
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

# 加载训练好的网络模型
# 如果要将在gpu上运行的模型拿到cpu去测试，需要添加映射地址的参数 ------ 不同环境中的模型需要映射
# m = torch.load('./model/model_6.pth', map_location=torch.device('cpu'))
m = torch.load('./model/model_6.pth')
print(m)

# 验证图片输入
image = torch.reshape(image, (1, 3, 32, 32))
m.eval()        # 将模型转换为测试类型
with torch.no_grad():           # 可以节省参数空间
    output = m(image)
print(output)
print(output.argmax(1))
