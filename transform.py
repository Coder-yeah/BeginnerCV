from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms      # transforms工具箱，里面各个类就是工具，将图片处理成相应的变换结果
from PIL import Image
import cv2

# python用法：transform和tensor
# 绝对路径的分隔符是需要修改的，如D:\Study\LearnTorch\datasets\中国画细支\2.png
img_path = "datasets/中国画细支/2.png"
img = Image.open(img_path)
print(type(cv2.imread(img_path)))

writer = SummaryWriter('logs')

# 1、如何使用transform：从transform中选择一个class进行创建，之后传入所需要的参数进行使用（查看文档）--主要关注输入和输出
trans_tensor = transforms.ToTensor()        # 定义totensor的class对象
img_trans = trans_tensor(img)       # 传入PIL类型的图像，torch常用的图片打开方法
print(img_trans)

writer.add_image('tensor_img', img_trans)
writer.close()
