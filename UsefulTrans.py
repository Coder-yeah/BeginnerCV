from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
# 需要注意图片打开后是否是3个信道的，不是的话需要RGB转化，否则归一化会报错
img = Image.open("datasets/七匹狼观海中支/4.png").convert('RGB')
print(img)

# ToTensor的使用
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image('tensor', img_tensor)

# Normalize的使用
# output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])     # 2*input-1
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('norm', img_norm)

# Resize用法:注意输入输出格式为PIL
print(img.size)
trans_size = transforms.Resize(200)
img_resize = trans_size(img)
print(img_resize)       # 输出仍然是PIL的数据类型，需要转换
img_resize = trans_tensor(img_resize)
writer.add_image('resize', img_resize)

# Compose用法：需要传入的是transforms数据类型的参数列表
trans_size_2 = transforms.Resize((256, 256))
# 关注输入和输出的变化是否匹配：PIL->PIL->tensor
trans_compose = transforms.Compose([trans_size_2, trans_tensor])
img_resize_2 = trans_compose(img)
writer.add_image('compose', img_resize_2)

# RandomCrop随即裁剪:输入输出都是PIL
trans_random = transforms.RandomCrop(128)
trans_compose_2 = transforms.Compose([trans_random, trans_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('random2', img_crop, i)

writer.close()

"""
总结用法：1、关注输入和输出；2、多看官方文档；3、关注方法需要什么参数，以及参数的解释
不知道返回值的数据类型时，可以1、print；2、print(type())；3、debug
"""

"""
class Person:
    # 内置call函数的使用：可以直接传入参数进行调用，如果是使用方法的话需要用点。例如：
    def __call__(self, name):
        print('__call__'+'hello'+name)

    def hello(self, name):
        print('method'+'hello'+name)


person = Person()
person('coco')
person.hello('hihi')
"""

