# encoding:utf-8
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToPILImage



import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train_path = '/home/zt/Documents/Data_Tensorflow_keras_sklearn/mnist/mnist/x_train.npy'
y_train_path = '/home/zt/Documents/Data_Tensorflow_keras_sklearn/mnist/mnist/y_train.npy'
x_test_path = '/home/zt/Documents/Data_Tensorflow_keras_sklearn/mnist/mnist/x_test.npy'
y_test_path = '/home/zt/Documents/Data_Tensorflow_keras_sklearn/mnist/mnist/y_test.npy'


"""直接读取numpy的数据类型"""
class NumpyDataset(Dataset):
    """直接读取numpy的数据类型

    transforms.ToTensor():把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    transforms.ToPILImage():将shape为(C,H,W)的Tensor或shape为(H,W,C)的numpy.ndarray转换成PIL.Image
    """
    def __init__(self, x_path, y_path,x_transform=None):
        super(NumpyDataset, self).__init__()
        self.x = np.load(x_path)
        self.y = np.load(y_path)

    def __getitem__(self, index):
        x,y = self.x[index],self.y[index]
        x = x.reshape(28, 28, 1)
        x = transforms.ToTensor()(x)

        return x, y.astype(np.int64)

    def __len__(self):
        return len(self.x)

train_dataset = NumpyDataset(x_train_path, y_train_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=True)
test_dataset = NumpyDataset(x_test_path, y_test_path)
test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False)
for i, (images, labels) in enumerate(train_loader):
    # Move tensors to the configured device
    images = images.reshape(-1, 28 * 28).to(device)
    labels = labels.to(device)
    print(images.shape)
    print(labels)



"""读取图片"""
# 定义对数据的预处理
transform_img = transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)), # 归一化
        ])

# 训练集
# /home/zt/Documents/Data_Tensorflow_keras_sklearn/dog_cat/train
# /home/zt/Documents/Data_Tensorflow_keras_sklearn/dog_cat/train/dog
# /home/zt/Documents/Data_Tensorflow_keras_sklearn/dog_cat/train/cat
# dog cat 里面存着各自类型的图片
trainset = ImageFolder('/home/zt/Documents/Data_Tensorflow_keras_sklearn/dog_cat/train', transform_img)

trainloader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next() # 返回4张图片及标签
show = ToPILImage() # 可以把Tensor转成Image，方便可视化
show(torchvision.utils.make_grid((images+1)/2)).resize((400,100))


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# 保存和加载整个模型.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# 仅仅保存和加载模型参数(recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))