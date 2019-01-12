# encoding:utf-8
"""
在训练模型时会在前面加上
model.train()

在测试模型时在前面使用：
model.eval()

因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout。
Batch Normalization主要时对网络中间的每层进行归一化处理，并且使用变换重构(Batch Normalizing Transform)保证每层所提取的特征分布不会被破坏
训练时是正对每个min-batch的，但是在测试中往往是针对单张图片，即不存在min-batch的概念。
由于网络训练完毕后参数都是固定的，因此每个批次的均值和方差都是不变的，因此直接结算所有batch的均值和方差。
所有Batch Normalization的训练和测试时的操作不同

Dropout能够克服Overfitting，在每个训练批次中，通过忽略一半的特征检测器，可以明显的减少过拟合现象
在训练中，每个隐层的神经元先乘概率P，然后在进行激活，在测试中，所有的神经元先进行激活，然后每个隐层神经元的输出乘P
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

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


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')