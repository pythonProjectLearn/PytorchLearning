# encoding:utf-8
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# MNIST dataset
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


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
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
with torch.no_grad():
    """torch.no_grad不更新权重"""
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')