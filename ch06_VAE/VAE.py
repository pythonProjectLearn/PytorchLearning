# encoding:utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 128
learning_rate = 1e-3

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



# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start training
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_dataset):
        # Forward pass
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)

        # Compute reconstruction loss and kl divergence
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, len(train_dataset), reconst_loss.item(), kl_div.item()))

    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))