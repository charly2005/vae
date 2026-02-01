import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from losses import loss_func
from vae import VAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tensor_transform = transforms.ToTensor()

batch_size = 16
MNIST_data = datasets.MNIST(root = "./data",
							train = True,
							download = True,
							transform = tensor_transform)

MNIST_data_loader = torch.utils.data.DataLoader(dataset = MNIST_data,
							                    batch_size = batch_size,
								                shuffle = True)
hidden_dims = [128, 32, 16, 4]
input_dim = torch.prod(torch.tensor(MNIST_data[0][0].shape)).item()
vae_test = VAE(input_dim, hidden_dims).to(device)
test_imgs, _ = next(iter(MNIST_data_loader))
test_batch_size = test_imgs.shape[0]

with torch.no_grad():
    for iter, (images, labels) in enumerate(MNIST_data_loader):
            batch_size = images.shape[0]
            # images = images.reshape(batch_size, -1).to(device)
            images = images.to(device)
            print(images.shape)
            output = vae_test(images)
            print(output['imgs'].shape)
            loss = loss_func(output, images)
            break

def loss_VAE(output, x):
    return loss_func(output, x)

print(loss_VAE(output, images))