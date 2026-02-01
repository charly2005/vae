import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, random_split
import wandb

wandb.init(project="mnist-vae")

# set up data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tensor_transform = transforms.ToTensor()

batch_size = 256
MNIST_data = datasets.MNIST(root = "./data",
							train = True,
							download = True,
							transform = tensor_transform)

MNIST_test = datasets.MNIST(root="./data",
                            train=False,
                            download=True,
                            transform=tensor_transform
)

train_size = int(0.8 * len(MNIST_data))
val_size = len(MNIST_data) - train_size

MNIST_train, MNIST_val = random_split(MNIST_data, [train_size, val_size])

MNIST_train_loader = torch.utils.data.DataLoader(dataset = MNIST_train,
							                    batch_size = batch_size,
								                shuffle = True)

MNIST_val_loader = torch.utils.data.DataLoader(dataset = MNIST_val,
                                                batch_size = batch_size,
                                                shuffle = False)

MNIST_test_loader = torch.utils.data.DataLoader(dataset = MNIST_test,
                                                batch_size = batch_size,
                                                shuffle = False)

from math import e
from losses import loss_func

def train(train_dataloader, val_dataloader, model, loss_func, optimizer, epochs):
    losses = []
    val_losses =[]
    # training loop
    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        running_loss = 0.0
        batch_progress = tqdm(train_dataloader, desc='Train Batches', leave=False)

        for iter, (images, labels) in enumerate(batch_progress):
            batch_size = images.shape[0]
            # images = images.reshape(batch_size, -1).to(device)
            images = images.to(device)
            output = model(images)
            loss = loss_func(output, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
 
            wandb.log({"train_loss": loss.item(), 
                        "epoch": epoch, 
                        "step": iter + epoch * len(train_dataloader)})

        avg_loss = running_loss / len(train_dataloader)
        losses.append(avg_loss)
        # tqdm.write(f'----\nEpoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}\n')
        
        # validation
        model.eval()
        val_loss = 0.0
        batch_progress = tqdm(val_dataloader, desc='Val Batches', leave=False)
        with torch.no_grad():  
            for iter, (images, labels) in enumerate(batch_progress):
                batch_size = images.shape[0]
                # images = images.reshape(batch_size, -1).to(device)
                images = images.to(device)
                output = model(images)
                loss = loss_func(output, images)
                val_loss += loss.item() 
                
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
        tqdm.write(f'----\nEpoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n')
        
        if epoch % 5 == 0:
            original = images[0]
            reconstructed = model(images[0].unsqueeze(0))['imgs']
            orig_numpy = original.cpu().detach().numpy().squeeze()
            recon_numpy = reconstructed.cpu().detach().numpy().squeeze()
            wandb.log({
                "original": wandb.Image(orig_numpy), 
                "reconstructed": wandb.Image(recon_numpy)})

    return losses, val_losses

# train
from vae import VAE

def loss_VAE(output, x):
    return loss_func(output, x)

image_shape = MNIST_train[0][0].shape
print(image_shape)
input_dim = torch.prod(torch.tensor(image_shape)).item()
print("input_dim: ", input_dim)

# we r decreasing hidden_dims here to force compression/a bottleneck
# some popular models like vggnet/resnet do increasing hidden_dims bc as they decrease the img size, they want to preserve features to extract
hidden_dims = [128, 32, 16, 4]

vae = VAE(input_dim, hidden_dims).to(device)
print(vae)
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

print(count_parameters(vae))

#---------------------
optimizer_vae = torch.optim.Adam(vae.parameters(),
                                lr = 1e-4,
                                weight_decay = 1e-8)

epochs = 20
#---------------------

log_vae = train(MNIST_train_loader,MNIST_val_loader, vae, loss_VAE, optimizer_vae, epochs)


torch.save(vae.state_dict(), "vae_model.pth")