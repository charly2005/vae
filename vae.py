import torch

from torchvision.datasets.utils import zipfile
from collections import OrderedDict
from torch.nn import Conv2d as Conv2d
from torch.nn import BatchNorm2d as BatchNorm2d
from torch.nn import ConvTranspose2d as ConvT2d
import torch.nn.functional as F
from torch.nn import Linear
class VAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, decode_dim=-1, use_sigmoid=True):
        '''
        input_dim: The dimensionality of the input data.
        hidden_dims: A list of hidden dimensions for the layers of the encoder and decoder.
        decode_dim: (Optional) Specifies the dimensions to decode, if different from input_dim.
        '''
        super().__init__()
        self.img_w = int(input_dim**0.5)
        self.z_size = hidden_dims[-1] // 2

        ##################
        ### Problem 2(b): finish the implementation for encoder and decoder
        ##################
        enc_layers = OrderedDict()
        dec_layers = OrderedDict()
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        curr_ch = 1

        for i, h_dim in enumerate(hidden_dims[:-1]):
          enc_layers[f"conv_{i}"] = Conv2d(curr_ch, h_dim,kernel_size=3,stride=2,padding=1)
          enc_layers[f"relu_{i}"] = self.relu
          curr_ch = h_dim

        enc_layers[f"flatten"] = torch.nn.Flatten()
        self.encoder = torch.nn.Sequential(enc_layers)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.img_w, self.img_w)
            dummy_output = self.encoder(dummy_input)
            
            flat_ch = dummy_output.view(1, -1).shape[1]

        feature_w = self.img_w // (2 ** len(hidden_dims))

        enc_layers[f"fc"] = Linear(flat_ch, self.z_size * 2) 
        self.encoder = torch.nn.Sequential(enc_layers)

        dec_layers[f"fc"] = Linear(self.z_size, curr_ch * (feature_w)**2,)
        dec_layers[f"unflatten"] = torch.nn.Unflatten(1, (curr_ch, feature_w, feature_w))
        reversed_dim = hidden_dims[:-1][::-1] + [1]

        for i, h_dim in enumerate(reversed_dim):
            dec_layers[f"conv_{i}"] = ConvT2d(curr_ch, h_dim, kernel_size=3,stride=2,output_padding=1)
            if i < len(reversed_dim) - 1:
                dec_layers[f"relu_{i}"] = self.relu
            else:
                if use_sigmoid:
                    dec_layers[f"sigmoid_{i}"] = self.sigmoid
            curr_ch = h_dim
        
        self.decoder = torch.nn.Sequential(dec_layers)

    def encode(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, self.img_w, self.img_w)
        # take fc layer w output channels self.z_size *2 and split it into 2 separate layers
        mean, logvar = torch.split(self.encoder(x), split_size_or_sections=[self.z_size, self.z_size], dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar, n_samples_per_z=1):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def decode(self, z):
        probs = self.decoder(z)
        return probs

    def forward(self, x, n_samples_per_z=1):
        mean, logvar = self.encode(x)

        batch_size, latent_dim = mean.shape
        if n_samples_per_z > 1:
            mean = mean.unsqueeze(1).expand(batch_size, n_samples_per_z, latent_dim)
            logvar = logvar.unsqueeze(1).expand(batch_size, n_samples_per_z, latent_dim)

            mean = mean.contiguous().view(batch_size * n_samples_per_z, latent_dim)
            logvar = logvar.contiguous().view(batch_size * n_samples_per_z, latent_dim)

        z = self.reparameterize(mean, logvar, n_samples_per_z)
        x_probs = self.decode(z)

        x_probs = x_probs.reshape(batch_size, n_samples_per_z, -1)
        x_probs = torch.mean(x_probs, dim=[1])

        return {
            "imgs": x_probs,
            "z": z,
            "mean": mean,
            "logvar": logvar
        }