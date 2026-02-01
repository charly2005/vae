import torch
from vae import VAE
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import Linear

class PositionalEncoding3D(torch.nn.Module):
    def __init__(self, num_frequencies=10):
        """
        Initializes the positional encoding for 3D coordinates.

        Args:
            num_frequencies (int): The number of different frequencies to use for encoding.
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.frequencies = 2 ** torch.arange(num_frequencies, dtype=torch.float32)

    def forward(self, points):
        """
        Applies positional encoding to the 3D points.

        Args:
            points (torch.Tensor): N x 3 tensor of 3D coordinates.

        Returns:
            torch.Tensor: N x (6*num_frequencies) tensor of encoded coordinates.
        """
        encoded_points = []
        for i in range(points.shape[1]):  # For each dimension (x, y, z)
            for freq in self.frequencies:
                encoded_points.append(torch.sin(freq * points[:, i:i+1]))
                encoded_points.append(torch.cos(freq * points[:, i:i+1]))
        return torch.cat(encoded_points, dim=-1)

class PointVAE(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()

        self.pos_enc = PositionalEncoding3D()
        self.input_dim=self.pos_enc.num_frequencies * 6
        self.decode_dim=3
        self.use_sigmoid=False
        self.z_size = hidden_dims[-1] // 2


        enc_layers = OrderedDict()
        dec_layers = OrderedDict()

        curr_dim = self.input_dim
        for i, h_dim in enumerate(hidden_dims[:-1]):
            enc_layers[f"fc_{i}"] = Linear(curr_dim, h_dim)
            enc_layers[f"relu_{i}"] = torch.nn.ReLU()
            curr_dim = h_dim

        enc_layers[f"fc_{len(hidden_dims)-1}"] = torch.nn.Linear(curr_dim, self.z_size*2)
        
        self.encoder = torch.nn.Sequential(enc_layers)

        curr_dim = self.z_size
        reversed_dim = hidden_dims[:-1][::-1]
        for i, h_dim in enumerate(reversed_dim):
            dec_layers[f"fc_{i}"] = Linear(curr_dim, h_dim)
            dec_layers[f"relu_{i}"] = torch.nn.ReLU()
            curr_dim = h_dim

        final_dim = self.decode_dim if self.decode_dim != -1 else 3 
        dec_layers[f"fc_out"] = Linear(curr_dim, final_dim)
        if self.use_sigmoid:
            dec_layers[f"sigmoid_out"] = torch.nn.Sigmoid()

        self.decoder = torch.nn.Sequential(dec_layers)

    def encode(self, x):
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
        x = self.pos_enc(x)
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
