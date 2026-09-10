"""Convolutional VAE for 64x64 RGB images, factored from the learning notebook."""
import torch
from torch import nn


class VAE(nn.Module):
    """New instances have fresh weights; forward returns reconstruction, mu, logvar."""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            # [B, 3, 64, 64] → [B, 32, 31, 31]
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # → [B, 64, 14, 14]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # → [B, 128, 6, 6]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # → [B, 256, 2, 2]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # Preserve batch dimension; flatten channels, height, and width.
            # → [B, 1024]
            nn.Flatten(start_dim=1),
        )
        self.mu_head = nn.Linear(1024, latent_dim)
        self.logvar_head = nn.Linear(1024, latent_dim)
        self.decoder = nn.Sequential(
            # [B, 32] → [B, 1024]
            nn.Linear(latent_dim, 1024),

            # Reshape without changing the values.
            # → [B, 1024, 1, 1]
            nn.Unflatten(dim=1, unflattened_size=(1024, 1, 1)),

            # → [B, 128, 5, 5]
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),

            # → [B, 64, 13, 13]
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),

            # → [B, 32, 30, 30]
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),

            # → [B, 3, 64, 64]
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),

            # Match our image values, which are scaled to [0, 1].
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x)
        return self.mu_head(features), self.logvar_head(features)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar
