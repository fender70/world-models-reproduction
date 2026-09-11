import torch
from torch import nn

from world_models.models.mdn_rnn import MDNRNN


class Controller(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.linear = nn.Linear(latent_dim + hidden_dim, 3)

    def forward(self, z, h):
        # z: [B, latent_dim]
        # h: [B, hidden_dim]
        features = torch.cat([z, h], dim=-1)
        raw = self.linear(features)

        steering = torch.tanh(raw[:, 0:1])
        gas = (torch.tanh(raw[:, 1:2]) + 1) / 2
        brake = torch.tanh(raw[:, 2:3]).clamp_min(0)

        return torch.cat([steering, gas, brake], dim=-1)
