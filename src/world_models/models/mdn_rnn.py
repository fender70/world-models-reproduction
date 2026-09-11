"""LSTM memory with a per-coordinate Gaussian mixture output head."""

import torch
from torch import nn


class MDNHead(nn.Module):
    def __init__(self, hidden_dim, latent_dim=32, num_components=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components

        output_dim = latent_dim * num_components
        self.logits_head = nn.Linear(hidden_dim, output_dim)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden):
        # hidden: [B, T, H]
        B, T, _ = hidden.shape
        shape = (B, T, self.latent_dim, self.num_components)

        logits = self.logits_head(hidden).reshape(shape)
        mu = self.mu_head(hidden).reshape(shape)
        log_std = self.log_std_head(hidden).reshape(shape)

        return logits, mu, log_std


class MDNRNN(nn.Module):
    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        hidden_dim=256,
        num_components=5,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = MDNHead(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_components=num_components,
        )

    def forward(self, z, actions, state=None):
        """
        Args:
            z: [B, T, D] current latent vectors.
            actions: [B, T, A] corresponding actions.
            state: Optional (hidden_state, cell_state) tuple.
                Each tensor has shape [1, B, H].
                None starts from zero memory.

        Returns:
            logits, mu, log_std: Each [B, T, D, K].
            next_state: Updated (hidden_state, cell_state).
        """
        inputs = torch.cat([z, actions], dim=-1)
        hidden_sequence, next_state = self.lstm(inputs, state)
        logits, mu, log_std = self.head(hidden_sequence)

        return logits, mu, log_std, next_state
