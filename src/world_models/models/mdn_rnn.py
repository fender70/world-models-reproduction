"""Mixture-density recurrent network for latent dynamics.

The model consumes ``(z_t, a_t)`` pairs and predicts a diagonal mixture of
Gaussians for every dimension of ``z_{t+1}``.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

RecurrentState = tuple[Tensor, Tensor]


@dataclass
class MDNOutput:
    """Parameters of p(z[t+1] | z[<=t], a[<=t]) and the final LSTM state."""

    mixture_logits: Tensor
    means: Tensor
    log_scales: Tensor
    state: RecurrentState


class MDNRNN(nn.Module):
    """LSTM dynamics model with a per-latent-dimension Gaussian mixture head."""

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 256,
        num_mixtures: int = 5,
        num_layers: int = 1,
        min_log_scale: float = -7.0,
        max_log_scale: float = 7.0,
    ) -> None:
        super().__init__()
        if min(latent_dim, action_dim, hidden_dim, num_mixtures, num_layers) <= 0:
            raise ValueError("model dimensions and num_mixtures must be positive")
        if min_log_scale >= max_log_scale:
            raise ValueError("min_log_scale must be smaller than max_log_scale")

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.num_layers = num_layers
        self.min_log_scale = min_log_scale
        self.max_log_scale = max_log_scale

        self.rnn = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # For each latent coordinate and mixture: weight, mean, and log(scale).
        self.mdn_head = nn.Linear(hidden_dim, latent_dim * num_mixtures * 3)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RecurrentState:
        """Return zero-valued hidden and cell states."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = next(self.parameters())
        state_device = reference.device if device is None else device
        state_dtype = reference.dtype if dtype is None else dtype
        shape = (self.num_layers, batch_size, self.hidden_dim)
        h = torch.zeros(shape, device=state_device, dtype=state_dtype)
        return h, torch.zeros_like(h)

    def forward(
        self,
        latents: Tensor,
        actions: Tensor,
        state: RecurrentState | None = None,
    ) -> MDNOutput:
        """Predict mixture parameters for the next latent at every input step."""
        self._validate_inputs(latents, actions)
        rnn_input = torch.cat((latents, actions), dim=-1)
        hidden, final_state = self.rnn(rnn_input, state)

        batch_size, sequence_length, _ = hidden.shape
        raw = self.mdn_head(hidden).reshape(
            batch_size,
            sequence_length,
            self.latent_dim,
            self.num_mixtures,
            3,
        )
        mixture_logits, means, raw_log_scales = raw.unbind(dim=-1)
        log_scales = raw_log_scales.clamp(self.min_log_scale, self.max_log_scale)
        return MDNOutput(mixture_logits, means, log_scales, final_state)

    def _validate_inputs(self, latents: Tensor, actions: Tensor) -> None:
        if latents.ndim != 3 or actions.ndim != 3:
            raise ValueError("latents and actions must have shape [batch, time, features]")
        if latents.shape[:2] != actions.shape[:2]:
            raise ValueError("latents and actions must have matching batch and time dimensions")
        if latents.shape[-1] != self.latent_dim:
            raise ValueError(f"expected latent dimension {self.latent_dim}")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"expected action dimension {self.action_dim}")


def mdn_nll(output: MDNOutput, target: Tensor, reduction: str = "mean") -> Tensor:
    """Negative log-likelihood of target latents under an MDN prediction.

    With ``reduction='none'``, returns one loss for every batch, time, and
    latent coordinate. The mean reduction averages over all three axes.
    """
    expected_shape = output.means.shape[:-1]
    if target.shape != expected_shape:
        raise ValueError(f"target must have shape {expected_shape}, got {tuple(target.shape)}")

    target = target.unsqueeze(-1)
    inverse_scales = torch.exp(-output.log_scales)
    standardized = (target - output.means) * inverse_scales
    component_log_probs = (
        -0.5 * standardized.square()
        - output.log_scales
        - 0.5 * math.log(2.0 * math.pi)
    )
    mixture_log_probs = F.log_softmax(output.mixture_logits, dim=-1)
    losses = -torch.logsumexp(mixture_log_probs + component_log_probs, dim=-1)

    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")


def sample_mdn(
    output: MDNOutput,
    temperature: float = 1.0,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw next-latent samples, returning shape ``[batch, time, latent]``."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    probabilities = F.softmax(output.mixture_logits / temperature, dim=-1)
    flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    components = torch.multinomial(
        flat_probabilities, num_samples=1, replacement=True, generator=generator
    ).reshape(*probabilities.shape[:-1], 1)
    means = output.means.gather(-1, components).squeeze(-1)
    log_scales = output.log_scales.gather(-1, components).squeeze(-1)
    noise = torch.randn(
        means.shape,
        device=means.device,
        dtype=means.dtype,
        generator=generator,
    )
    return means + torch.exp(log_scales) * math.sqrt(temperature) * noise
