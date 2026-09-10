"""Training utilities for the latent dynamics model."""

from torch import Tensor, nn

from world_models.models.mdn_rnn import MDNRNN, mdn_nll


def align_latent_action_sequences(latents: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Create ``z_t``, ``a_t``, and ``z_{t+1}`` tensors for teacher forcing."""
    if latents.ndim != 3 or actions.ndim != 3:
        raise ValueError("latents and actions must have shape [batch, time, features]")
    if latents.shape[0] != actions.shape[0]:
        raise ValueError("latents and actions must have the same batch size")
    if latents.shape[1] != actions.shape[1] + 1:
        raise ValueError("latents must contain exactly one more timestep than actions")
    return latents[:, :-1], actions, latents[:, 1:]


def train_mdn_rnn_step(
    model: MDNRNN,
    latents: Tensor,
    actions: Tensor,
    optimizer,
    *,
    max_grad_norm: float | None = 1.0,
) -> dict[str, float]:
    """Take one teacher-forced optimization step on a batch of trajectories."""
    model.train()
    z_t, a_t, z_next = align_latent_action_sequences(latents, actions)
    optimizer.zero_grad()
    loss = mdn_nll(model(z_t, a_t), z_next)
    loss.backward()

    grad_norm = None
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    metrics = {"loss": loss.detach().item()}
    if grad_norm is not None:
        metrics["grad_norm"] = grad_norm.detach().item()
    return metrics
