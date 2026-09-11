"""Negative log-likelihood for the MDN-RNN."""

import math

import torch
from torch.nn import functional as F


def mdn_loss(logits, mu, log_std, targets):
    """
    Args:
        logits, mu, log_std: [B, T, D, K].
        targets: [B, T, D] next latent vectors.

    Returns:
        NLL summed over latent coordinates, averaged over batch and time.

    Each coordinate has its own mixture; coordinates are assumed
    conditionally independent.
    """
    log_pi = F.log_softmax(logits, dim=-1)
    log_std = log_std.clamp(min=-7.0, max=7.0)

    target = targets.unsqueeze(-1)
    standardized = (target - mu) * torch.exp(-log_std)

    log_density = (
        -0.5 * standardized.square()
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    )

    log_prob_per_dim = torch.logsumexp(
        log_pi + log_density, dim=-1
    )

    return -log_prob_per_dim.sum(dim=-1).mean()

def run_memory_epoch(
    memory,
    loader,
    device,
    optimizer=None,
    validation_seed=12345,
):
    training = optimizer is not None
    memory.train(training)

    validation_rng = torch.Generator().manual_seed(validation_seed)
    total_nll = 0.0
    total_transitions = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            mu = batch["mu"]
            logvar = batch["logvar"]

            if training:
                epsilon = torch.randn_like(mu)
            else:
                epsilon = torch.randn(
                    mu.shape,
                    generator=validation_rng,
                    dtype=mu.dtype,
                )

            z = (mu + torch.exp(0.5 * logvar) * epsilon).to(device)
            actions = batch["actions"].to(device)

            logits, means, log_std, _ = memory(
                z[:, :-1],
                actions,
                state=None,
            )
            loss = mdn_loss(logits, means, log_std, z[:, 1:])

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite memory loss.")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    memory.parameters(), max_norm=1.0
                )
                optimizer.step()

            transitions = actions.shape[0] * actions.shape[1]
            total_nll += loss.item() * transitions
            total_transitions += transitions

    if total_transitions == 0:
        raise ValueError("Empty data loader.")

    return total_nll / total_transitions
