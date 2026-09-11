import torch

def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """Sum pixels/latent dimensions per image, then average over the batch."""
    reconstruction = (x_hat - x).square().flatten(1).sum(1).mean()
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(1).mean()
    return reconstruction + beta * kl, reconstruction, kl


def train_vae_step(model, x, optimizer, beta=1.0):
    model.train()
    optimizer.zero_grad()
    x_hat, mu, logvar = model(x)
    loss, reconstruction, kl = vae_loss(x_hat, x, mu, logvar, beta)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "reconstruction": reconstruction.item(), "kl": kl.item()}

def paper_vae_loss(reconstruction, target, mu, logvar):
    reconstruction_per_image = (
        (reconstruction - target).square()
        .flatten(start_dim=1)
        .sum(dim=1)
    )

    kl_per_image = -0.5 * (
        1 + logvar - mu.square() - logvar.exp()
    ).sum(dim=1)

    kl_floor = 0.5 * mu.shape[1]
    effective_kl = kl_per_image.clamp_min(kl_floor)

    loss = (reconstruction_per_image + effective_kl).mean()

    metrics = {
        "loss": loss.detach().item(),
        "reconstruction": reconstruction_per_image.mean().detach().item(),
        "kl_raw": kl_per_image.mean().detach().item(),
    }
    return loss, metrics

def run_vae_epoch(model, loader, device, optimizer=None):
    """Train when an optimizer is supplied; otherwise evaluate."""
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "kl_raw": 0.0,
    }
    count = 0

    with torch.set_grad_enabled(training):
        for images in loader:
            images = images.to(device)
            reconstruction, mu, logvar = model(images)

            loss, metrics = paper_vae_loss(
                reconstruction, images, mu, logvar
            )

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite VAE loss.")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            count += batch_size

            for name in totals:
                totals[name] += metrics[name] * batch_size

    if count == 0:
        raise ValueError("Empty data loader.")

    return {name: value / count for name, value in totals.items()}
