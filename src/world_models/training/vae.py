"""Basic learning objective, without historical KL tolerance/clamping."""

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
