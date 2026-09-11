import numpy as np
import torch
from torch.utils.data import Dataset


@torch.no_grad()
def encode_episode(vae, observations, batch_size=64):
    """Return ordered float32 mu/logvar arrays of shape [T+1, D]."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if (
        observations.dtype != np.uint8
        or observations.ndim != 4
        or observations.shape[1:] != (64, 64, 3)
        or len(observations) == 0
    ):
        raise ValueError("Expected nonempty uint8 [N, 64, 64, 3].")

    vae.eval()
    device = next(vae.parameters()).device
    all_mu, all_logvar = [], []

    for start in range(0, len(observations), batch_size):
        images = (
            torch.from_numpy(observations[start:start + batch_size])
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            / 255.0
        )

        mu, logvar = vae.encode(images)

        if not (
            torch.isfinite(mu).all()
            and torch.isfinite(logvar).all()
        ):
            raise RuntimeError("Non-finite encoded distribution.")

        all_mu.append(mu.cpu().numpy())
        all_logvar.append(logvar.cpu().numpy())

    return (
        np.concatenate(all_mu, axis=0),
        np.concatenate(all_logvar, axis=0),
    )

class LatentEpisodes(Dataset):
    def __init__(self, directory, manifest, split):
        self.episodes = []

        for record in manifest["episodes"]:
            if record["split"] != split:
                continue

            with np.load(directory / record["file"]) as data:
                self.episodes.append({
                    key: torch.from_numpy(data[key].copy()).float()
                    for key in ("mu", "logvar", "actions")
                })

        if not self.episodes:
            raise ValueError(f"No episodes for split: {split}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        return self.episodes[index]
