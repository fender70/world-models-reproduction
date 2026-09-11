import numpy as np
import torch
from PIL import Image

def preprocess_observation(observation):
    """Emulate the published CarRacing wrapper's legacy preprocessing."""
    frame = np.asarray(observation)

    if frame.shape != (96, 96, 3) or frame.dtype != np.uint8:
        raise ValueError(
            f"Expected raw uint8 (96, 96, 3), "
            f"got {frame.dtype} {frame.shape}"
        )

    # Original wrapper removes the bottom instrument panel.
    cropped = frame[:84].astype(np.float64) / 255.0

    # Emulate scipy.misc.imresize's float-to-byte scaling.
    minimum = cropped.min()
    span = cropped.max() - minimum
    if span == 0:
        span = 1.0

    scaled = (cropped - minimum) * (255.0 / span)
    byte_image = (
        np.clip(scaled, 0.0, 255.0) + 0.5
    ).astype(np.uint8)

    resized = np.array(
        Image.fromarray(byte_image).resize(
            (64, 64),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )

    # Preserve the original wrapper's final uint8 wraparound explicitly.
    transformed = np.rint(
        (1.0 - resized.astype(np.float64)) * 255.0
    ).astype(np.int64)

    return np.remainder(transformed, 256).astype(np.uint8)

@torch.no_grad()
def encode_observation(observation, vae):
    """Encode one environment observation into a sampled latent."""
    pixels = preprocess_observation(observation)
    device = next(vae.parameters()).device

    x = (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )

    mu, logvar = vae.encode(x)
    return vae.reparameterize(mu, logvar)
