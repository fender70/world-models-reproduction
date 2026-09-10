# Episode data contract

Proposed compressed NPZ per episode, plus a JSON manifest. Implement and validate this contract before collecting a large dataset.

- `observations`: uint8 [T+1, H, W, 3], including the final observation.
- `actions`: float32 [T, 3]; action[t] is applied to observation[t].
- `rewards`: float32 [T].
- `terminated`, `truncated`: bool [T]; retain separately.
- observation[t+1] must be the actual result of action[t], never an auto-reset frame.

Store raw observations; make resizing/cropping explicit and versioned. The manifest records schema version, environment version and kwargs, collection policy, seed, episode length, preprocessing, filenames and checksums. Episode IDs and split membership must be stable. Never allow sequences to cross episode boundaries.

For memory training cache VAE posterior means and log variances with VAE checkpoint identity; document whether and how latents are sampled. Caches from different encoders are not interchangeable.
