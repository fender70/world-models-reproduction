import numpy as np
import torch

from world_models.agent import agent_step
from world_models.envs.carracing import (
    encode_observation,
    preprocess_observation,
)

@torch.no_grad()
def randomize_collection_models(vae, memory, controller, seed):
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(0.0, 0.01))

    model_scales = [
        (controller, scale),
        (vae, scale / 10_000.0),
        (memory, scale / 10_000.0),
    ]

    for model, effective_scale in model_scales:
        for parameter in model.parameters():
            values = (
                rng.standard_cauchy(size=tuple(parameter.shape))
                * effective_scale
            )

            parameter.copy_(
                torch.as_tensor(
                    values,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )

    return scale

@torch.no_grad()
def collect_episode(
    env,
    vae,
    memory,
    controller,
    seed=0,
    max_steps=1000,
):
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")

    vae.eval()
    memory.eval()
    controller.eval()

    scale = randomize_collection_models(
        vae, memory, controller, seed
    )
    torch.manual_seed(seed)

    # Continue with the existing base_env/reset/recording code.
    scale = randomize_collection_models(seed)
    torch.manual_seed(seed)

    # Historical collection deliberately continued despite done signals.
    base_env = env.unwrapped
    observation, _ = base_env.reset(seed=seed)
    state = None

    observations = [preprocess_observation(observation)]
    actions = []
    rewards = []
    terminated_flags = []
    truncated_flags = []

    for t in range(max_steps):
        z = encode_observation(observation, vae)

        if not torch.isfinite(z).all():
            raise RuntimeError(
                f"Non-finite latent: seed={seed}, step={t}. "
                "Do not save this episode."
            )

        action_tensor, state = agent_step(
            z,
            state,
            memory,
            controller,
        )

        state_is_finite = all(
            torch.isfinite(value).all().item()
            for value in state
        )
        if not torch.isfinite(action_tensor).all() or not state_is_finite:
            raise RuntimeError(
                f"Non-finite action or memory: seed={seed}, step={t}. "
                "Do not save this episode."
            )

        action = action_tensor.squeeze(0).cpu().numpy()

        observation, reward, terminated, truncated, _ = (
            base_env.step(action)
        )

        observations.append(preprocess_observation(observation))
        actions.append(action.copy())
        rewards.append(reward)
        terminated_flags.append(terminated)
        truncated_flags.append(truncated)

    return {
        "observations": np.stack(observations),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        # These are raw environment signals; collection continues.
        "terminated": np.asarray(terminated_flags, dtype=bool),
        "truncated": np.asarray(truncated_flags, dtype=bool),
        "collection_cutoff": np.asarray(True),
        "seed": np.asarray(seed, dtype=np.int64),
        "randomization_scale": np.asarray(scale),
    }
