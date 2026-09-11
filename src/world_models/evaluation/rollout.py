import torch

from world_models.agent import agent_step

@torch.no_grad()
def rollout(
    env,
    encode_observation,
    memory,
    controller,
    seed=0,
    max_steps=1000,
):
    """Evaluate one episode using a Gymnasium-style environment."""
    memory.eval()
    controller.eval()

    observation, info = env.reset(seed=seed)
    state = None
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        # Must return [1, latent_dim] on the models' device.
        z = encode_observation(observation)

        action, state = agent_step(
            z, state, memory, controller
        )

        observation, reward, terminated, truncated, info = env.step(
            action.squeeze(0).cpu().numpy()
        )

        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            break

    return {"reward": total_reward, "steps": steps}
