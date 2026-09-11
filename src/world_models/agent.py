import torch

@torch.no_grad()
def agent_step(z, state, memory, controller):
    if state is None:
        # Beginning of an episode: no previous memory.
        h = z.new_zeros(z.shape[0], memory.lstm.hidden_size)
    else:
        # state = (hidden_state, cell_state)
        h = state[0][-1]  # [B, hidden_dim]

    action = controller(z, h)

    _, _, _, next_state = memory(
        z.unsqueeze(1),       # [B, 1, latent_dim]
        action.unsqueeze(1),  # [B, 1, action_dim]
        state=state,
    )

    return action, next_state
