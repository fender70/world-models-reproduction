import torch

from world_models.models.mdn_rnn import MDNRNN, MDNOutput, mdn_nll, sample_mdn
from world_models.training.mdn_rnn import align_latent_action_sequences, train_mdn_rnn_step


def test_forward_shapes_and_normalized_weights():
    model = MDNRNN(latent_dim=4, action_dim=2, hidden_dim=8, num_mixtures=3)
    output = model(torch.randn(2, 5, 4), torch.randn(2, 5, 2))

    assert output.mixture_logits.shape == (2, 5, 4, 3)
    assert output.means.shape == (2, 5, 4, 3)
    assert output.log_scales.shape == (2, 5, 4, 3)
    assert output.state[0].shape == (1, 2, 8)
    weights = output.mixture_logits.softmax(dim=-1)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5, 4))


def test_chunked_recurrence_matches_full_sequence():
    torch.manual_seed(0)
    model = MDNRNN(latent_dim=3, action_dim=2, hidden_dim=7, num_mixtures=2)
    latents = torch.randn(2, 6, 3)
    actions = torch.randn(2, 6, 2)

    full = model(latents, actions)
    first = model(latents[:, :3], actions[:, :3])
    second = model(latents[:, 3:], actions[:, 3:], first.state)

    torch.testing.assert_close(
        torch.cat((first.mixture_logits, second.mixture_logits), dim=1),
        full.mixture_logits,
    )


def test_nll_prefers_a_mean_close_to_the_target():
    target = torch.tensor([[[2.0]]])
    state = (torch.empty(0), torch.empty(0))
    close = MDNOutput(
        mixture_logits=torch.zeros(1, 1, 1, 1),
        means=torch.tensor([[[[2.0]]]]),
        log_scales=torch.zeros(1, 1, 1, 1),
        state=state,
    )
    far = MDNOutput(
        mixture_logits=close.mixture_logits,
        means=torch.tensor([[[[-2.0]]]]),
        log_scales=close.log_scales,
        state=state,
    )

    assert mdn_nll(close, target) < mdn_nll(far, target)


def test_nll_is_finite_and_backpropagates():
    model = MDNRNN(latent_dim=3, action_dim=2, hidden_dim=8, num_mixtures=4)
    output = model(torch.randn(2, 5, 3), torch.randn(2, 5, 2))
    loss = mdn_nll(output, torch.randn(2, 5, 3))
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_sampling_shape_and_seed_reproducibility():
    model = MDNRNN(latent_dim=3, action_dim=2, hidden_dim=8, num_mixtures=4)
    output = model(torch.randn(2, 5, 3), torch.randn(2, 5, 2))
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)

    first = sample_mdn(output, generator=first_generator)
    second = sample_mdn(output, generator=second_generator)
    assert first.shape == (2, 5, 3)
    torch.testing.assert_close(first, second)


def test_alignment_and_training_step():
    torch.manual_seed(0)
    model = MDNRNN(latent_dim=3, action_dim=2, hidden_dim=8, num_mixtures=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    latents = torch.randn(2, 6, 3)
    actions = torch.randn(2, 5, 2)

    z_t, a_t, z_next = align_latent_action_sequences(latents, actions)
    assert z_t.shape == a_t.shape[:2] + (3,)
    assert z_next.shape == z_t.shape
    before = model.mdn_head.weight.detach().clone()
    metrics = train_mdn_rnn_step(model, latents, actions, optimizer)

    assert metrics["loss"] > 0
    assert torch.isfinite(torch.tensor(metrics["grad_norm"]))
    assert not torch.equal(before, model.mdn_head.weight)
