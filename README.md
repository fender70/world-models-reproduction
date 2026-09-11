# World Models Reproduction

A guided PyTorch reimplementation of **World Models** by David Ha and Jürgen Schmidhuber (2018). The project rebuilds the vision, memory, and controller pipeline for CarRacing while keeping reusable code separate from explanatory notebooks.

> **Status:** End-to-end architecture and pilot training pipeline implemented. A 20-episode pilot dataset has been collected, the VAE and MDN-RNN have been trained on it, and controller optimization with CMA-ES is the current milestone. The results below are pipeline checks from a small pilot, not a reproduction of the paper's reported benchmark.

## Architecture

The agent is divided into three components:

| Component | Role | Implementation |
| --- | --- | --- |
| **Vision (V)** | Compress each 64 × 64 RGB frame into a 32-dimensional latent distribution | Convolutional VAE with reparameterization and decoder |
| **Memory (M)** | Model the next latent distribution from current latent, action, and recurrent state | LSTM with a per-coordinate Gaussian mixture density head |
| **Controller (C)** | Map the current latent and memory state to steering, gas, and brake | Single linear layer with 867 parameters |

At timestep (t):

[
z_t \sim V(o_t), \qquad
a_t = C([z_t, h_t]), \qquad
h_{t+1} = M(z_t, a_t, h_t).
]

The controller reads (z_t) and the MDN-RNN hidden state (h_t). The predicted next latent is not fed directly into the controller.

## Current progress

- [x] Implement the convolutional VAE and paper-style VAE loss.
- [x] Implement the MDN-RNN, Gaussian-mixture likelihood, and recurrent state handling.
- [x] Implement the 867-parameter linear controller and full agent step.
- [x] Connect the VAE, memory, controller, and Gymnasium CarRacing environment.
- [x] Implement the paper-inspired random-network collection policy.
- [x] Collect a 20-episode, 20,000-transition pilot dataset.
- [x] Preserve episode-level training and validation splits.
- [x] Train a pilot VAE and inspect held-out reconstructions.
- [x] Encode ordered episodes into latent distribution parameters.
- [x] Train a pilot MDN-RNN on full latent-action sequences.
- [ ] Optimize the controller with CMA-ES.
- [ ] Evaluate vision-only and vision-plus-memory controllers on held-out tracks.
- [ ] Scale data collection and training toward the paper's experimental budget.
- [ ] Explore imagined rollouts and training inside a learned environment.

## Pilot results

These values verify that the training pipeline learns on the current small dataset. They should not be compared directly with the paper's CarRacing score.

| Stage | Result |
| --- | --- |
| Dataset | 20 episodes, 1,000 transitions each |
| Split | 16 training episodes, 4 validation episodes |
| VAE validation loss | 171.08 → 68.75 over 5 epochs |
| VAE validation reconstruction term | 134.78 → 50.95 |
| VAE validation raw KL | 34.61 → 17.37 |
| MDN-RNN validation NLL | 42.118 → 31.146 over 10 epochs |
| Untrained-controller baseline | −81.12 mean reward on 3 development tracks |

Held-out VAE reconstructions preserve the main road geometry, car location, and scene structure in the pilot data. Controller performance has not yet been optimized or evaluated as a reproduction result.

## Getting started

Use Python 3.11 or later and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --extra research --extra dev
uv run --all-extras python -m ipykernel install \
  --user --name world-models --display-name "World Models"
uv run --all-extras jupyter lab
```

Select the **World Models** kernel in Jupyter.

To inspect the smoke configuration:

```bash
uv run --all-extras python -m world_models.inspect_config \
  configs/carracing_smoke.json
```

Gymnasium's Box2D dependencies may require SWIG and a native compiler. See the [CarRacing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/) for platform-specific setup.

## Notebook guide

Run the notebooks in order:

| Notebook | Purpose |
| --- | --- |
| [00_vae.ipynb](notebooks/00_vae.ipynb) | Build and inspect the VAE, latent sampling, and reconstruction objective |
| [01_memory.ipynb](notebooks/01_memory.ipynb) | Build the MDN-RNN, inspect mixture likelihoods, and verify recurrent state |
| [02_controller.ipynb](notebooks/02_controller.ipynb) | Build the controller and connect it to memory |
| [03_carracing.ipynb](notebooks/03_carracing.ipynb) | Run the full agent and collect paper-inspired CarRacing trajectories |
| [04_train_vae.ipynb](notebooks/04_train_vae.ipynb) | Train and validate the VAE on collected frames |
| [05_train_memory.ipynb](notebooks/05_train_memory.ipynb) | Encode ordered episodes and train the MDN-RNN |

Generated datasets and checkpoints are stored under `data/` and `runs/` and are excluded from source control.

## Repository structure

```text
world-models-reproduction/
├── configs/                    # Experiment configurations
├── data/                       # Generated datasets and manifests
├── docs/                       # Protocol, data contract, and deviations
├── notebooks/                  # Guided implementation and training notebooks
├── reports/                    # Curated figures and results
├── runs/                       # Checkpoints and metrics
├── scripts/                    # Future command-line experiment entry points
├── src/world_models/
│   ├── agent.py                # One agent interaction step
│   ├── data/
│   │   ├── collection.py       # Random-network trajectory collection
│   │   ├── frames.py           # Frame dataset
│   │   └── latents.py          # Episode encoding and latent dataset
│   ├── envs/carracing.py       # CarRacing preprocessing and VAE encoding
│   ├── evaluation/rollout.py   # Real-environment rollout evaluation
│   ├── models/
│   │   ├── controller.py       # Linear controller
│   │   ├── mdn_rnn.py          # LSTM and mixture-density output head
│   │   └── vae.py              # Convolutional VAE
│   └── training/
│       ├── mdn_rnn.py          # MDN loss and memory training epoch
│       └── vae.py              # VAE losses and training epoch
└── tests/                      # Correctness checks
```

## Reproduction design

The current collection pipeline follows the original implementation's broad procedure:

1. Randomize VAE, MDN-RNN, and controller parameters for each rollout.
2. Run the resulting fixed random policy for up to 1,000 steps.
3. Store aligned observations and actions.
4. Train the VAE without reward labels.
5. Encode ordered episodes and train the MDN-RNN without reward labels.
6. Freeze V and M, then optimize C from cumulative reward.

For each transition, the stored alignment is:

```text
observation[t], action[t] -> observation[t + 1]
```

The saved format retains (T+1) observations for (T) actions. Episode splits are preserved to prevent neighboring frames from leaking across training and validation.

## Scope and deviations

This is an independent learning and reproduction project. It targets modern PyTorch and Gymnasium's `CarRacing-v3`, while the original work used TensorFlow and `CarRacing-v0`.

The pilot also differs from the reported experiment in scale: it uses 20 rollouts, while the paper describes 10,000. Legacy image preprocessing is approximated with Pillow, so pixel-level output can vary from the original SciPy implementation. These differences must be resolved or documented before making benchmark-level comparisons.

See:

- [Reproduction protocol](docs/reproduction_protocol.md)
- [Data contract](docs/data_contract.md)
- [Implementation deviations](docs/deviations.md)
- [Experiment template](docs/experiment_template.md)

## References

- David Ha and Jürgen Schmidhuber. **World Models** (2018). [Interactive paper](https://worldmodels.github.io/)
- David Ha. [World Models Experiments](https://blog.otoro.net/2018/06/09/world-models-experiments/)
- [Original experiment code](https://github.com/hardmaru/WorldModelsExperiments)

Original implementation code and pretrained weights are not bundled.
