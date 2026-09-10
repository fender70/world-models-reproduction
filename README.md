# World Models Reproduction

An independent PyTorch reimplementation of **World Models** by David Ha and Jürgen Schmidhuber (2018), developed to understand learned visual representations, recurrent dynamics, and control.

The project builds the architecture component by component, pairing reusable implementations with explanatory notebooks. The initial reproduction target is CarRacing; training a controller inside a learned environment is a later milestone.

**Status:** Work in progress. The VAE and a basic training step are implemented. The memory notebook introduces the LSTM backbone; the complete MDN-RNN and controller are not yet implemented. No benchmark reproduction results are reported.

## Architecture

| Component | Role | Current implementation |
| --- | --- | --- |
| **Vision (V)** | Encode a 64 × 64 RGB image as a distribution over 32-dimensional latent vectors | Convolutional VAE, reparameterization, decoder, reconstruction and KL losses |
| **Memory (M)** | Summarize observations and actions to predict the next latent distribution | Introductory LSTM notebook; mixture-density head and training pending |
| **Controller (C)** | Select actions from the current latent representation and memory state | Planned |

The VAE reconstructs the current frame. The memory model will learn temporal dynamics from latent–action sequences. The controller will use visual and memory features to select actions.

## Getting started

Use Python 3.11 or later and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --extra research --extra dev
uv run --all-extras python -m ipykernel install \
  --user --name world-models --display-name "World Models"
uv run --all-extras jupyter lab
```

Select the **World Models** kernel when opening a notebook. The `research` and `dev` extras are defined in `pyproject.toml` and include the modeling, environment, and notebook dependencies.

To check the configuration loader:

```bash
uv run --all-extras python -m world_models.inspect_config configs/carracing_smoke.json
```

Gymnasium's Box2D dependencies may require SWIG and a native compiler, depending on the platform. See the [CarRacing documentation](https://gymnasium.farama.org/environments/box2d/car_racing/) for environment details.

## Notebooks

| Notebook | Focus |
| --- | --- |
| [Vision architecture](notebooks/00_world_models_architecture.ipynb) | Run the factored VAE, inspect latent sampling, and take a basic training step |
| [Memory](notebooks/01_memory.ipynb) | Align latent–action sequences, inspect LSTM shapes, and carry recurrent state across timesteps |
| [Environment and data](notebooks/00_environment_and_data.ipynb) | Starting point for environment inspection and data collection |

Start with **Vision architecture**, then **Memory**. Their synthetic inputs are computation checks, not evidence of learned driving behavior. Earlier exploratory work is preserved in `notebooks/archive/`.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/world_models/models/vae.py` | VAE architecture and latent sampling |
| `src/world_models/training/vae.py` | Basic VAE objective and optimizer step |
| `src/world_models/` | Package structure for data, environments, models, training, and evaluation |
| `notebooks/` | Guided explanations and interactive inspection |
| `configs/` | Experiment configurations |
| `docs/` | Reproduction protocol, data contract, deviations, and experiment notes |
| `tests/` | Reserved for implementation correctness checks |
| `scripts/` | Reserved for experiment entry points |
| `data/` | Generated datasets and manifests |
| `runs/` | Run outputs, checkpoints, and metrics |
| `reports/` | Curated results and figures |

## Using the VAE

```python
import torch

from world_models.models.vae import VAE
from world_models.training.vae import train_vae_step

model = VAE(latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Synthetic batch for checking the computation; image values are in [0, 1].
images = torch.rand(2, 3, 64, 64)
reconstructions, mu, logvar = model(images)
metrics = train_vae_step(model, images, optimizer, beta=1.0)
print(metrics)
```

Each new model starts with randomly initialized weights. Saved notebook outputs do not preserve model weights; trained parameters must be saved separately as checkpoints.

The basic loss sums reconstruction errors over pixels and KL divergence over latent dimensions, then averages each term over the batch. It does not yet reproduce the historical KL tolerance/clamping recipe.

## Reproduction scope

This project uses a modern PyTorch implementation and targets Gymnasium's `CarRacing-v3`. Environment and training differences must be documented before comparing results with the original paper. The smoke configuration is a debugging budget, not a faithful reproduction configuration.

Before running experiments:

- Follow the [reproduction protocol](docs/reproduction_protocol.md) and [data contract](docs/data_contract.md).
- Record implementation differences in [deviations](docs/deviations.md).
- Log seeds, hardware, dependency versions, configuration, and outcomes using the [experiment template](docs/experiment_template.md).
- Commit `uv.lock` after resolving dependencies, and record the Python version. Keep generated datasets and run artifacts out of source control.

## Roadmap

- [x] Implement the convolutional VAE and basic training step.
- [x] Introduce sequence alignment and recurrent state in a memory notebook.
- [ ] Implement the mixture-density output head, likelihood loss, and sampling.
- [ ] Collect and validate environment trajectories.
- [ ] Train visual and memory models on recorded data.
- [ ] Implement and optimize the controller.
- [ ] Evaluate vision-only and vision-plus-memory controllers on held-out tracks.
- [ ] Explore training inside a learned environment and transfer to the real simulator.

## References

- David Ha and Jürgen Schmidhuber. **World Models** (2018). [Interactive paper](https://worldmodels.github.io/).
- David Ha. [World Models experiments and reproduction guide](https://blog.otoro.net/2018/06/09/world-models-experiments/).

This is an independent learning and reproduction project. Original implementation code and pretrained weights are not bundled.
