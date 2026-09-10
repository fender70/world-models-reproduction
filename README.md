# World Models reproduction

A guided PyTorch reimplementation of Ha & Schmidhuber (2018).
Status: VAE and basic VAE training step implemented; guided memory notebook started. Data collection, complete MDN-RNN, and controller training remain unimplemented.
There is no single canonical research layout; this uses a conventional installable `src/` package with configuration-driven experiments.

## Setup

Use Python 3.11+ in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[research,dev]'
python -m ipykernel install --user --name world-models --display-name 'World Models'
python -m world_models.inspect_config configs/carracing_smoke.json
jupyter lab
```

On Windows activate `.venv\Scripts\activate` instead. Box2D may require SWIG and a native compiler if no compatible wheel is available. Choose the PyTorch build for your hardware before installing research dependencies. Dependency ranges are deliberately unresolved until the target machine is known; this is not a locked environment yet. After a successful install, capture `python -m pip freeze > requirements-lock.txt` and record Python/OS/hardware in the experiment log.

## Structure

| Path | Responsibility |
| --- | --- |
| `src/world_models/` | Reusable implementation; importable from notebooks and scripts |
| `configs/` | Explicit experiment settings, tracked in Git |
| `notebooks/` | Inspection, learning, and plots; no duplicated training implementations |
| `scripts/` | Thin launchers when training commands are implemented |
| `tests/` | Meaningful checks for alignment, probability losses, and state resets |
| `data/` | Local generated datasets and manifests; ignored by Git |
| `runs/` | Per-run checkpoints, metrics, resolved configs, provenance; ignored |
| `docs/` | Protocol, deviations, and experiment notes |
| `reports/` | Small curated findings and figures suitable for Git |

## Workflow

1. Read `docs/reproduction_protocol.md` and `docs/data_contract.md`.
2. Open `notebooks/00_environment_and_data.ipynb` for the first milestone.
3. Implement collection in `src/world_models/data/`; keep notebooks as clients.
4. Validate the small pipeline before scaling VAE, memory, then controller training.
5. Record each run using `docs/experiment_template.md`; preserve failures too.

The smoke configuration is a proposed debugging budget, not a faithful reproduction configuration. CarRacing-v3 is a modern environment deviation. The historical target used CarRacing-v0. Do not directly equate scores across versions.

## Sources

- Interactive paper: https://worldmodels.github.io/
- Author reproduction guide: https://blog.otoro.net/2018/06/09/world-models-experiments/
- Modern environment documentation: https://gymnasium.farama.org/environments/box2d/car_racing/

No original implementation or weights are bundled. No public repository has been created. After extraction, initialize version control with `git init -b main`, review `git status`, then make your initial commit.

## Architecture learning

Run `uv sync --extra research --extra dev` and launch `uv run --all-extras jupyter lab`.
Start with `notebooks/00_world_models_architecture.ipynb`, then `notebooks/01_memory.ipynb`.
The original uploaded learning notebook is retained in `notebooks/archive/`.
The refactor does not recover in-memory learned weights from saved notebook outputs.
Commit your locally generated `uv.lock` and `.python-version`; keep `.venv` ignored.
