# Reproduction protocol

## Scope

Phase 1: reproduce the CarRacing comparison between visual-only and visual-plus-memory control. Phase 2: reproduce latent-environment training and transfer. Phase 1 alone is not the complete paper.

## Historical architecture anchors

For CarRacing: 64x64 RGB preprocessing, 32 latent dimensions, LSTM hidden size 256, five Gaussian mixtures, and a small controller optimized with CMA-ES. Verify exact preprocessing, mixture parameterization, posterior sampling, action transforms, and optimization settings against the reference implementation before declaring fidelity.

## Milestones

1. Collect a few complete episodes; inspect consecutive observations and intervening actions.
2. VAE: separate reconstruction and KL metrics; inspect held-out road geometry.
3. Memory: held-out negative log likelihood, a simple prediction baseline, and multistep rollout inspection. Reset recurrent state at episode boundaries.
4. Controller: freeze V/M; optimize on training seeds; select on validation seeds.
5. Evaluate selected controllers on untouched test tracks. Report mean, standard deviation, per-episode returns, and number of episodes. Distinguish track variability from variation across independent training seeds.
6. Add the dream-training experiment only after the first pipeline is understood.

## Evaluation discipline

Split whole episodes, not adjacent frames. Keep final evaluation seeds out of tuning and checkpoint selection. Compare ablations on shared evaluation seeds and document compute budgets. A history-only baseline is a later extension; the z versus z+h comparison alone does not isolate predictive training from access to history.

Before each run record code commit and dirty status, resolved configuration, package versions, hardware, seeds, dataset identity and split, upstream checkpoint identity, and elapsed time. Store metrics as append-only JSONL with explicit step and metric names. Never overwrite a prior run directory.

## Current limitations

No models trained or results claimed. Dependencies and reference code revision are not pinned yet. Target hardware is unknown. Resolve these before full-scale experiments.
