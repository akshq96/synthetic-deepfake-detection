# Reproducibility

## Seed policy

Every training run seeds Python's `random`, `numpy`, and `torch` from
`training.seed` in its config (`ml.training.train.set_seed`, called first
thing in `run_training`). Synthetic-data generation (`ml/synthetic/`) and
split assignment (`ml/data_pipeline/split.py`) are independently seeded via
their own `seed` parameters — a full pipeline run's randomness is therefore
controlled by exactly three seed values: `training.seed` (model init,
data shuffling), the split seed passed to `assign_splits`, and the seed
passed to `mix_into_manifest`/synthetic generation. All three default to
matching `training.seed` when driven through `ml.training.train`, so citing
one seed is enough to describe a run.

**Not fully deterministic today:** GPU (CUDA) execution is not bit-exact
reproducible without additionally setting
`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`,
which this repo does not do by default (it measurably slows some CUDA
kernels). CPU runs are deterministic given a fixed seed. If bit-exact GPU
reproduction is needed, add those two settings to `ml/training/train.py`'s
`set_seed` and expect a training-speed cost.

## Config versioning

Every run's fully-resolved config (after CLI dotlist overrides) is written
to `artifacts/checkpoints/<run_name>/resolved_config.yaml` and logged as an
MLflow artifact — so a run is reproducible from that one file plus the seed,
even if the checkout's config files have since changed. When citing a
result, cite the MLflow run id; `mlflow.artifacts.download_artifacts` (or
the `/api/mlflow/runs` endpoint) retrieves the exact config that produced
it.

## Reproducing the fixture smoke suite (no real data needed)

Every phase of the pipeline has a smoke test that runs the real pipeline —
manifest building, leakage checking, training, evaluation, XAI, video
processing — end to end against a tiny synthetic (non-photographic) fixture
dataset generated on the fly (`ml/tests/fixtures/synthetic_fixture.py`), so
correctness is verified without needing any real dataset:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
pytest
```

This is what CI (or a fresh clone) should run to confirm the pipeline is
intact — it does **not** produce any research result, only confirms every
stage runs correctly.

## Reproducing a real experiment (requires real data)

1. Obtain at least one dataset — see `docs/dataset_setup.md`.
2. Build and leakage-check its manifest via `ml/data_pipeline/` (a dedicated
   per-dataset ingestion script is a natural next addition once a specific
   dataset's raw layout is in hand — the manifest/split/leakage-check
   primitives are already dataset-agnostic).
3. Train a baseline (`ml/configs/cnn_baseline.yaml` or `vit_baseline.yaml`),
   and one synthetic-augmented run (same config,
   `data.synthetic_ratio` > 0), on a real GPU — see `docs/colab_setup.md` or
   `docs/kaggle_setup.md`.
4. Run `ml.evaluation.experiments.unseen_manipulation` and
   `cross_dataset` against both checkpoints to compare generalization.
5. Run `ml.evaluation.ablation` to sweep `data.synthetic_ratio` for the
   ratio-vs-generalization curve the research question is ultimately
   answered from.
6. Every step above logs to MLflow (`artifacts/mlflow.db` by default) —
   `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db` gives a
   local dashboard over every run, or use this project's own
   `/api/mlflow/runs` / `/compare/cnn-vit` / results pages.

No result from this step should be written into `docs/`, a paper draft, or
any commit message until it has actually been produced this way — see
`docs/methodology.md`'s opening note.
