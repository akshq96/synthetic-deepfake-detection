# Kaggle Notebooks Setup

Kaggle Notebooks give a free P100/T4 GPU with a 30 hrs/week quota — often
more stable for long training runs than Colab's session limits, and several
deepfake datasets (notably DFDC's sample set) are hosted directly on
Kaggle, avoiding a separate download step.

## 1. Create a notebook, enable GPU

Notebook Settings → Accelerator → GPU. Kaggle notebooks persist their
working directory (`/kaggle/working/`) across a session, which serves the
same role Drive does for Colab — no separate mount step needed, but do copy
anything you want to keep out of `/kaggle/working/` before the notebook's
storage quota is hit.

## 2. Clone and install

```bash
!git clone <your-repo-url> repo
%cd repo
!pip install -q -e . --no-deps
!pip install -q -r requirements-colab.txt
```

(The file is named for Colab but its reasoning — skip reinstalling
torch/torchvision, since the platform preinstalls a pinned CUDA build —
applies identically here.)

## 3. Attach a dataset instead of downloading

If your target dataset (e.g. DFDC's sample set) is already available as a
Kaggle Dataset, attach it via Notebook → Add Data rather than downloading —
it mounts read-only under `/kaggle/input/<dataset-name>/`. Point
`data.manifest_path` (after building a manifest from it) or your raw-data
ingestion step at that path instead of `data/raw/`.

## 4. Train / evaluate

Identical commands to Colab (see `docs/colab_setup.md` steps 4–6) — only
the storage-mounting step differs between the two platforms; every `ml.*`
command is otherwise platform-agnostic.

```bash
!python -m ml.training.train --config ml/configs/cnn_baseline.yaml \
    training.debug=false \
    data.manifest_path=/kaggle/working/repo/data/manifests/<your_dataset>_split.parquet \
    artifacts_root=/kaggle/working/artifacts
```

## 5. Save results before the session ends

Kaggle notebook sessions are also time-limited. Before it ends, either
commit the notebook (Kaggle snapshots `/kaggle/working/` on commit) or
manually download `artifacts/checkpoints/` and `artifacts/mlflow.db` —
there is no always-on persistent mount analogous to Colab+Drive here.
