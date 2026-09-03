# Google Colab Setup

Colab's free tier gives a T4 GPU, which is enough to train
EfficientNetV2-S/ConvNeXt-Tiny/ViT-Base at reasonable batch sizes. All
training code is plain Python scripts + YAML configs — a Colab notebook
cell just invokes them, it never contains real logic itself.

## 1. Mount Drive (so checkpoints survive a disconnect)

Colab's local disk is ephemeral — a session disconnect loses anything not
in Drive. Mount it and symlink `artifacts/` there before training:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/major_project_artifacts', exist_ok=True)
```

## 2. Clone and install

```bash
!git clone <your-repo-url> repo
%cd repo
!ln -s /content/drive/MyDrive/major_project_artifacts artifacts

# Colab preinstalls torch/torchvision — installing the [torch] extra would
# fight its pinned CUDA build, so skip it and use the trimmed requirements
# file instead (see requirements-colab.txt for what it covers and why).
!pip install -q -e . --no-deps
!pip install -q -r requirements-colab.txt
```

## 3. Get your data onto the runtime

Either upload/download it directly into `data/raw/` (see
`docs/dataset_setup.md`), or mount it from Drive the same way as
`artifacts/` above if you've stored it there.

## 4. Train

```bash
!python -m ml.training.train --config ml/configs/cnn_baseline.yaml \
    training.debug=false \
    data.manifest_path=data/manifests/<your_dataset>_split.parquet
```

Because `artifacts/` is a symlink into Drive, checkpoints
(`artifacts/checkpoints/<run_name>/latest.pt`) and the MLflow store
(`artifacts/mlflow.db`) persist across disconnects.

## 5. Resume after a disconnect

```bash
!python -m ml.training.train --config ml/configs/cnn_baseline.yaml \
    training.debug=false training.resume=auto \
    data.manifest_path=data/manifests/<your_dataset>_split.parquet
```

`training.resume=auto` finds `artifacts/checkpoints/<run_name>/latest.pt`
and continues from its saved epoch/step/optimizer state
(`ml/training/checkpoint.py`).

## 6. Everything else runs the same way

Evaluation experiments, ablation sweeps, and XAI all follow the same
pattern — `!python -m ml.evaluation.experiments.<name> --config ... `. See
`docs/reproducibility.md` for the full sequence.
