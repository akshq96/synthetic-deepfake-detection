# Dataset Setup

This repository does not ship any deepfake dataset — all three target
datasets require accepting a EULA / requesting access, which can't be
scripted. Run `python -m ml.scripts.download_datasets` at any time to check
what's present under `data/raw/` and get the setup instructions below
reprinted with your local paths filled in.

## FaceForensics++ (`ffpp`)

1. Request access via the form linked from the official repo:
   <https://github.com/ondyari/FaceForensics> (Access section) — requires
   agreeing to the FaceForensics++ terms of use.
2. Once approved, use the provided download script to fetch the
   c23-compressed videos (recommended for a first pass; raw/c40 are also
   available).
3. Place the result under:
   ```
   data/raw/ffpp/original_sequences/...
   data/raw/ffpp/manipulated_sequences/<Deepfakes|Face2Face|FaceSwap|NeuralTextures>/...
   ```
   This matches FaceForensics++'s own download-script output layout.

## Celeb-DF v2 (`celebdf`)

1. Request access per the instructions at
   <https://github.com/yuezunli/celeb-deepfakeforensics> — the authors grant
   access after a request-form submission.
2. Place the result under:
   ```
   data/raw/celebdf/Celeb-real/...
   data/raw/celebdf/Celeb-synthesis/...
   data/raw/celebdf/YouTube-real/...
   ```

## DFDC (`dfdc`)

1. Easiest path: the DFDC sample/preview set is hosted on Kaggle —
   <https://www.kaggle.com/c/deepfake-detection-challenge/data> (free Kaggle
   account required; also usable directly from a Kaggle Notebook without
   downloading — see `docs/kaggle_setup.md`).
2. The full DFDC dataset (larger) is described at
   <https://ai.meta.com/datasets/dfdc/>.
3. Place (or symlink, from a Kaggle Notebook) the result under:
   ```
   data/raw/dfdc/train_sample_videos/...
   ```
   Each video's real/fake label and speaker id come from that directory's
   `metadata.json`.

## After downloading

Run the check again to confirm the layout is recognized:

```bash
python -m ml.scripts.download_datasets
```

Building a manifest from raw video files (face detection, cropping, frame
sampling, split assignment, leakage check) uses the primitives in
`ml/data_pipeline/` — see `ml/tests/fixtures/synthetic_fixture.py` for the
shape of a finished manifest, and `docs/methodology.md` for how identity
grouping is defined per dataset.
