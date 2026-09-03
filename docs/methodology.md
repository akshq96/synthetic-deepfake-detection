# Methodology

This document is the single place that explains *why* the pipeline is built the
way it is — the research design, the leakage-prevention approach, what each
synthetic technique is meant to approximate, and the caveats that matter when
interpreting results. Individual modules reference this file rather than
repeating the reasoning inline.

## Research question

> Does synthetic data augmentation improve the generalization of deepfake
> detectors to unseen manipulation techniques?

This is treated as an open empirical question, not a foregone conclusion. The
system is built so the four experiment types in `ml/evaluation/experiments/`
can actually falsify the hypothesis — a synthetic-augmented model doing worse
than baseline on held-out manipulation types is a valid, reportable outcome,
not a bug. No result described anywhere in this repository's code, comments,
or generated plots should be read as a claim until it has been produced by an
actual run against real data and is traceable to a logged MLflow run id.

## Datasets and identity/leakage handling

Target datasets: FaceForensics++ (`ffpp`), Celeb-DF v2 (`celebdf`), DFDC
(`dfdc`) — see `docs/dataset_setup.md` for how to obtain them.

Every sample (image, or a face-cropped video frame) is one row in a Parquet
manifest (`ml/data_pipeline/manifest.py`, schema in
`ml/data_pipeline/schema.py`) carrying `source_dataset`, `source_video_id`,
`identity_id`, `label`, `manipulation_type`, `is_synthetic`, and `split`.

**Grouping key for splits.** `ml/data_pipeline/split.py` assigns whole
identities (not individual frames or clips) to a single split, using
`identity_id` when available and falling back to `source_video_id` when a
dataset doesn't label identities:

- **FaceForensics++**: identity is a source actor; `original_sequences` and
  each `manipulated_sequences/<method>` pair trace back to the same
  originating actor pair. Use the actor pair as `identity_id`.
- **Celeb-DF v2**: identity is the named subject folder.
- **DFDC**: identity is the speaker id from each chunk's `metadata.json`
  where present; falls back to per-video grouping otherwise (DFDC's speaker
  labeling is not exhaustive — this is a known, documented limitation of that
  dataset, not of this pipeline).

Split assignment (`assign_splits`) is a **deterministic hash** of
`(source_dataset, majority_label, group_key, seed)` — not
`sklearn.GroupShuffleSplit` — specifically so a given `(manifest, seed,
ratios)` always produces byte-identical splits without depending on any
library's internal RNG state or version.

**Leakage checker.** `ml/data_pipeline/leakage_check.py` is a hard gate, not
a warning: it asserts no `identity_id` and no `source_video_id` appears in
more than one split, and `ml/training/train.py` calls it before every run
starts (and again after synthetic-data mixing, since that mutates the
manifest). This is the single most safety-critical control in the codebase —
every other result depends on it being correct.

## Synthetic manipulation techniques

Five techniques live in `ml/synthetic/`, all implementing a shared
`SyntheticTechnique.apply(image, seed)` interface (`ml/synthetic/base.py`).
They deliberately operate on an **already face-cropped image**, not a raw
frame — Phase 1's face detection already solved "find the face"; these
modules' job is purely "apply a manipulation-like transform to a face crop."
This keeps them testable independent of face-detection accuracy and keeps
the boundary between "detect a face" and "manipulate a face" clean.

| Technique | File | What it approximates |
|---|---|---|
| `blend_warp` | `blend_warp.py` | Classic Deepfakes/FaceSwap splicing: a random smooth displacement field warps a donor (or self) image, Poisson-blended into a centered ellipse standing in for "the face region" (no landmark re-detection). |
| `freq_perturb` | `freq_perturb.py` | The "unnatural frequency spectrum" fingerprint reported for GAN-generated faces — perturbs high-frequency FFT magnitude, keeps phase. |
| `compression_artifact` | `compression_artifact.py` | Re-encoding/recompression traces from manipulated videos being re-saved. **Also reused for robustness evaluation — see the caveat below.** |
| `color_perturb` | `color_perturb.py` | Illumination/white-balance mismatch between a spliced face and its background. |
| `autoencoder_swap` | `autoencoder_swap.py` | The one *learned* (not hand-crafted) technique: a small, deliberately undertrained convolutional autoencoder — the same structure as the original toy "Deepfakes" method — trained for a handful of steps per call, producing genuine reconstruction/blur artifacts rather than a simulation of them. |

`ml/synthetic/mixer.py` controls what fraction of *training-split* data is
synthetic (`synthetic_ratio`) and which techniques are drawn from
(`synthetic_techniques`). Synthetic samples are generated only from
train-split real rows and inherit their source row's `identity_id`,
`source_video_id`, and `split` — they can never leak into val/test by
construction, and the leakage checker re-verifies this after every mix
rather than trusting it.

### The compression caveat

`compression_artifact` is used in two different roles: as a *training*
technique (in `synthetic_techniques`) and as a *robustness-evaluation*
perturbation (`ml/evaluation/experiments/robustness.py`). A model trained
with compression in its synthetic mix will trivially look "robust to
compression" at evaluation time — that's an expected, controlled result of
what it was shown during training, not evidence of general robustness to
unseen perturbations. `robustness.py`'s results always record
`compression_in_training_mix` explicitly for exactly this reason; any report
or plot built from those results must carry that flag rather than silently
presenting compression-robustness as a general finding.

## What counts as "unseen manipulation"

`ml/evaluation/experiments/unseen_manipulation.py` implements leave-one-
manipulation-type-out evaluation: for each fake `manipulation_type` present
in the manifest (e.g. FF++'s `Deepfakes`/`Face2Face`/`FaceSwap`/
`NeuralTextures`, or a `synthetic_<technique>` label), it trains with every
train-split row of that type excluded, then evaluates specifically on
val-split rows of that type. "Unseen" here means unseen *during training of
that specific model* — the type still exists elsewhere in the manifest (in
val/test, and in other models' training data), which is what makes the
comparison meaningful: the same held-out type is evaluated the same way
across a baseline (`synthetic_ratio=0`) and an augmented run, isolating the
effect of synthetic augmentation on generalization to that type.

A held-out type with zero eval-split samples (possible on small manifests)
is skipped with a recorded reason rather than silently omitted or forced to
raise — visible in the results JSON as `{"skipped": true, "reason": ...}`.

## Ablation design

`ml/evaluation/ablation.py` sweeps a config grid (most commonly
`data.synthetic_ratio` alone, but any dotted config key) sequentially —
real runs have exactly one GPU/CPU available, so sweeps are not
parallelized. Each grid point is trained and evaluated via the exact same
`run_training`/`evaluate_on_manifest` path as every other experiment, and
logged as its own MLflow run tagged with the sweep id, so any two points are
directly comparable and diffable in the MLflow UI.

## Calibration and abstention

Temperature scaling (`ml/training/calibrate.py`, Guo et al. 2017) is fit on
the validation set only, never train or test. The abstain margin
(`ml/models/predictor.py`) is not a hardcoded guess — `sweep_abstain_margin`
computes abstain-rate vs. accuracy-on-non-abstained across a range of
candidate margins on the val set, and that sweep (logged to MLflow as
`abstain_sweep_acc_margin_<m>` metrics) is what should drive the deployed
margin, trading off coverage against reliability.

## Face detection

MediaPipe (`ml/data_pipeline/face_detector.py`) was chosen over
dlib/MTCNN for being pip-installable with no compiled toolchain and
CPU-fast — important for the "runs identically on a CPU debug subset"
design goal. During development, mediapipe's newer releases (0.10.15+ and
the 1.0 line) either dropped the legacy `solutions.face_detection` API this
project uses or, in 1.0's Tasks-API replacement, crashed natively on macOS
in this project's dev environment (a native abort from a GPU-related graph
service, not a catchable Python exception). `mediapipe==0.10.14` is
exact-pinned as the newest version confirmed to expose the needed API
without crashing. The `FaceDetector` ABC keeps this swappable — a
`facenet-pytorch` MTCNN backend could be added behind the same interface if
mediapipe's accuracy proves insufficient on real data, without touching any
calling code.

## Backend job execution model

Inference (image/video detect) runs in-process and synchronously — fast
enough on CPU for single-request use in a solo/local-first tool. Training
and experiment runs go through `backend/app/services/job_launcher.py`
instead: a subprocess (`python -m ml.training.train ...`), tracked via a
`Run` database row holding its PID and log file path, polled rather than
pushed. This was chosen over a job queue (Celery/Redis) deliberately — at
most one training job is ever meaningfully running at a time (one GPU/CPU),
so a queue would add operational complexity (a broker to run, a worker
process to manage) with no corresponding benefit for this project's scope.
The known limitation this accepts: if the backend process restarts while a
job is running, its exit code can no longer be observed directly (the
in-memory `Popen` handle is gone), so status polling falls back to checking
whether the PID is still alive and reports a since-exited process as
failed with an explanatory message, rather than leaving it stuck "running"
forever.
