# On-Device Personalized ECG Classifier

A small ECG classifier that tests two questions:

1. How much does performance drop when a heartbeat model sees a completely new patient?
2. Can a short, label-free enrollment period recover some of that drop on-device?

I tested this on the MIT-BIH Arrhythmia Database using an inter-patient split, then moved the model and enrollment pipeline to an iPhone with Core ML.

> **Research prototype, not a medical device.** The iPhone app replays MIT-BIH records. It does not use a live ECG sensor.

## Main results

| Result | Value |
|---|---:|
| Population model on unseen patients | macro-F1 0.403 ± 0.031 (v2) |
| v3 CNN + timing model | **macro-F1 0.499** |
| S sensitivity, v2 → v3 | 0.056 → **0.325** |
| V sensitivity, v2 → v3 | 0.864 → **0.909** |
| 60 s label-free prototype enrollment, v2 | **+0.074 macro-F1** |
| Same enrollment after fixing v2 overtraining | +0.036 |
| v3 inference on iPhone 15 Pro | about **0.152 ms/beat** |
| v2 prototype enrollment on iPhone, 60 s | **6.3 ms** |
| v2 FP16 model size | 62.9 KB |
| v2 6-bit model size | 32.5 KB |

The main result changed as the project went on. The first version made personalization look stronger than it really was. After switching to patient-grouped cross-validation, I found that v2 had been trained too long. Fixing that improved the population model and reduced the benefit of enrollment.

That became one of the more useful findings from the project.

For the full experiments, tables, parity checks, and failure analysis, see [`REPORT.md`](REPORT.md).

## Why the split matters

A random beat-level split on MIT-BIH can put beats from the same patient in both training and test data. That makes it much easier for a model to recognize patient-specific patterns instead of generalizing to someone new.

This project uses the de Chazal inter-patient split:

- DS1 records for training
- DS2 records for testing
- no patient record appears in both
- paced records 102, 104, 107, and 217 are excluded

The scores are much lower than the ~99% numbers often seen with random beat splits, but that is the point: DS2 is made of patients the model has not seen before.

## Model

Each beat uses:

- a 256-sample MLII waveform window, with the R peak at index 100
- per-beat z-score normalization
- four RR features:
  - pre-RR
  - post-RR
  - mean RR over the previous 10 beats
  - pre-RR / local RR ratio

The CNN has four convolution blocks followed by global average pooling and a 64-dimensional embedding. A linear head gets the embedding plus the RR features.

The main metric is macro-F1 over N, S, V, and F. I also report sensitivity and PPV by class. Accuracy is not very useful here because about 89% of DS2 beats are N.

## v2: label-free enrollment

For each DS2 patient, I used the first 30 seconds, 60 seconds, 2 minutes, or 5 minutes as an enrollment window and evaluated on beats after the 5-minute mark.

I compared two ways to adapt while keeping the backbone frozen:

### Linear-head fine-tuning

The phone treats every enrollment beat as N because there are no cardiologist annotations available during normal use.

Fine-tuning the linear head on those labels hurt performance:

- population macro-F1: 0.411
- after 60 s enrollment: 0.380
- change: **-0.030**

It mostly teaches the model to predict N more often.

### Prototype adaptation

The prototype method is more conservative. Instead of retraining the whole head, it moves the N class prototype toward the new patient's enrollment mean.

On v2:

- population macro-F1: 0.402
- after 60 s enrollment: 0.476
- change: **+0.074**

The improvement mostly came from reducing false positives, not from finding more arrhythmias.

## v3: fixing the population model

The original model was especially poor on S beats from unseen patients, so I went back to the training procedure.

Two things stood out.

### 1. v2 was overtrained

The first version used four DS1 validation patients and selected epoch 33.

Patient-grouped 5-fold cross-validation across all 22 DS1 patients showed that performance actually peaked around epochs 3-7. v3 therefore trains for a fixed four epochs.

That change alone raised DS2 macro-F1 from:

- **0.403 ± 0.031** to
- **0.445 ± 0.007**

### 2. The CNN was not using timing well

On its training patients, the CNN could classify S beats almost entirely from waveform shape, so it had little reason to use the RR features. Those shape patterns did not transfer well to new patients.

I added a separate logistic-regression timing model using the four RR features normalized by the patient's recent average RR.

Combining the CNN and timing model produced:

| | v2 | v3 CNN | v3 CNN + timing |
|---|---:|---:|---:|
| Macro-F1 | 0.403 ± 0.031 | 0.445 ± 0.007 | **0.499** |
| S sensitivity | 0.056 | 0.014 | **0.325** |
| V sensitivity | 0.864 | 0.816 | **0.909** |

This also changed how I interpret the enrollment result. With the better v3 CNN, 60 seconds of label-free prototype enrollment improves macro-F1 by only **+0.036** instead of +0.074.

I have not yet combined the prototype enrollment method with the timing model.

## On-device

The Core ML / Swift implementation lives in `swift/ProtoHead`, and the iPhone app is in `ios/ECGReplay`.

For v3, `V3Classifier` runs:

- the Core ML embedding model
- the CNN head in Swift
- the RR timing model in Swift

On all 22 DS2 records, the Mac implementation matches Python on the same embeddings. End to end, only 6 of 49,668 predictions differ because of FP16 rounding.

On an iPhone 15 Pro, record 214 gives the same predictions as the Mac. The full v3 path runs at about **0.152 ms per beat** on CPU in a Debug build.

The v2 experiments also tested:

- on-device prototype enrollment
- `MLUpdateTask` head fine-tuning
- FP16 vs 6-bit palettization
- CPU / GPU / Neural Engine placement

The detailed results are in [`REPORT.md`](REPORT.md).

## Key takeaways

- **Patient-level validation changed the conclusion.** Four held-out patients selected epoch 33; grouped cross-validation across all DS1 patients pointed to about epoch 4.
- **Timing generalized better than waveform shape for S beats.** A separate RR model improved S sensitivity much more than the CNN alone.
- **Naive label-free fine-tuning was a bad fit.** Treating every enrollment beat as normal pushed the classifier toward N and hurt V sensitivity.
- **Prototype adaptation was more stable.** It adjusted the normal class without collapsing the rest of the classifier.
- **Part of the original personalization gain was really a training problem.** Once the population model improved, the enrollment gain became smaller.
- **The model is easy to run on-device.** The full v3 pipeline is well under 1 ms per beat on the tested iPhone.
- **6-bit compression changed very little.** It roughly halved the v2 model size while keeping the enrollment result almost the same.

## Running it

```bash
pip install -r requirements.txt

python prep.py
python train.py
python enroll.py
python convert.py
python compressed.py
python updatable.py
```

### v3

```bash
python prep.py --out data_norm --rr-norm long
python cv.py --name base
python train.py --select last --epochs 4 --out runs/v3_baseline
python fusion.py --seeds 0 1 2
python fusion.py --test
```

### Swift parity checks

```bash
swift build -c release --package-path swift/ProtoHead

python replay.py export --rec 214

swift/ProtoHead/.build/release/replay \
  runs/replay/214.json \
  runs/coreml/ecg_embedding_fp16.mlpackage \
  runs/enroll/proto_constants.json \
  runs/replay/214_swift.json

python replay.py check --rec 214

swift/ProtoHead/.build/release/finetune \
  runs/replay/214.json \
  runs/coreml/ecg_embedding_fp16.mlpackage \
  runs/coreml/ecg_head_updatable.mlmodel \
  runs/replay/214_finetune.json

python replay.py finetune --rec 214
```

The iPhone app is in `ios/ECGReplay`. It expects `214.json`, generated with `python replay.py export --rec 214`, to be added to the app target.

## Prior work

Patient-specific ECG classification is already well studied. The point of this project is not that personalization itself is new. I was interested in measuring it with a strict patient split, testing a no-label enrollment setup, and checking whether the same pipeline could actually run on-device.

- P. de Chazal, M. O'Dwyer, R. B. Reilly. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE TBME 51(7), 2004.
- S. Kiranyaz, T. Ince, M. Gabbouj. *Real-time patient-specific ECG classification by 1-D convolutional neural networks.* IEEE TBME 63(3), 2016.
