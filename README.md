# On-Device Personalized ECG Classifier

A heartbeat classifier trained on other people's hearts does worse on a new person's heart.
This project measures that gap on MIT-BIH with a proper inter-patient split, then tests whether a
short per-patient enrollment, using no cardiologist labels, closes part of it, and runs that
enrollment on an iPhone with Core ML.

**Research prototype, not a medical device.** Everything on the phone is a **replayed** MIT-BIH
record, not a live sensor.

## Results at a glance

| | |
|---|---|
| Population model on unseen patients (DS2) | macro-F1 0.407 (v2) → **0.499** with CNN + timing model (v3, 3 seeds) |
| S / V sensitivity, v2 → v3 | S 0.06 → **0.33**, V 0.86 → **0.91** (3 seeds) |
| 60 s enrollment, no labels, prototype head (v2) | **+0.074 macro-F1**, 77% of the recoverable gap, but see [v3](#v3-fixing-overtraining-and-adding-a-timing-model): part of it was v2 overtraining |
| 60 s enrollment, no labels, head fine-tune | −0.030 (hurts, see below) |
| Same enrollment with the 6-bit model | +0.071, 74% of the gap |
| Enrollment on iPhone 15 Pro (60 s of beats) | **6.3 ms** prototype, 25 ms `MLUpdateTask` fine-tune |
| Inference per beat on iPhone 15 Pro | about 0.04 ms, FP16 and 6-bit |
| Model size | about 62 KB FP16, 32 KB 6-bit |
| Swift / Core ML vs Python | same predictions (differences only from FP16 rounding) |

## The problem and the leakage trap

If you split MIT-BIH beats at random, the same patient ends up in train and test and you score
around 99%. That number measures memorizing patients, not generalizing. Here the split is the
de Chazal inter-patient one: DS1 records train, DS2 records test, no record in both. Paced records
(102, 104, 107, 217) are left out. The numbers below are lower than the random-split ones you see
around, and that's the point.

## Setup

- MIT-BIH Arrhythmia Database, lead MLII, 360 Hz, AAMI classes N / S / V / F / Q.
- Baseline wander removed with 200 ms + 600 ms median filters.
- 256-sample window per beat (R peak at index 100), z-scored per beat.
- 4 RR features per beat: pre-RR, post-RR, mean RR of the last 10 beats, pre / local ratio.
  These go into the model. S beats are mostly told apart by timing, not shape.
- Model: 4 conv blocks → global average pool → 64-d embedding, then one linear layer on
  [embedding, RR]. Weighted cross-entropy (1/√count).
- Epoch picked on 4 held-out DS1 patients (106, 118, 124, 223). DS2 is never used to choose anything.
- Metric: macro-F1 over N, S, V, F (Q has 7 beats in DS2), plus per-class sensitivity and PPV.
  Never accuracy: 89% of beats are N.

## Step A: the gap

Population model on all DS2 beats. **Macro-F1 0.407.**

| Class | Sens | PPV | Beats |
|---|---|---|---|
| N | 0.832 | 0.957 | 44218 |
| S | 0.084 | 0.039 | 1836 |
| V | 0.856 | 0.571 | 3219 |
| F | 0.005 | 0.001 | 388 |

S is the weak class. Record 232 holds 1381 of the 1836 DS2 S beats, and the model gets almost
none of them (1013 go to N, 363 to V), even though their RR ratio is clearly short (median 0.74
vs 1.76 for that record's N beats). Leaving 232 out, S sensitivity is 0.327.

## Step B: enrollment

For each DS2 patient, enroll on the first 30 s / 60 s / 2 min / 5 min, and test on every beat
after 5 min. The test beats are the same for every length, and enrollment beats never touch them.

Two methods, backbone frozen in both:
- **Fine-tune:** plain full-batch SGD on the linear head only, 30 steps, no regularizer (exactly
  what `MLUpdateTask` can do on device).
- **Prototype head:** class prototypes = class means of [embedding, RR] on DS1 train, scaled by
  DS1 std. Enrollment blends the prototypes toward the patient's mean with weight α, then
  classify by nearest prototype. α picked on DS1 val patients (0.5).

Two label conditions:
- **Realistic:** every enrollment beat labeled N, since a phone gets no annotations. This is the
  headline.
- **Oracle:** true MIT-BIH labels. Upper bound.

Control: random labels at 60 s. It should not help, and it doesn't.

**Recovered share = (enrolled − population) / (oracle 5 min − population)**, on pooled macro-F1,
with each method's own baseline and ceiling.

| Condition (60 s) | Macro-F1 | Change | V sens | S sens | Recovered |
|---|---|---|---|---|---|
| Prototype, population | 0.402 | | 0.908 | 0.084 | |
| Prototype, realistic | 0.476 | **+0.074** | 0.874 | 0.057 | 77% |
| Prototype, oracle 5 min (ceiling) | 0.499 | +0.097 | 0.945 | 0.139 | 100% |
| Prototype, random labels | 0.368 | −0.034 | 0.895 | 0.138 | |
| Fine-tune, population | 0.411 | | 0.853 | 0.094 | |
| Fine-tune, realistic | 0.380 | −0.030 | 0.350 | 0.054 | −14% |
| Fine-tune, oracle 5 min (ceiling) | 0.628 | +0.217 | 0.835 | 0.646 | 100% |
| Fine-tune, random labels | 0.255 | −0.156 | 0.810 | 0.172 | |

`enroll.py` also prints every condition with record 232 left out (reporting only, nothing is
chosen on it). The prototype gain holds without it: +0.071.

![enrollment curve](figures/enrollment_curve.png)

![per patient at 60 s](figures/per_patient_60s.png)

## v3: fixing overtraining and adding a timing model

v2 detected almost no S beats on new patients (sensitivity 0.08). Two things were wrong, both
found with **patient-grouped 5-fold cross-validation on all 22 DS1 patients** (`cv.py`,
`fusion.py`) instead of the 4 validation patients. DS2 was only used once, at the end, and the
settings were fixed before that.

**1. v2 was overtrained.** Pooled over folds and 3 seeds, the CV curve peaks at 3-7 epochs
(macro-F1 0.393) and is lower by 30-36 epochs (0.365). v2's 4-patient validation picked epoch 33.
v3 trains the same model for a fixed 4 epochs. On its own this raises DS2 macro-F1 from
0.403 ± 0.031 to 0.445 ± 0.007 (3 seeds) and V F1 from 0.65 to 0.80, but S gets even worse (0.01).

Tried and not kept (no CV gain over 3 seeds): shift/scale/noise augmentation + dropout on the
embedding, per-patient RR normalization fed to the CNN, 1/count class weights.

**2. The CNN ignores timing.** On its training patients it gets 99.9% S sensitivity from beat
shape alone, so it never learns to use the RR features, and shape doesn't carry over to new
patients. de Chazal got about 76% S with a linear model on mostly RR features. So v3 adds a
separate **timing model**: logistic regression on 4 RR features normalized by the patient's own
average RR over the past 5 minutes (causal, works live), class-balanced. Its log-probabilities are
added to the CNN's with weight 3 (picked by CV from 0 to 100; CNN alone 0.404, timing alone
0.484, combined 0.545 on DS1 CV).

DS2, 3 CNN seeds (the timing model is the same for all):

| | v2 | v3 CNN alone | **v3 CNN + timing** |
|---|---|---|---|
| Macro-F1 | 0.403 ± 0.031 | 0.445 ± 0.007 | **0.499** (0.477 / 0.488 / 0.532) |
| S sens | 0.056 | 0.014 | **0.325** (0.22 / 0.31 / 0.45) |
| S PPV | ~0.04 | ~0.04 | **0.30** |
| V sens | 0.864 | 0.816 | **0.909** |
| V PPV | ~0.57 | ~0.77 | 0.62 |

The S gain isn't only record 232 (0.004 → 0.19-0.43): without 232, S sensitivity is 0.30-0.53
(v2: 0.33), with low PPV (about 0.12). Still well below de Chazal's 76%, which used both ECG
leads; we use MLII only.

**What this means for enrollment.** On the better v3 CNN, 60 s of label-free prototype enrollment
adds only +0.036 (0.400 → 0.436), and the enrolled prototype head no longer beats the plain linear
head (0.442). A good part of v2's +0.074 was undoing v2's own overtraining (false alarms on N
beats), not patient differences. Combining enrollment with the timing model is not done yet.

**v3 on the phone.** `V3Classifier` in `swift/ProtoHead` runs the v3 Core ML embedding model, the
CNN's last layer and the timing model in Swift (`fusion.py --export` writes the constants). On a
Mac it matches Python on all 22 DS2 records (0 disagreements on the same embeddings, 6 of 49,668
beats off end to end from FP16, per-record macro-F1 within 0.0007). On the iPhone 15 Pro, record
214 gives identical predictions to the Mac. The rest of "On the phone" below is v2.

## On the phone

Everything on-device lives in `swift/ProtoHead`, a Swift package used by both a Mac command line
tool (for checking against Python) and the iPhone app in `ios/ECGReplay`.

### Core ML conversion

- FP32 ML Program vs PyTorch on all DS2 beats: max logit difference 1.3e-5, argmax agreement 100%.
- The phone gets only the **embedding** model (backbone), FP16 and 6-bit palettized.
  The head runs in Swift (prototype) or as a separate updatable model (fine-tune).

### Prototype head (adapted on-device)

`EmbeddingModel` runs the Core ML embedding model one beat at a time, and `ProtoHead` appends the
4 RR values, does the at-rest enrollment (N prototype blended toward the enrollment mean) and
classifies by nearest prototype.

Checked on all 22 DS2 records on a Mac (FP16, CPU):
- Swift embeddings = coremltools embeddings exactly.
- Swift head vs Python head on the same embeddings: 100% same predictions.
- End to end vs the PyTorch path in `enroll.py`: 10 of 41,459 test beats differ (FP16 rounding),
  per-record macro-F1 within 0.001.

On an iPhone 15 Pro (iOS 26.6.2), record 214 bundled in the app:
- CPU only: `replay.py check` on the phone's saved output gives the same three results as the Mac.
- All compute units: same per-class results (N 1613/1662, V 156/212). Enrollment, 76 beats (60 s)
  embedded + N prototype update: **6.3 ms**. Model load 127 ms.

### Head fine-tune with `MLUpdateTask`

`MLUpdateTask` only works on the old NeuralNetwork format, so `updatable.py` builds the head
(Linear 68 → 5, softmax) as a NeuralNetwork with only the linear layer updatable, cross-entropy
loss and plain SGD. Swift (`UpdatableHead`) runs it full batch, 30 steps, lr 0.01, same as
`enroll.py`. The backbone stays in the ML Program embedding model.

Checked on all 22 DS2 records on a Mac, realistic@60 and oracle@300:
- Updated weights, Core ML vs PyTorch on the same features: max difference 6e-8 (the update
  itself moves them by about 0.1).
- Predictions on the same features: identical, 0 differences.
- End to end vs `enroll.py`: 25 of about 83,000 test predictions differ (FP16 embeddings),
  macro-F1 within 0.004.
- Update time on the Mac: about 30 ms for 60 s of beats, 120 ms for 5 min.

On an iPhone 15 Pro (iOS 26.6.2), record 214: `replay.py finetune` on the phone's saved output
gives the same results as the Mac (weights within 6e-8 of PyTorch, predictions identical, macro-F1
equal to `enroll.py`). The update itself takes **25.4 ms** for 60 s of beats (76) and 70 ms for
5 min (383), 30 SGD steps each, so the head is **trained on-device**.

It works, but it's the method that doesn't help without labels (−0.030 at 60 s), so the
prototype head is still the one to ship.

### Latency (Xcode performance report, iPhone 15 Pro, iOS 26.6.2, batch 1)

Predict time per beat, median / p95 over 120 predictions (40 x 3 runs):

| Compute units | FP16 (62.9 KB) | 6-bit palettized (32.5 KB) |
|---|---|---|
| All | 0.040 / 0.062 ms | 0.039 / 0.062 ms |
| CPU only | 0.044 / 0.067 ms | 0.042 / 0.070 ms |
| CPU + GPU | 0.044 / 0.061 ms | 0.045 / 0.064 ms |
| CPU + Neural Engine | 0.042 / 0.066 ms | 0.045 / 0.072 ms |

Model load is about 6.5-6.8 ms in every case.

### Does it survive compression?

`compressed.py` embeds every DS2 beat with the FP16 and 6-bit Core ML models (batch 1, CPU),
then runs the prototype head with the same constants the phone ships (made from the FP32 model).
Each variant gets its own population baseline and oracle@300 ceiling.

| Variant | Size | Population | Realistic 60 s | Change | Recovered | Oracle 5 min |
|---|---|---|---|---|---|---|
| PyTorch FP32 | | 0.402 | 0.476 | +0.074 | 76.7% | 0.499 |
| Core ML FP16 | 62.9 KB | 0.402 | 0.476 | +0.074 | 76.8% | 0.499 |
| Core ML 6-bit | 32.5 KB | 0.406 | 0.477 | +0.071 | 73.5% | 0.503 |

## What surprised me

- **Label-free enrollment helps the prototype head and hurts fine-tuning.** Fine-tuning on
  all-N labels teaches the head to call everything N, and V sensitivity falls from 0.85 to 0.35.
  Moving only the N prototype doesn't have that problem.
- **The prototype gain is fewer false alarms, not more arrhythmias found.** V PPV goes from 0.50 to
  0.89, and S sensitivity actually drops a bit. The 77% is of a small ceiling (+0.097), so +0.074
  macro-F1 is the more honest number.
- **The prototype curve is flat from 30 s to 5 min.** 30 seconds of beats already gives the N mean.
- **One record drives the fine-tune ceiling.** Without 232, the oracle fine-tune S sensitivity is
  0.197, below the population model's 0.331. Most of the oracle's S gain is learning 232.
- **The Neural Engine never gets used.** Every op is supported on it, but in all four settings,
  including forced CPU + Neural Engine and CPU + GPU, Core ML placed all 17 ops on the CPU for
  both models. The model is small enough that the scheduler decides the CPU is cheapest, so
  there's no Neural Engine number for it.
- **6-bit is free here.** It halves the size, moves embeddings by up to 0.24, and barely changes the
  enrollment gain (+0.071 vs +0.074) or the speed.

- **The biggest problem wasn't cross-patient shift, it was how I picked the epoch.** 4 validation
  patients picked epoch 33; cross-validation over all 22 DS1 patients says 4. And a CNN that can
  memorize shapes will ignore timing features even when you hand them to it.

## What broke

- PhysioNet returned a 502 halfway through the download, and the prep script only checked for one
  file, so a rerun skipped the download and crashed. Now it checks all records.
- 6-bit palettization needs scikit-learn (k-means), which wasn't in the requirements.
- `MLUpdateTask` refused full-batch training: the model spec only allowed the mini-batch size it
  was built with, and the number of enrollment beats is different for every patient. Fixed by
  allowing a range (1-4096) in the spec.
- A `data/` line in `.gitignore` also matched the `Data/` folder inside `.mlpackage` bundles
  (macOS paths are case-insensitive), so the model weights silently didn't get committed.

## Running it

```
pip install -r requirements.txt
python prep.py         # download MIT-BIH + preprocess
python train.py        # Step A
python enroll.py       # Step B
python convert.py      # Core ML + parity (macOS)
python compressed.py   # FP16 vs 6-bit enrollment
python updatable.py    # updatable head for MLUpdateTask

# v3
python prep.py --out data_norm --rr-norm long
python cv.py --name base                          # patient-grouped CV on DS1
python train.py --select last --epochs 4 --out runs/v3_baseline
python fusion.py --seeds 0 1 2                    # pick the timing model + weight on DS1 CV
python fusion.py --test                           # one DS2 check

# Swift checks against Python (macOS)
swift build -c release --package-path swift/ProtoHead
python replay.py export --rec 214
swift/ProtoHead/.build/release/replay runs/replay/214.json runs/coreml/ecg_embedding_fp16.mlpackage \
    runs/enroll/proto_constants.json runs/replay/214_swift.json
python replay.py check --rec 214
swift/ProtoHead/.build/release/finetune runs/replay/214.json runs/coreml/ecg_embedding_fp16.mlpackage \
    runs/coreml/ecg_head_updatable.mlmodel runs/replay/214_finetune.json
python replay.py finetune --rec 214
```

The iPhone app is `ios/ECGReplay`. It needs `214.json` (from `replay.py export`, not in the repo)
added to the app target.

## Prior work

This is not a new idea. Patient-specific ECG classification is well studied; the contribution
here is doing it carefully and running it on-device.

- P. de Chazal, M. O'Dwyer, R. B. Reilly. *Automatic classification of heartbeats using ECG
  morphology and heartbeat interval features.* IEEE TBME 51(7), 2004. (the inter-patient split)
- S. Kiranyaz, T. Ince, M. Gabbouj. *Real-time patient-specific ECG classification by 1-D
  convolutional neural networks.* IEEE TBME 63(3), 2016. (patient-specific 1D CNN, first 5 min
  of each record)
