# Experimental Report

This file contains the full experimental details for the on-device personalized ECG classifier.

The short version is in [`README.md`](README.md). This report keeps the intermediate results, controls, validation details, Core ML parity checks, compression experiments, and the parts of the project that did not work.

> **Research prototype, not a medical device.** All iPhone experiments replay MIT-BIH records. There is no live ECG sensor in this project.

## 1. Research question

A heartbeat classifier trained across a population usually performs worse when it is tested on a patient it has never seen.

I wanted to measure that drop under an inter-patient split, then test whether a short enrollment period could adapt the model to a new patient without cardiologist labels.

The project eventually became three related questions:

1. How well does a population model generalize to unseen patients?
2. Does label-free per-patient enrollment help?
3. Can the same adaptation pipeline run accurately and quickly on an iPhone?

A later round of experiments also checked whether the apparent personalization gain was actually caused by patient shift or partly by the way the original model was trained.

## 2. Dataset and split

I use the MIT-BIH Arrhythmia Database, lead MLII, sampled at 360 Hz.

Classes follow the AAMI grouping:

- N
- S
- V
- F
- Q

Paced records 102, 104, 107, and 217 are excluded.

### Inter-patient split

I use the de Chazal DS1 / DS2 split:

- DS1: training and model selection
- DS2: final testing
- no record appears in both

This matters because a random beat-level split lets beats from the same patient appear in both train and test data. That can produce results around 99%, but it does not measure generalization to a new patient.

All main test results below are on DS2.

## 3. Preprocessing and features

### ECG preprocessing

- baseline wander removed with 200 ms and 600 ms median filters
- 256 samples per beat
- R peak placed at index 100
- each beat z-scored independently

### RR features

Each beat also gets four timing features:

1. pre-RR
2. post-RR
3. mean RR over the previous 10 beats
4. pre-RR / local RR ratio

The timing features matter especially for S beats, where rhythm can be more informative than local morphology.

## 4. Model

The waveform model uses:

- 4 convolution blocks
- global average pooling
- 64-dimensional embedding
- one linear classification layer over `[embedding, RR]`

Training uses weighted cross-entropy with weights proportional to `1 / sqrt(class count)`.

The original v2 epoch was selected using four held-out DS1 records:

- 106
- 118
- 124
- 223

DS2 was not used to choose the epoch or tune the model.

### Metrics

The primary metric is macro-F1 over N, S, V, and F.

Q is not included in the main macro-F1 because DS2 contains only seven Q beats.

I also report per-class:

- sensitivity
- PPV

I do not use accuracy as the headline metric because about 89% of DS2 beats are N.

---

# Part I: v2 population model and enrollment

## 5. Population model

The v2 population model reaches **0.407 macro-F1** on all DS2 beats.

| Class | Sensitivity | PPV | Beats |
|---|---:|---:|---:|
| N | 0.832 | 0.957 | 44,218 |
| S | 0.084 | 0.039 | 1,836 |
| V | 0.856 | 0.571 | 3,219 |
| F | 0.005 | 0.001 | 388 |

The main weakness is S.

Record 232 contains 1,381 of the 1,836 S beats in DS2. The model gets almost none of them correct:

- 1,013 are predicted as N
- 363 are predicted as V

That happens even though the RR ratio is clearly different within the record:

- median S RR ratio: 0.74
- median N RR ratio: 1.76

If record 232 is left out, S sensitivity rises to 0.327.

This became an important warning for later results because one patient has a very large influence on the S metrics.

## 6. Enrollment setup

For every DS2 patient, enrollment uses the first:

- 30 seconds
- 60 seconds
- 2 minutes
- 5 minutes

Testing starts after the 5-minute mark.

That means:

- every enrollment length is evaluated on the same test beats
- enrollment beats never overlap the test set

The backbone stays frozen for both adaptation methods.

## 7. Adaptation methods

### 7.1 Linear-head fine-tuning

The first method fine-tunes only the final linear head.

Settings:

- full-batch SGD
- 30 steps
- no regularizer
- backbone frozen

This was chosen because the same setup can be reproduced on-device with `MLUpdateTask`.

### 7.2 Prototype head

The prototype method represents each class by the mean `[embedding, RR]` vector from DS1 training data.

Features are scaled using the DS1 standard deviation.

During enrollment, the N prototype is blended toward the new patient's enrollment mean using weight alpha.

Alpha was selected on DS1 validation patients:

`alpha = 0.5`

Classification is then nearest-prototype.

## 8. Enrollment labels

I evaluated two label conditions.

### Realistic

Every enrollment beat is labeled N.

This reflects the intended deployment case: the phone sees beats but does not get cardiologist annotations during enrollment.

### Oracle

Enrollment uses the true MIT-BIH labels.

This is an upper bound, not a realistic deployment condition.

### Random-label control

I also tested random labels at 60 seconds.

The control should not improve performance, and it does not.

## 9. Recovered share

For some results I report:

`recovered share = (enrolled - population) / (oracle 5 min - population)`

This is calculated from pooled macro-F1, with each method using its own population baseline and oracle ceiling.

Because the oracle ceiling can be small, the recovered percentage can sound more impressive than the absolute improvement. The macro-F1 change should be read alongside it.

## 10. v2 enrollment results

| Condition | Macro-F1 | Change | V sens | S sens | Recovered |
|---|---:|---:|---:|---:|---:|
| Prototype, population | 0.402 | | 0.908 | 0.084 | |
| Prototype, realistic 60 s | 0.476 | **+0.074** | 0.874 | 0.057 | 77% |
| Prototype, oracle 5 min | 0.499 | +0.097 | 0.945 | 0.139 | 100% |
| Prototype, random labels | 0.368 | -0.034 | 0.895 | 0.138 | |
| Fine-tune, population | 0.411 | | 0.853 | 0.094 | |
| Fine-tune, realistic 60 s | 0.380 | **-0.030** | 0.350 | 0.054 | -14% |
| Fine-tune, oracle 5 min | 0.628 | +0.217 | 0.835 | 0.646 | 100% |
| Fine-tune, random labels | 0.255 | -0.156 | 0.810 | 0.172 | |

`enroll.py` also reports every condition with record 232 excluded. Nothing is selected based on that analysis; it is reporting only.

The realistic prototype gain still holds without 232:

**+0.071 macro-F1**

![Enrollment curve](figures/enrollment_curve.png)

![Per-patient result at 60 seconds](figures/per_patient_60s.png)

## 11. What the v2 enrollment result actually does

The prototype method improves macro-F1, but it does not mainly do that by detecting more arrhythmias.

The main effect is fewer false positives.

For example, V PPV increases from about 0.50 to about 0.89, while S sensitivity falls slightly.

That is why I consider **+0.074 macro-F1** a clearer summary than "77% of the recoverable gap." The 77% is relative to a fairly small oracle ceiling of +0.097.

## 12. Why fine-tuning fails without labels

With realistic enrollment, every observed beat is labeled N.

Fine-tuning the linear head directly on those labels pushes the model toward predicting N more often.

The largest visible effect is on V:

- V sensitivity before fine-tuning: about 0.85
- V sensitivity after realistic 60 s fine-tuning: about 0.35

The prototype method is less destructive because it changes only the N prototype instead of updating all class boundaries.

## 13. Enrollment length

The prototype enrollment curve is nearly flat from 30 seconds to 5 minutes.

About 30 seconds of beats is already enough to estimate the patient's N mean well enough for this adaptation method.

## 14. Effect of record 232 on the oracle result

Record 232 also drives much of the fine-tuning oracle ceiling.

Without record 232:

- oracle fine-tune S sensitivity: 0.197
- population S sensitivity: 0.331

So most of the large oracle S improvement comes from learning record 232 specifically.

---

# Part II: v3

## 15. Why I revisited the model

v2 detected very few S beats on unseen patients.

Its S sensitivity was around 0.08, and the enrollment result looked surprisingly strong.

I switched from the four-patient validation set to **patient-grouped 5-fold cross-validation across all 22 DS1 patients** using `cv.py` and `fusion.py`.

DS2 remained untouched until the settings were fixed.

This exposed two separate problems.

## 16. v2 was trained too long

Across the grouped CV folds and three seeds:

- macro-F1 peaks around epochs 3-7 at about 0.393
- by epochs 30-36 it falls to about 0.365

The original four-patient validation split had selected epoch 33.

v3 keeps the same CNN architecture but trains it for a fixed four epochs.

On DS2, over three seeds:

- v2: **0.403 ± 0.031**
- v3 CNN: **0.445 ± 0.007**

V F1 improves from roughly 0.65 to 0.80.

S sensitivity, however, gets even worse at about 0.01.

## 17. Changes tried but not kept

I tested several other changes and did not keep them because they did not produce a consistent cross-validation gain over three seeds:

- shift augmentation
- scale augmentation
- noise augmentation
- dropout on the embedding
- per-patient RR normalization fed directly to the CNN
- `1 / class_count` weighting

## 18. The CNN was mostly ignoring timing

On its training patients, the CNN gets about 99.9% S sensitivity from waveform shape alone.

That makes it possible for the network to largely ignore the RR features even though timing is important for S beats.

The problem is that waveform shape does not transfer as well across patients.

de Chazal reported roughly 76% S sensitivity with a linear model using mostly RR features, which suggested separating the timing pathway from the CNN.

## 19. Timing model

v3 adds a separate logistic-regression timing model using the four RR features.

The RR features are normalized by the patient's own average RR over the previous five minutes.

That calculation is causal and can be used in a live setting.

The timing model is class-balanced.

At inference time, its log-probabilities are added to the CNN log-probabilities.

The timing-model weight is 3.

It was selected by DS1 cross-validation from values between 0 and 100.

DS1 CV results:

- CNN alone: 0.404
- timing alone: 0.484
- combined: 0.545

## 20. v3 DS2 results

The CNN numbers below use three seeds. The timing model is the same for all of them.

| | v2 | v3 CNN alone | v3 CNN + timing |
|---|---:|---:|---:|
| Macro-F1 | 0.403 ± 0.031 | 0.445 ± 0.007 | **0.499** (0.477 / 0.488 / 0.532) |
| S sensitivity | 0.056 | 0.014 | **0.325** (0.22 / 0.31 / 0.45) |
| S PPV | ~0.04 | ~0.04 | **~0.30** |
| V sensitivity | 0.864 | 0.816 | **0.909** |
| V PPV | ~0.57 | ~0.77 | 0.62 |

The S improvement is not only record 232.

For record 232:

- v2 S sensitivity: 0.004
- v3 S sensitivity: about 0.19-0.43 depending on seed

With record 232 removed:

- v3 S sensitivity: about 0.30-0.53
- v2 S sensitivity: about 0.33
- v3 S PPV remains low at about 0.12

This is still below de Chazal's reported ~76% S sensitivity. Their setup used both ECG leads; this project uses MLII only.

## 21. What v3 changes about the enrollment result

On the stronger v3 CNN, 60 seconds of realistic prototype enrollment changes macro-F1 from:

`0.400 -> 0.436`

That is a gain of **+0.036**, compared with +0.074 on v2.

The enrolled prototype head also no longer beats the plain linear head:

- enrolled prototype: 0.436
- plain linear head: 0.442

The interpretation is different now.

A meaningful part of the original +0.074 gain was fixing errors caused by v2 overtraining, especially false alarms on N beats. It was not all evidence of patient-specific distribution shift.

I have not yet combined prototype enrollment with the v3 timing model.

---

# Part III: Core ML and iPhone implementation

## 22. On-device structure

On-device code lives in `swift/ProtoHead`.

That Swift package is used by:

- a Mac command-line tool for Python parity checks
- the iPhone app in `ios/ECGReplay`

The phone stores the embedding model. The classification logic then runs either:

- as a prototype head in Swift
- as an updatable Core ML head for fine-tuning

## 23. Core ML conversion

For the FP32 ML Program compared with PyTorch across all DS2 beats:

- maximum logit difference: `1.3e-5`
- argmax agreement: 100%

The phone uses the embedding backbone in:

- FP16
- 6-bit palettized form

## 24. v2 prototype head on-device

`EmbeddingModel` runs one beat at a time through the Core ML backbone.

`ProtoHead` then:

1. appends the four RR values
2. blends the N prototype toward the enrollment mean
3. classifies by nearest prototype

### Mac parity, all 22 DS2 records

Using FP16 on CPU:

- Swift embeddings match `coremltools` embeddings exactly
- Swift head and Python head give the same predictions on the same embeddings
- end to end, 10 of 41,459 test predictions differ from the PyTorch path
- those differences come from FP16 rounding
- per-record macro-F1 stays within 0.001

### iPhone 15 Pro

Using record 214:

- CPU-only replay gives the same results as the Mac
- all-compute-units mode gives the same per-class results
- N: 1613 / 1662
- V: 156 / 212
- 60 s enrollment contains 76 beats
- embedding those beats and updating the N prototype takes **6.3 ms**
- model load takes 127 ms

## 25. On-device fine-tuning with `MLUpdateTask`

`MLUpdateTask` only supports the older NeuralNetwork model format.

`updatable.py` therefore builds the classification head separately as:

`Linear 68 -> 5 -> softmax`

Only the linear layer is updatable.

Training settings:

- cross-entropy loss
- full-batch SGD
- 30 steps
- learning rate 0.01

The backbone remains in the ML Program embedding model.

### Mac parity

Across all 22 DS2 records for realistic@60 and oracle@300:

- updated Core ML weights differ from PyTorch by at most `6e-8`
- the update itself moves weights by about 0.1
- predictions on the same features are identical
- end to end, 25 of about 83,000 predictions differ because of FP16 embeddings
- macro-F1 stays within 0.004
- 60 s update time: about 30 ms
- 5 min update time: about 120 ms

### iPhone 15 Pro

Using record 214:

- updated weights stay within `6e-8` of PyTorch
- predictions match
- macro-F1 matches `enroll.py`
- 60 s update, 76 beats: **25.4 ms**
- 5 min update, 383 beats: **70 ms**
- 30 SGD steps in both cases

The training itself works on-device.

The issue is not implementation. The issue is that realistic all-N fine-tuning hurts the classifier, so it is not the adaptation method I would use.

## 26. v2 latency

Xcode performance report on iPhone 15 Pro, iOS 26.6.2, batch size 1.

Times are median / p95 over 120 predictions, from 40 predictions across three runs.

| Compute units | FP16 (62.9 KB) | 6-bit palettized (32.5 KB) |
|---|---:|---:|
| All | 0.040 / 0.062 ms | 0.039 / 0.062 ms |
| CPU only | 0.044 / 0.067 ms | 0.042 / 0.070 ms |
| CPU + GPU | 0.044 / 0.061 ms | 0.045 / 0.064 ms |
| CPU + Neural Engine | 0.042 / 0.066 ms | 0.045 / 0.072 ms |

Model load is around 6.5-6.8 ms in each setting.

## 27. Neural Engine placement

Core ML did not place any of the model's 17 operations on the Neural Engine.

That remained true for:

- all compute units
- CPU + Neural Engine
- CPU + GPU

Every operation stayed on the CPU.

The model is small enough that the scheduler appears to prefer keeping the work on the CPU rather than paying the overhead of moving it elsewhere.

## 28. Compression

`compressed.py` embeds every DS2 beat using both the FP16 and 6-bit Core ML models, batch size 1 on CPU.

The prototype head then uses the same constants shipped to the phone.

Each variant gets its own population baseline and oracle@300 ceiling.

| Variant | Size | Population | Realistic 60 s | Change | Recovered | Oracle 5 min |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | | 0.402 | 0.476 | +0.074 | 76.7% | 0.499 |
| Core ML FP16 | 62.9 KB | 0.402 | 0.476 | +0.074 | 76.8% | 0.499 |
| Core ML 6-bit | 32.5 KB | 0.406 | 0.477 | +0.071 | 73.5% | 0.503 |

The 6-bit model is about half the size.

The embeddings can move by as much as 0.24, but the enrollment result changes very little:

- FP16: +0.074
- 6-bit: +0.071

Latency is also nearly unchanged.

## 29. v3 on-device

`V3Classifier` in `swift/ProtoHead` runs:

- the v3 Core ML embedding model
- the CNN final layer in Swift
- the timing model in Swift

`fusion.py --export` writes the constants used by the Swift implementation.

### Mac parity

Across all 22 DS2 records:

- zero disagreements when Python and Swift use the same embeddings
- 6 of 49,668 beats differ end to end because of FP16
- per-record macro-F1 stays within 0.0007

### iPhone 15 Pro

On record 214:

- predictions are identical to the Mac
- full v3 pipeline: about **0.152 ms per beat**
- CPU
- Debug build

The full path here includes:

- Core ML embedding
- Swift CNN head
- Swift timing model

The Swift math is not optimized, so this should be treated as a functional timing measurement rather than a tuned performance ceiling.

---

# Part IV: Takeaways

## 30. Patient-level validation mattered more than I expected

The four-patient validation setup selected epoch 33.

Grouped cross-validation across all 22 DS1 patients pointed to roughly epoch 4.

That changed both the baseline performance and the story around personalization.

## 31. The original enrollment gain was partly fixing overtraining

v2 made label-free prototype enrollment look like it recovered a large part of patient-specific performance loss.

After the training procedure was fixed, the gain became much smaller.

So the original result mixed together at least two effects:

- adaptation to the new patient
- correction of errors from an overtrained population model

## 32. RR timing needed its own path

Simply concatenating RR features with the learned embedding did not guarantee that the CNN would use them.

The network could almost perfectly separate S beats on its training patients using morphology, so it learned a shortcut that did not generalize well.

A separate timing model forced the RR information to contribute.

## 33. Label-free fine-tuning is too aggressive here

If all enrollment beats are treated as N, direct fine-tuning changes all of the class boundaries.

That caused a large drop in V sensitivity.

The prototype method is safer because it changes only the representation of the normal class.

## 34. Prototype enrollment mostly reduced false positives

The v2 prototype improvement came mainly from improving PPV, especially for V.

It did not produce a comparable increase in arrhythmia sensitivity.

## 35. Very little enrollment data was needed

The prototype curve was almost flat from 30 seconds to 5 minutes.

For this method, the N mean is already estimated reasonably well from a short window.

## 36. Compression was not a real tradeoff

The 6-bit model was about half the size of FP16 and produced almost the same enrollment result and latency.

For a model this small, compression was easy to keep.

---

# Part V: Implementation notes

## 37. Issues that came up

### Partial PhysioNet download

PhysioNet returned a 502 halfway through a download.

The original preparation script checked for only one file before deciding that the dataset was already present. A rerun therefore skipped the download and later crashed.

The script now checks all required records.

### Missing compression dependency

6-bit palettization uses scikit-learn for k-means.

It was missing from the requirements.

### `MLUpdateTask` batch-size restriction

`MLUpdateTask` initially rejected full-batch training because the model spec only allowed the mini-batch size used when the model was created.

The number of enrollment beats varies by patient.

The fix was to allow a batch-size range from 1 to 4096 in the model spec.

### `.gitignore` and `.mlpackage`

A `data/` rule in `.gitignore` also matched the `Data/` directory inside `.mlpackage` bundles on macOS because paths are case-insensitive.

That caused model weights to be left out of Git without an obvious error.

---

# Part VI: Reproducing the experiments

## 38. Python

```bash
pip install -r requirements.txt

python prep.py         # download MIT-BIH + preprocess
python train.py        # v2 population model
python enroll.py       # enrollment experiments
python convert.py      # Core ML conversion + parity checks, macOS
python compressed.py   # FP16 vs 6-bit enrollment
python updatable.py    # build updatable head for MLUpdateTask
```

## 39. v3

```bash
python prep.py --out data_norm --rr-norm long

python cv.py --name base

python train.py \
  --select last \
  --epochs 4 \
  --out runs/v3_baseline

python fusion.py --seeds 0 1 2
python fusion.py --test
```

`fusion.py --seeds 0 1 2` is used to choose the timing model and fusion weight on DS1 cross-validation.

`fusion.py --test` performs the final DS2 evaluation.

## 40. Swift checks against Python

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

The iPhone app is in `ios/ECGReplay`.

It needs `214.json`, produced by:

```bash
python replay.py export --rec 214
```

The file is not stored in the repository and needs to be added to the app target.

---

# Prior work

Patient-specific ECG classification is not a new idea. The main contribution of this project is the combination of a strict inter-patient evaluation, label-free enrollment experiments, and an on-device implementation with parity checks against Python.

- P. de Chazal, M. O'Dwyer, R. B. Reilly. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE Transactions on Biomedical Engineering, 51(7), 2004.
- S. Kiranyaz, T. Ince, M. Gabbouj. *Real-time patient-specific ECG classification by 1-D convolutional neural networks.* IEEE Transactions on Biomedical Engineering, 63(3), 2016.
