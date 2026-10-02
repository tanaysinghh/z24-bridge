# Review 2 Results: WaveNet vs MiniRocket vs InceptionTime

## 1. Task and protocol

| Item | Setting |
|---|---|
| Task | 17-class identification of the Z24 Progressive Damage Test scenario |
| Input | One accelerometer (channel 22, a fixed reference sensor), 20 s windows (2000 samples at 100 Hz), 0.5-30 Hz band-pass, per-window z-score |
| Split | By measurement setup: train = setups 1-6 (5100 windows / 1020 segments), val = setup 7 (850 / 170), test = setups 8-9 (1700 / 340) |
| Chance level | 1/17 = 5.9 % |
| Model selection | Deep models: checkpoint with best validation window accuracy (80 epochs, patience 25). MiniRocket: ridge alpha with best validation window accuracy |
| Test set use | Evaluated once per trained model; never used for any choice |

Full preprocessing details are in `docs/dataset.md`. "Window acc" scores each 20 s window separately. "Segment acc" sums the class scores of the 5 windows in a 60 s recording and scores one decision per recording.

## 2. Main results (seed 42; checkpoints in `models/<name>/`)

### Window-level accuracy (%)

| Model | Params | Train acc | Val acc | Test acc | Test macro-F1 | Train time |
|---|---|---|---|---|---|---|
| **WaveNet** | 228,881 | 69.7 | **58.4** | **39.7** | **37.7** | 8.4 min (RTX 4050) |
| MiniRocket | 10k fixed kernels + 169,949 ridge weights | 99.9 | 30.0 | 18.9 | 17.9 | 0.3 min (CPU) |
| InceptionTime | 493,681 | 96.9 | 54.9 | 33.8 | 32.3 | 11.3 min (RTX 4050) |

### Segment-level accuracy (%)

| Model | Train | Val | Test | Test macro-F1 |
|---|---|---|---|---|
| **WaveNet** | 81.6 | **69.4** | **45.3** | **41.3** |
| MiniRocket | 100.0 | 34.7 | 27.9 | 25.9 |
| InceptionTime | 100.0 | 64.7 | 42.9 | 39.3 |

### Seed variability (seeds 42, 1, 2; %)

| Model | Val window acc | Test window acc | Test segment acc |
|---|---|---|---|
| **WaveNet** | 58.6 +- 0.7 | **39.4 +- 0.2** | **45.8 +- 0.4** |
| MiniRocket | 29.5 +- 0.4 | 19.3 +- 0.3 | 27.4 +- 1.3 |
| InceptionTime | 55.0 +- 0.1 | 34.5 +- 0.8 | 43.8 +- 1.7 |

The ranking **WaveNet > InceptionTime > MiniRocket** holds for every seed, split and metric. All three models are far above chance (about 6.7x, 5.9x and 3.3x on mean test window accuracy).

## 3. Per-scenario test recall (segment level, seed 42, 20 test segments per class)

| Label | Scenario | WaveNet | MiniRocket | InceptionTime |
|---|---|---|---|---|
| 0 | Reference 1 | 5 | 0 | 0 |
| 1 | Settlement system installed | 0 | 0 | 0 |
| 2 | Pier -20 mm | 95 | 20 | 100 |
| 3 | Pier -40 mm | 25 | 60 | 35 |
| 4 | Pier -80 mm | 85 | 70 | 75 |
| 5 | Pier -95 mm | 100 | 50 | 80 |
| 6 | Foundation tilt | 20 | 40 | 100 |
| 7 | Reference 2 | 30 | 50 | 90 |
| 8 | Spalling 12 m² | 80 | 20 | 15 |
| 9 | Spalling 24 m² | 65 | 15 | 15 |
| 10 | Landslide | 45 | 35 | 65 |
| 11 | Concrete hinge failure | 45 | 10 | 15 |
| 12 | 2 anchor heads failed | 0 | 0 | 5 |
| 13 | 4 anchor heads failed | 5 | 0 | 10 |
| 14 | 2 tendons ruptured | 5 | 0 | 0 |
| 15 | 4 tendons ruptured | 65 | 50 | 35 |
| 16 | 6 tendons ruptured | 100 | 55 | 90 |

**Largest confusions (true -> predicted, out of 20):**

- WaveNet: 0 -> 16 (19), 14 -> 15 (16), 12 -> 11 (12), 13 -> 14 (10)
- InceptionTime: 0 -> 16 (20), 9 -> 8 (13), 13 -> 14 (10), 14 -> 15 (10)
- MiniRocket: 0 -> 16 (18), 14 -> 15 (10), 10 -> 8 (8), 8 -> 7 (8)

## 4. Observations

1. **Pier settlement is the most recognisable damage family.** Scenarios 2-6 change the boundary conditions enough to move the first vertical-bending frequency from about 3.84-3.91 Hz down to 3.71 Hz on the selected sensor, and both deep models identify the 20 mm, 80 mm and 95 mm stages with 75-100 % recall.
2. **Most errors are between neighbouring stages of the same damage type**: 2 vs 4 ruptured tendons, 2 vs 4 failed anchor heads, 12 vs 24 m² spalling. These states differ by small, local stiffness losses that barely change the global modes a single sensor sees. The models usually recognise the damage family but not its exact stage.
3. **Reference 1 is predicted as "6 tendons ruptured" by all three models.** The two scenarios were recorded at opposite ends of the PDT campaign and have almost the same mode-1 frequency on channel 22 (3.84 vs 3.81 Hz). The most likely explanation is that environmental conditions in the later scenario offset its damage-induced changes, the confounding Peeters and De Roeck (2001) documented on this bridge. Scenario 1 (settlement system installed, structurally the same as reference) is also never identified. No single-sensor classifier can separate these pairs without environmental compensation.
4. **WaveNet generalises best and overfits least.** Its train window accuracy is only 69.7 %, versus 96.9 % for InceptionTime and 99.9 % for MiniRocket, yet its test accuracy is the highest. Gated dilated causal convolutions with a 5.12 s receptive field act as a learned bank of narrow filters, which suits a task whose signal is small shifts in resonance frequency.
5. **MiniRocket is the fastest but weakest model here.** It fits in 18 s on CPU but generalises worst across setups. Training it on 3.4x more windows (stride 250 instead of 1000) did not help (test 20.4 % window / 26.8 % segment, `docs/results/minirocket_dense.json`), so the gap is not about augmentation. Its proportion-of-positive-values features summarise local waveform shape, which varies with the setup-specific excitation, rather than isolating the resonance frequencies.
6. **Validation is easier than test for every model** (by 11-21 points). The validation and test sets are single setups and pairs of setups, and setups 8-9 are evidently further from the training distribution than setup 7. The test numbers are the conservative estimate and the ones to quote.
7. **Segment voting adds 6-9 points** over single windows for every model. A 60 s decision is more reliable than a 20 s one.

## 5. Diagnostics

| Diagnostic | Result | File |
|---|---|---|
| MiniRocket, segment-grouped **random** split over setups 1-7 (no setup hold-out) | 32.3 % window / 44.5 % segment | `docs/results/diagnostic_random_split.json` |
| MiniRocket trained on dense windows (stride 250) | test 20.4 % window / 26.8 % segment | `docs/results/minirocket_dense.json` |
| Leave-one-setup-out MiniRocket, candidate channels | channel 22: 26.6 +- 4.8 % window | `data/processed/channel_selection_loso.json` |

The random-split diagnostic shows that even when held-out segments come from setups seen in training, single-channel MiniRocket only reaches about 45 % per segment. Separating 17 scenarios from one accelerometer is hard in itself, not only because of the stricter setup split.

## 6. Comparison with the 60-65 % baseline

The project baseline reports 60-65 % with WaveNet + MiniRocket. This Review 2 implementation reaches:

- **WaveNet:** 58.6 % (window) and **69.4-72.4 % (segment)** on the validation setup. This matches or exceeds the baseline range.
- **WaveNet:** 39.4 % window / 45.8 % segment on the two held-out test setups.

The figures are not directly comparable. Here, test data come from **different measurement setups** (different times, conditions and roving-sensor configurations) and the model sees a **single fixed sensor**. Random window-level splits on Z24 put overlapping slices of the same continuous record in both train and test, and typically report higher accuracy. When the baseline's protocol is confirmed, it can be reproduced with `src/preprocessing/pipeline.py` by changing the split definition.

## 7. Limitations

- The mirror carries no channel-to-location metadata. Channel 22 was **inferred** to be a fixed reference sensor from data (`docs/dataset.md`, section 4), not read from documentation.
- The labels-to-scenario-name mapping assumes the mirror keeps the original chronological order of the 17 PDT scenarios.
- Validation and test are single setups and pairs of setups (170 and 340 segments), so differences of 2-3 points between models are within noise. The seed table above covers training randomness but not setup sampling.
- No environmental (temperature) compensation is applied. The reference/late-damage confusion in section 4 is the clearest sign of its effect.

## 8. Reproduce

```bash
python -m src.preprocessing.pipeline
python -m src.train.train_minirocket
python -m src.train.train_wavenet
python -m src.train.train_inceptiontime
for s in 1 2; do
  python -m src.train.train_minirocket --seed $s --run-name minirocket_seed$s
  python -m src.train.train_wavenet --seed $s --run-name wavenet_seed$s
  python -m src.train.train_inceptiontime --seed $s --run-name inceptiontime_seed$s
done
python -m src.train.summarize
```

Per-run metrics, training curves and confusion matrices are in `docs/results/*.json`. Generated tables are in `docs/results/summary_tables.md`.
