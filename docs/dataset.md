# Dataset and Preprocessing Pipeline

## 1. Source

| Item | Value |
|---|---|
| Dataset | Z24 Bridge Progressive Damage Tests (KU Leuven), pre-processed mirror |
| Structure | Z24 Bridge, built 1963, between Koppigen and Utzenstorf near Solothurn, Switzerland. Post-tensioned concrete box girder, 58 m long over three continuous spans of 14 m + 30 m + 14 m (KU Leuven, bwk.kuleuven.be/bwm/z24) |
| Mirror | https://huggingface.co/datasets/thanglexuan/Z24-dataset-processed |
| Local path | `data/z24/inputs.npy`, `data/z24/labels.npy` (not tracked by git) |
| `inputs.npy` | `(1530, 27, 6000)`, float32, 991,440,128 bytes |
| `inputs.npy` SHA-256 | `56fb5923433e4b47cbd9c19ce079fc098f59d501b3385f9cb26636cc3928ef6e` (matches the Hub LFS object id) |
| `labels.npy` | `(1530,)`, int64, values 0..16, exactly 90 samples per class |

The mirror was derived from the original `(17 scenarios, 9 setups, 27 channels, 65530 samples)` array by trimming each record to 60,000 samples and cutting it into 10 consecutive 6,000-sample segments: `1530 = 17 x 9 x 10`.

## 2. Verified structure of the processed array

Nothing in the mirror documents the index ordering or the sampling rate, so both were checked empirically (`python -m src.preprocessing.channel_analysis`, output saved to `data/processed/channel_analysis.json`).

**Index ordering.** `labels[i] == i // 90` for all 1530 samples, so samples are scenario-major. Within each scenario block, the last sample of segment `k` joins smoothly onto the first sample of segment `k+1` (median jump 0.48x the mean sample-to-sample step). Every tenth boundary instead shows a large discontinuity (median 3.93x), which marks a change of setup. Rows within a sample are genuine channels: they are mutually correlated and each is continuous in time across segments. The full index is therefore

```
i = scenario * 90 + (setup - 1) * 10 + segment      scenario in 0..16, setup in 1..9, segment in 0..9
```

and the 10 segments of a setup form one continuous 600 s record.

**Sampling rate.** The dominant spectral peaks of the reference scenario fall at 3.91, 5.13, 9.96, 12.74, 16.2-17.3 and 25.7 Hz when the data is read at **fs = 100 Hz**. These match the published Z24 natural frequencies (about 3.9 Hz first vertical bending, 5.0 Hz first lateral, 9.8-10.3 Hz, 12.4-13.2 Hz), and 100 Hz is the documented acquisition rate of the KU Leuven PDT ambient tests. Each 6,000-sample segment therefore lasts **60 s**, and each setup record lasts 600 s.

## 3. Classes

The 17 labels are the 17 Progressive Damage Test scenarios. Assuming the mirror keeps the original chronological scenario order of the KU Leuven raw array (the mirror itself does not name them), they are:

| Label | Scenario |
|---|---|
| 0 | First reference measurement |
| 1 | After installation of the pier settlement system |
| 2 | Pier lowered 20 mm |
| 3 | Pier lowered 40 mm |
| 4 | Pier lowered 80 mm |
| 5 | Pier lowered 95 mm |
| 6 | Pier lifted, tilt of foundation |
| 7 | Second reference measurement |
| 8 | Spalling of concrete at soffit, 12 m² |
| 9 | Spalling of concrete at soffit, 24 m² |
| 10 | Landslide at abutment |
| 11 | Failure of concrete hinge |
| 12 | Failure of 2 anchor heads |
| 13 | Failure of 4 anchor heads |
| 14 | Rupture of 2 of 16 tendons |
| 15 | Rupture of 4 of 16 tendons |
| 16 | Rupture of 6 of 16 tendons |

The classes are perfectly balanced (90 segments each).

## 4. Channel selection

The project frames VibraDiagnose as a **single-accelerometer** system, so one of the 27 channels is used. **Channel 22** was selected.

### 4.1 Why the choice matters

In the Z24 PDT campaign the bridge was measured in 9 setups. A few reference accelerometers stayed in place for every setup, while the others roved to new locations in each one. The mirror does not keep the channel-to-location labels. A roving channel therefore measures a *different point on the bridge* in each setup. The split below is by setup, so with a roving channel the test setups would see the sensor somewhere training never saw it, and the evaluation would mostly measure sensor relocation rather than damage. A real single-sensor monitoring system also has one sensor fixed in one place. The channel should therefore be a fixed reference sensor.

### 4.2 Identifying the fixed sensors

`python -m src.preprocessing.channel_analysis` (output: `data/processed/channel_analysis.json`) computes the log-Welch spectrum (0.5-30 Hz) of every segment of every channel. For each segment it finds the most similar segment from a **different record**, meaning a different (scenario, setup) pair, and asks:

- **NN same setup:** is the neighbour from the same setup but a different scenario? This is high for a roving channel, whose spectrum is dominated by where it is mounted. Chance is 16/152 = 0.105.
- **NN same scenario:** is the neighbour from the same scenario but a different setup? This is high for a fixed channel that tracks the structural state across setups. Chance is 8/152 = 0.053.

| Channel | NN same setup | NN same scenario | Mode-1 minus mode-2 log-power | Max repeated-sample fraction |
|---|---|---|---|---|
| 25 | 0.324 | 0.205 | 1.52 | 0.129 |
| **22** | **0.349** | **0.244** | **1.72** | **0.274** |
| 24 | 0.507 | 0.129 | 2.69 | 0.278 |
| 26 | 0.534 | 0.165 | -0.57 | 0.166 |
| 23 | 0.547 | 0.169 | -1.25 | 0.727 |
| 5 | 0.609 | 0.138 | 2.06 | 0.532 |
| 14 | 0.614 | 0.139 | 1.91 | 0.123 |
| ... | ... | ... | ... | ... |
| 1 | 0.920 | 0.021 | 4.29 | 0.230 |

Channels 0-21 cluster overwhelmingly by setup: 66-92% of their nearest neighbours come from the same setup in another scenario, while their same-scenario rate sits at chance. They are roving sensors. **Channels 22-26 form a clear separate group** with the least setup clustering and 2.5-4.6x chance scenario clustering. They are inferred to be the fixed reference sensors. Mode-1 minus mode-2 log-power separates vertical sensors (positive, dominated by the 3.9 Hz vertical-bending mode) from transverse sensors (negative, dominated by the 5.1 Hz lateral mode). The repeated-sample fraction flags amplitude quantisation. Channel 23 repeats its previous value in up to 73% of samples in some segments and was excluded.

### 4.3 Choosing among the reference sensors

The remaining reference channels and roving channel 1 were compared with **leave-one-setup-out cross-validation of MiniRocket over setups 1-7** (`python -m src.preprocessing.channel_selection`, output: `data/processed/channel_selection_loso.json`). The test setups 8-9 were not used.

| Channel | Type | LOSO window acc (mean +- std over 7 folds) | LOSO segment acc |
|---|---|---|---|
| 1 | roving, vertical | 0.301 +- 0.092 | 0.374 |
| 26 | reference, transverse | 0.267 +- 0.079 | 0.316 |
| **22** | **reference, vertical-dominant** | **0.266 +- 0.048** | **0.343** |
| 24 | reference, vertical | 0.246 +- 0.086 | 0.291 |
| 25 | reference, vertical-dominant | 0.238 +- 0.038 | 0.284 |

Channel 1 has the highest mean, but its accuracy swings from 0.17 to 0.41 depending on which setup is held out, because its location changes. Its score on any unseen setup is luck of placement, and it fails the fixed-sensor requirement. **Channel 22** has:

- the best segment-level LOSO accuracy among the reference sensors;
- the strongest scenario clustering of all 27 channels;
- the second-lowest setup clustering;
- a fold-to-fold spread half that of channel 1;
- a vertical-dominant response.

Vertical response is preferred because pier settlement and tendon rupture act mainly on vertical bending stiffness. Its mode-1 frequency follows the published PDT trend: about 3.84-3.91 Hz in the reference states (labels 0, 1, 7), dropping to 3.77 Hz at 80 mm and 3.71 Hz at 95 mm pier settlement (labels 4-5) and recovering once the pier is lifted back.

### 4.4 Development history (for transparency)

An earlier draft selected channel 1 using a setup-stability score computed relative to the median of all channels in each setup. That normalisation is dominated by the 22 roving channels and made the fixed sensors look unstable, so the metric was replaced by the nearest-neighbour test above. MiniRocket test accuracy on channel 1 (0.208 window) was computed once before the error was found. It played no part in the corrected selection, which rests only on the analyses in 4.2-4.3, both computed without the test setups.

## 5. Preprocessing parameters

The 4096-sample / 12.5 Hz windowing figures from unrelated reference material do **not** apply here. That combination would be a 328 s window, longer than a 60 s segment, at a rate that aliases every mode above 6.25 Hz. All parameters below are derived from this dataset's verified 100 Hz rate and 6,000-sample segments.

| Step | Parameter | Value | Reasoning |
|---|---|---|---|
| Channel | index | 22 | Section 4 |
| Detrend | per setup record | remove mean of the 600 s record | removes sensor DC offset before filtering |
| Band-pass | Butterworth, zero-phase (`sosfiltfilt`) | order 4, 0.5-30 Hz | keeps every identified bridge mode (3.9-25.7 Hz) and removes sub-0.5 Hz drift and the 40-44 Hz narrow-band peaks, which are not structural and could act as setup-specific shortcuts |
| Filtering scope | | whole 600 s setup record (10 segments joined) | segments are contiguous, so filtering the joined record avoids edge transients at every 60 s boundary |
| Window length | samples | 2000 (20 s) | about 78 cycles of the 3.9 Hz fundamental. A 20 s window gives 0.05 Hz frequency resolution, enough to separate the 3.9/5.1 Hz modes and to register the few-percent frequency drops caused by damage. It is about 4x WaveNet's 512-sample receptive field. 2000 divides evenly into the 6000-sample segment |
| Window stride | samples | 1000 (50 % overlap) | 5 windows per segment. Windows never cross segment boundaries, so every window maps to exactly one labelled segment |
| Normalisation | z-score | per window: `(x - mean) / std` | ambient excitation (traffic, wind) changes signal amplitude from hour to hour independently of damage. Per-window z-scoring removes amplitude and keeps spectral shape, where damage information lives. No statistics are shared across splits, so there is no leakage |
| Training augmentation (deep models only) | random crop + sign flip | random 2000-sample crops from the filtered 6000-sample training segments, z-scored per crop, polarity flipped with p = 0.5 | more distinct training views without leaving the training setups. Accelerometer polarity carries no damage information |

## 6. Split: by setup, not random

Random splits leak information on Z24: the 10 segments of a setup are consecutive slices of one continuous record, and overlapping windows share samples. A random split would put near-duplicate signal from the same record in train and test. Splits are therefore made on the **setup number**, which also means each scenario is tested on setups recorded at a different time with roving sensors placed elsewhere.

| Split | Setups | Segments | Windows | Per class |
|---|---|---|---|---|
| Train | 1, 2, 3, 4, 5, 6 | 1020 | 5100 | 60 segments |
| Validation | 7 | 170 | 850 | 10 segments |
| Test | 8, 9 | 340 | 1700 | 20 segments |

Every split contains all 17 classes in equal proportion. The validation setup is used for checkpoint selection (deep models) and ridge-alpha selection (MiniRocket). Channel selection used leave-one-setup-out over setups 1-7. The test setups are used only for final reporting.

## 7. Output format

`python -m src.preprocessing.pipeline` writes `data/processed/z24_ch22_w2000_s1000.npz` (compressed) and a JSON summary next to it. For each split `s` in `train`, `val`, `test`:

| Key | Shape | Content |
|---|---|---|
| `s_X` | `(n_windows, 1, 2000)` float32 | filtered, z-scored windows |
| `s_y` | `(n_windows,)` | scenario label per window |
| `s_window_segment_id` | `(n_windows,)` | global segment index (0..1529) of each window |
| `s_setup` | `(n_windows,)` | setup number of each window |
| `s_segments` | `(n_segments, 1, 6000)` float32 | filtered, un-normalised full segments (source for random crops) |
| `s_segment_y`, `s_segment_id` | `(n_segments,)` | segment labels and global indices |

`window_starts` stores the 5 window offsets (0, 1000, 2000, 3000, 4000). The arrays load with `src.preprocessing.pipeline.load_processed()`.

## 8. Evaluation granularity

Models are scored at two levels:

- **Window accuracy:** one 20 s window, one prediction.
- **Segment accuracy:** the per-class scores (log-probabilities for the deep models, ridge decision values for MiniRocket) of a segment's 5 windows are summed. This gives one decision per 60 s recording, the natural unit of a monitoring system.

Every reported number comes from setup-held-out data.
