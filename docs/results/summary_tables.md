## Window-level accuracy (%)

| Model | Params | Train acc | Val acc | Test acc | Test segment acc | Test macro-F1 | Train time |
|---|---|---|---|---|---|---|---|
| WaveNet | 228,881 | 69.7 | 58.4 | 39.7 | 45.3 | 37.7 | 8.4 min (cuda) |
| MiniRocket | 169,949 (ridge) + 10k fixed kernels | 99.9 | 30.0 | 18.9 | 27.9 | 17.9 | 0.3 min (cpu) |
| InceptionTime | 493,681 | 96.9 | 54.9 | 33.8 | 42.9 | 32.3 | 11.3 min (cuda) |

## Segment-level accuracy (%)

| Model | Train segment acc | Val segment acc | Test segment acc | Test segment macro-F1 |
|---|---|---|---|---|
| WaveNet | 81.6 | 69.4 | 45.3 | 41.3 |
| MiniRocket | 100.0 | 34.7 | 27.9 | 25.9 |
| InceptionTime | 100.0 | 64.7 | 42.9 | 39.3 |

## Seed variability (%)

| Model | Seeds | Val acc (mean +- std) | Test acc (mean +- std) | Test segment acc (mean +- std) |
|---|---|---|---|---|
| WaveNet | 3 | 58.6 +- 0.7 | 39.4 +- 0.2 | 45.8 +- 0.4 |
| MiniRocket | 3 | 29.5 +- 0.4 | 19.3 +- 0.3 | 27.4 +- 1.3 |
| InceptionTime | 3 | 55.0 +- 0.1 | 34.5 +- 0.8 | 43.8 +- 1.7 |

## Test recall per scenario, segment level (%)

| Label | Scenario | WaveNet | MiniRocket | InceptionTime |
|---|---|---|---|---|
| 0 | Reference 1 | 5.0 | 0.0 | 0.0 |
| 1 | Settlement system installed | 0.0 | 0.0 | 0.0 |
| 2 | Pier -20 mm | 95.0 | 20.0 | 100.0 |
| 3 | Pier -40 mm | 25.0 | 60.0 | 35.0 |
| 4 | Pier -80 mm | 85.0 | 70.0 | 75.0 |
| 5 | Pier -95 mm | 100.0 | 50.0 | 80.0 |
| 6 | Foundation tilt | 20.0 | 40.0 | 100.0 |
| 7 | Reference 2 | 30.0 | 50.0 | 90.0 |
| 8 | Spalling 12 m2 | 80.0 | 20.0 | 15.0 |
| 9 | Spalling 24 m2 | 65.0 | 15.0 | 15.0 |
| 10 | Landslide | 45.0 | 35.0 | 65.0 |
| 11 | Concrete hinge failure | 45.0 | 10.0 | 15.0 |
| 12 | 2 anchor heads failed | 0.0 | 0.0 | 5.0 |
| 13 | 4 anchor heads failed | 5.0 | 0.0 | 10.0 |
| 14 | 2 tendons ruptured | 5.0 | 0.0 | 0.0 |
| 15 | 4 tendons ruptured | 65.0 | 50.0 | 35.0 |
| 16 | 6 tendons ruptured | 100.0 | 55.0 | 90.0 |
