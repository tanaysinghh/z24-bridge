# Damage Severity Index and Remaining Useful Life

This document defines the Damage Severity Index (DSI), the degradation model that turns classifier output into a remaining-useful-life (RUL) estimate, the maintenance actions per severity tier, and what can and cannot be validated. The implementation is `src/rul.py` (shared verbatim with the API as `api/vibra/rul.py`), the mapping lives in `src/taxonomy.py`, and the validation script is `src/rul_evaluate.py` (results in `docs/results/rul.json` and `docs/results/rul.md`).

## 1. What Z24 can and cannot support

**Z24 has no ground-truth lifespan labels.** The Progressive Damage Tests applied 17 controlled damage states to a bridge that was demolished right afterwards. Nobody observed how long the bridge would have lasted in any of those states, and no remaining-life measurement exists for any recording.

So the RUL produced here is a **model-based prognostic estimate derived from the severity ordering**. It depends on the DSI values and degradation parameters defined below, which are engineering assumptions, not quantities fitted to Z24. **It has not been validated against real remaining-life measurements and cannot be with this dataset.** The only validation that is possible, and the only one reported, is whether the *predicted DSI* follows the *defined severity ordering* on held-out test recordings (section 6).

## 2. Damage Severity Index per scenario

DSI runs from 0 (intact) to 1 (the most severe state reached in the Z24 campaign, 6 of 16 tendons ruptured). DSI = 1 is used as the end-of-service threshold of the degradation model: it does not mean collapse, it means the state at which a bridge owner would have to stop normal service and intervene.

The values were set from the scenario descriptions in `docs/dataset.md` using four rules:

1. **Healthy states are near 0.** Both reference measurements and the state after the settlement system was installed carry no structural damage. Reference 2 is given 0.05 because it was measured after the pier had been lowered by up to 95 mm and lifted back, which may have left minor residual effects.
2. **Settlement scales with magnitude.** Pier lowering of 20, 40, 80 and 95 mm maps to 0.20, 0.30, 0.45 and 0.55. Settlement was reversed in the campaign but is a serious support failure while present.
3. **From scenario 8 on, damage is cumulative.** After the settlement series, the PDT applied spalling, the landslide, hinge failure, anchor-head failure and tendon rupture one after another without repair, so each later state contains the earlier ones. DSI therefore rises monotonically with the chronological order of scenarios 8 to 16.
4. **Prestressing damage is the most critical.** For a post-tensioned box girder, losing anchorages and tendons removes the prestress the girder depends on, so the tendon family takes the top of the scale.

| Label | Scenario | Group | DSI | Tier |
|---|---|---|---|---|
| 0 | Reference 1 | Healthy | 0.00 | Healthy |
| 1 | Settlement system installed | Healthy | 0.02 | Healthy |
| 7 | Reference 2 | Healthy | 0.05 | Healthy |
| 2 | Pier lowered 20 mm | Settlement | 0.20 | Minor |
| 3 | Pier lowered 40 mm | Settlement | 0.30 | Minor |
| 8 | Spalling of concrete at soffit, 12 m² | Hinge (concrete) | 0.35 | Minor |
| 4 | Pier lowered 80 mm | Settlement | 0.45 | Moderate |
| 9 | Spalling of concrete at soffit, 24 m² | Hinge (concrete) | 0.45 | Moderate |
| 6 | Pier lifted, tilt of foundation | Foundation | 0.50 | Moderate |
| 5 | Pier lowered 95 mm | Settlement | 0.55 | Moderate |
| 10 | Landslide at abutment | Foundation | 0.55 | Moderate |
| 11 | Failure of concrete hinge | Hinge (concrete) | 0.65 | Severe |
| 12 | Failure of 2 anchor heads | Tendon | 0.70 | Severe |
| 13 | Failure of 4 anchor heads | Tendon | 0.75 | Severe |
| 14 | Rupture of 2 of 16 tendons | Tendon | 0.80 | Critical |
| 15 | Rupture of 4 of 16 tendons | Tendon | 0.90 | Critical |
| 16 | Rupture of 6 of 16 tendons | Tendon | 1.00 | Critical |

The exact numbers between anchors are a judgement call. What matters for validation is the ordering, and the rank-based metrics in section 6 depend only on that.

## 3. From class probabilities to a DSI distribution

For one recording the classifier (the deployed stacking ensemble) returns probabilities `p_c` over the 17 scenarios. Recording-level probabilities are the normalised geometric mean of the window probabilities, `softmax(mean_w log p_w)`, which gives the same decision as the summed log-probabilities used for recording accuracy.

- **Point estimate:** expected DSI, `E[DSI] = sum_c p_c * DSI_c`.
- **Distribution:** Monte Carlo with `N = 4000` draws. Each draw samples a scenario `c ~ Categorical(p)` and then `DSI = clip(DSI_c + e, 0, 1)` with `e ~ Normal(0, 0.04)`. The noise term represents uncertainty in the DSI value assigned to each scenario. The 90% interval is the 5th to 95th percentile of the draws.

A confident prediction gives a narrow DSI interval. A prediction split between, for example, Reference 1 and 6 tendons ruptured (a known confusion, see `docs/results_review2.md`) gives a wide, bimodal interval, and the app shows that uncertainty instead of hiding it.

## 4. Degradation model and RUL

Damage is assumed to accumulate along a power-law curve over a service life `L`:

```
DSI(t) = (t / L) ^ beta,        0 <= t <= L
```

`beta > 1` makes degradation accelerate with age, the usual shape for concrete deterioration and prestress loss. A structure observed at damage level `d` sits at the equivalent age `t_eq = L * d^(1/beta)` on this curve, and its remaining life before reaching DSI = 1 is

```
RUL = L - t_eq = L * (1 - d^(1/beta))
```

Each Monte Carlo draw pairs one sampled DSI with one draw of the model parameters:

| Parameter | Distribution | Reasoning |
|---|---|---|
| Service life `L` | Normal(100, 10) years, truncated at 60 | 100 years is the design working life of road bridges in EN 1990 (category 5) |
| Exponent `beta` | Uniform(1.5, 3.0) | accelerating degradation; covers the range from mild to strongly accelerating curves |
| DSI | from section 3 | classifier uncertainty plus scenario-value uncertainty |

The reported RUL is the median of the draws with the 5th to 95th percentile as the **90% interval**. The interval therefore combines classifier uncertainty, DSI-value uncertainty and degradation-model uncertainty. Typical outputs are about 100 years at DSI 0, about 30 years at DSI 0.45 and 0 at DSI 1.

The bridge's real age is not used: the equivalent age is inferred from the damage state alone. That keeps the model usable on a single recording, and is another reason the output is a relative prognostic indicator, not a forecast.

## 5. Severity tiers and maintenance actions

The tier is taken from the expected DSI. The app also shows the probability of each tier across the Monte Carlo draws.

| Tier | DSI range | Recommended action | Urgency |
|---|---|---|---|
| Healthy | 0.00 - 0.15 | Routine monitoring. Keep the scheduled visual inspection cycle; no intervention needed. | Next scheduled inspection |
| Minor | 0.15 - 0.40 | Increase monitoring frequency and carry out a targeted visual inspection of the suspected damage location. | Within 6 months |
| Moderate | 0.40 - 0.60 | Detailed inspection with non-destructive testing, a structural assessment of load capacity and a repair plan. | Within 1 month |
| Severe | 0.60 - 0.80 | Urgent engineering inspection, consider load restrictions and plan strengthening of the affected members. | Within 1 week |
| Critical | 0.80 - 1.00 | Restrict or close the bridge to traffic, inspect immediately and start emergency structural intervention. | Immediately |

These are generic recommendations tied to the severity scale, written for a demonstration system. They are not a substitute for an inspection regime set by the responsible authority.

## 6. Validation that is possible

Because no remaining-life ground truth exists, only the DSI is checked, and only against the ordering defined in section 2. On the test setups 8-9 (never used for any fitting or selection), for each recording:

- **Spearman rank correlation** and **Kendall tau** between predicted `E[DSI]` and the defined DSI of the true scenario. These depend only on the ordering, not on the chosen numbers.
- **Mean absolute DSI error**.
- **Tier accuracy** (5 tiers) and **tier within one step**.

These numbers say how well the classifier's output respects the severity ordering. **They are not a validation of RUL in years.** The results are in `docs/results/rul.md` and summarised in `docs/final_results.md`.

## 7. Limitations

- DSI values, the degradation law and its parameters are assumptions. Different but equally defensible choices would shift RUL values in years; the ordering-based validation would be unchanged.
- The classifier's known confusions carry straight through to DSI. Reference 1 is often predicted as 6 tendons ruptured (section 4 of `docs/results_review2.md`), which can turn an intact recording into a high-DSI, low-RUL estimate. The wide interval in such cases is the honest signal.
- The Z24 damage states were imposed over weeks, not decades. Nothing in the data describes how damage evolves over time, so the time axis of the degradation model comes entirely from the assumed service life.
