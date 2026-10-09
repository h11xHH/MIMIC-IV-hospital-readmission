# Project Report — 30-Day All-Cause Hospital Readmission Prediction

## Objective

Predict whether a hospital admission results in an unplanned readmission within
30 days, using preprocessed EHR data (demographics, diagnoses, labs, medications).
Primary metric: **AUROC**.

## Data

13,763 admissions from 11,041 patients (2,379 readmissions ≈ 17.3%). Each
admission is a day-by-day sequence of 171 features across four blocks:
demographics (3), ICD diagnoses (91), labs (36), and medications (41). Model
development used the labeled train + valid sets (~11,022 admissions); the
project's `test` set was reserved for the final ungraded submission and never
used for model development.

## Approach

The project followed a staged pipeline: exploratory analysis → feature
engineering → tabular modeling → feature selection → sequence modeling →
ensembling → submission. Two modeling representations were built:

- a **flat table** (one summary row per admission) for gradient-boosted trees, and
- **discharge-aligned day sequences** for a recurrent network.

All models were evaluated on identical **patient-grouped 5-fold cross-validation**
(grouping on `subject_id`), because ~34% of admissions belong to repeat patients
and a naive split would leak patients across the train/validation boundary.

## Key findings

- **Only the lab block varies over time;** demographics, diagnoses, and
  medications are constant within an admission. This shaped every later decision —
  most notably that a sequence model should focus on the labs.
- **Class-imbalance handling did not help.** Class weights and downsampling both
  reduced AUROC relative to leaving the data untouched — expected for a ranking
  metric, and downsampling additionally wastes scarce positive-class information.
- **The flat feature representation plateaued** near AUROC 0.803; feature
  selection could not exceed it, indicating the hand-built summaries were near
  their ceiling.
- **Lab trajectory shape carries real signal.** A GRU restricted to the lab
  sequence (with static features bypassing the recurrence) reached 0.808, beating
  the tree model — but only after correcting an initial design that diluted the
  labs with 158 constant columns.
- **The two models are complementary** (they rank patients differently), so an
  ensemble improved on both.

## Results

| Model | CV AUROC |
|---|---|
| Logistic Regression (baseline) | 0.777 |
| CatBoost (tuned, flat features) | 0.803 |
| GRU (lab-focused sequence) | 0.808 |
| **Rank-average ensemble (final)** | **~0.812** |

The final model is a rank-average ensemble of the CatBoost and GRU predictions.
The +0.035 AUROC improvement from baseline to ensemble is well outside the
measured cross-validation noise band (~0.001–0.01), and the CatBoost model was
confirmed stable across seven CV seeds (0.8032 ± 0.0011).

## Deliverable

`data/submission.csv` — one combined readmission-risk score per test admission,
produced by retraining both base models on all labeled data and rank-averaging
their test predictions.

## Limitations & honest caveats

- **The ~0.812 estimate is mildly optimistic:** the ensemble combination was
  chosen on out-of-fold predictions, so the true test AUROC may be slightly lower.
- **No independent held-out set:** the small dataset led us to pool train + valid
  for cross-validation, so the ungraded test submission is the only true hold-out.
  Test AUROC is not visible during development; the cross-validation estimate is
  our best expectation.
- **The GRU's final fit uses a fixed epoch count** (no validation set when training
  on all data), a reasonable but not fully rigorous shortcut.
- The absolute AUROC values are marginally inflated by best-epoch-per-fold
  selection, applied consistently to both models so comparisons remain fair.

## Conclusion

A disciplined, leakage-controlled pipeline produced a well-validated ensemble at
~0.812 AUROC, comfortably above a linear baseline and a solid result for tabular
EHR readmission prediction. The most informative finding was methodological: the
data's only temporal signal lives in the labs, and exploiting their trajectory via
a focused sequence model — then ensembling it with a strong tree model — delivered
the gains, while imbalance handling and feature selection did not.