# Project Pipeline — 30-Day Hospital Readmission Prediction

A step-by-step description of the pipeline. Each step lists its **purpose**,
**method**, **finding**, and any **problem discovered & how it was solved**.

---

## Step 1 — Exploratory Data Analysis (`01_eda.ipynb`)

- **Purpose:** understand the data before modeling — size, shape, balance, and
  which parts carry signal.
- **Method:** load the day-by-day feature arrays and labels; profile integrity,
  patient recurrence, length of stay, per-block variability, and per-feature
  association with readmission.
- **Finding:** ~11k admissions from ~8.8k patients; label imbalanced (~17.5%
  positive); data clean. Only the **lab block varies day to day** — demographics,
  diagnoses, and meds are constant within a stay. Age, medication-class count, and
  a few labs relate most strongly to readmission; diagnoses (ICD) are sparse and
  weak.
- **Problem discovered & solved:** ~34% of admissions belong to repeat patients →
  a random split would leak a patient across train/test. **Solved** by mandating
  patient-grouped splitting (`StratifiedGroupKFold` on `subject_id`) everywhere
  downstream.

## Step 2 — Data-type verification (within EDA)

- **Purpose:** confirm what the encoded integer values actually mean, since it
  drives feature design.
- **Method:** inspect value ranges per feature block.
- **Finding:** ICD is effectively **binary** (present/absent); MED is **counts**
  (up to 354/day); LAB is **0/1 abnormal-flags**.
- **Problem discovered & solved:** an early assumption that MED could be treated
  like flags was wrong (they are counts) → **solved** by designing MED features as
  per-day-averaged intensity, not raw presence, and never a raw sum (which would
  encode length of stay).

## Step 3 — Feature Engineering (`02_feat_agg.ipynb`)

- **Purpose:** collapse each variable-length admission into one fixed feature row
  for tree models.
- **Method:** per block — demographics from day 0; ICD as burden count + 91
  present flags; MED as count + 41 flags + 41 per-day intensities; LAB as last-day
  / ever-abnormal / fraction-of-days-abnormal; plus length of stay.
- **Finding:** a ~247-feature flat table (after pruning), built on train + valid
  only.
- **Problem discovered & solved:** dead (constant) columns exist and could break
  cross-column aggregates if dropped too early → **solved** by dropping dead
  columns **last** (after counts are computed) and judging "constant" on **train
  rows only** to avoid leakage.

## Step 4 — Modeling & Baselines (`03_model.ipynb`)

- **Purpose:** establish a baseline, a strong model, and the effect of imbalance
  handling.
- **Method:** logistic regression (scaled, one-hot, balanced) → raw CatBoost →
  CatBoost with class weights / downsampling → tuning; all on shared grouped folds,
  AUROC + PR-AUC.
- **Finding:** Logistic 0.777 → CatBoost 0.803. **Imbalance handling hurt** AUROC
  (rebalancing changes ranking little and downsampling wastes data). Tuning found a
  real direction (slow learning + strong regularization) for a small gain.
- **Problem discovered & solved:** tuning's reported std showed a fake `0.0000` →
  **solved** by carrying true per-fold arrays through instead of a pre-averaged
  value; a multi-seed check then confirmed stability (0.8032 ± 0.0011).

## Step 5 — Feature Selection (`04_feature_selection.ipynb`)

- **Purpose:** get a smaller/simpler model at equal AUROC, and possibly a small
  gain.
- **Method:** tiered ablation — structural redundancy cuts by block, then
  SHAP-ranked tail cuts — each judged against the seed-noise band.
- **Finding:** no reduced set matched the full-set 0.803 within noise; SHAP top
  features matched the EDA (validating the pipeline). **The flat representation is
  at its ceiling.**
- **Problem discovered & solved:** risk of leakage/over-cutting → **solved** by
  using the measured seed-noise band as the accept/reject rule, and (pragmatically)
  selecting once on pooled data with the optimism caveat noted.

## Step 6 — Sequence Model / GRU (`05_gru.ipynb`)

- **Purpose:** test whether lab **trajectory shape** (which the flat table can't
  see) carries extra signal.
- **Method:** discharge-aligned windows into a GRU; patient-grouped CV; GPU.
- **Finding (v1):** feeding all 171 columns scored 0.790 — **below** CatBoost.
- **Problem discovered & solved:** 158 of 171 columns are constant, diluting the
  ~36 varying labs → **solved** with a **two-branch** design: the lab sequence goes
  through the GRU, static features bypass it. **v2 scored 0.808**, above CatBoost —
  lab trajectory shape does help.

## Step 7 — Ensemble (`06_ensemble.ipynb`)

- **Purpose:** combine CatBoost and GRU for a possible further gain.
- **Method:** regenerate out-of-fold (OOF) predictions for both under one shared
  split; check decorrelation; combine by plain / rank / weighted average (no
  stacking).
- **Finding:** models rank patients differently (Spearman 0.72), so combining
  helps. All three blends reached **~0.812**, above both singles by ~0.008; the
  three agreeing makes the gain trustworthy.
- **Problem discovered & solved:** tree vs. neural-net probabilities sit on
  different scales → **solved** by rank-averaging (scale-invariant, matched to the
  AUROC metric).

## Step 8 — Final Submission (`07_submission.ipynb`)

- **Purpose:** produce the actual deliverable — predictions on `test.csv`.
- **Method:** retrain both base models on **all** labeled data; build test features
  with identical functions; combine by rank average; write `submission.csv`.
- **Finding:** submission file produced; estimated AUROC ~0.812 (from OOF).
- **Problem discovered & solved:** building the flat frame via stacking cast the
  categorical columns to float, which CatBoost rejects → **solved** by casting
  `gender`/`ethnicity` back to int before fitting. Also: no held-out set when
  training on all data → used a fixed epoch count near where CV early-stopping
  landed (noted as a pragmatic shortcut).

---

## Cross-cutting principles applied throughout

- **Patient-grouped CV** everywhere (no patient across the split boundary).
- **Leakage control:** every learned transform (scaling, encoding, dead-column
  choice, ensemble weight) fit on training data / OOF only.
- **Noise-aware decisions:** a measured seed-noise band (~0.001–0.01) decided
  whether any change was real, preventing chasing of noise.
- **Honest negatives recorded:** imbalance handling and feature selection both
  didn't help — kept and reported, not hidden.