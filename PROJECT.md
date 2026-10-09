# 30-day All-Cause Hospital Readmission Prediction

## 1. Background

Reducing avoidable hospital readmissions is a primary goal for healthcare systems worldwide. Unplanned readmissions within 30 days of discharge impose substantial burdens on patients and healthcare systems, and serve as critical indicators of healthcare quality. This project aims to leverage machine learning techniques to predict these readmissions, contributing to improved patient care and system efficiency

## 2. Objective

Develop a prediction model using only the provided Electronic Health Record (EHR) data (data/ehr_preprocessed_seq_by_day_cat_embedding.pkl).

## 3. Dataset Description

Electronic Health Record (EHR) Data (Tabular): Sourced from MIMIC-IV v1.0, this includes demographics, comorbidities (ICD-10 codes), lab results, and medications.

**Overview**: The dataset consists of 13763 hospital admissions from 11041 unique patients, with 2379 admissions resulting in a 30-day readmission.

**Provided Data Files**:
`train.csv`:
- id - (string): a unique id for each admission. This is the minimum unit of your prediction.
- subject_id - (string): id for each patient.
- hadm_id - (string): hospital admission id.
- dicom_id - (string): an unique id for each image. (Not related to this project)
- study_id - (string): the id of the study for each image. (Not related to this project)
- ViewPosition - (string): the view position of the chest x-ray. (Not related to this project)
- image_path - (string): the path of the image. (Not related to this project)
- readmitted_within_30days - (string): (groud truth) whether this admission is readmitted.

`valid.csv`, `test.csv` have the same format except that the `test.csv` doesn't have the ground truth.

> Note that the labeled data are pre-splitted to the `train.csv` and `valid.csv`, we may mix them together and re-split if required. For `test.csv`, we are supposed to fill in our predictions, and it is submitted to the grader, however, we cannot see the test AUROC, therefore, if we need a complete pipeline that use train and valid to develop model and use test as final score, we would have to mix train and valid together, then split to train, valid and test ourself (In specific, we would treat the provided test as additional future data)

`ehr_preprocessed_seq_by_day_cat_embedding.pkl`
- feat_dict: a dictionary, {"id": EHR features}.
- feat_cols: the names of these 171 features.
- cat_idxs: the index of the categorical features.
- cat_dims: the dimension of each categorical feature.
- demo_cols: the names of demographic features.
- icd_cols: the names of ICD features.
- lab_cols: the names of lab test features.
- med_cols: the names of medications administered features.

**Evaluation Metric**: **AUROC** is the primary metric for evaluating model performance. Other metrics may be displayed in the final reporting, but they should not lead the project design.

No additional data is allowed.

## 4. Local Device

- CPU: AMD Ryzen 7 Pro 6850H with Radeon Graphics
- GPU: NVIDIA RTX A2000 Laptop GPU

## 5. Directory Structure (Updating)

```
PROJECT_ROOT/
├── data
    ├── ehr_preprocessed_seq_by_day_cat_embedding.pkl
    ├── train.csv
    └── valid.csv
├── requirements.txt # recording package versions
├── PROJECT.md
└── 01_eda.ipynb
```


## 6. Exploratory Data Analysis (EDA) Plan

The purpose of this EDA is not to produce charts for their own sake, but to
answer a fixed set of **decision questions** whose answers directly determine
the modeling approach (tabular vs. sequence model), the preprocessing
(encoding, aggregation, scaling), the validation design (splitting, leakage
control), and the handling of class imbalance. Every step below states the
decision it feeds.

All EDA is performed on **train + valid only**. The provided `test.csv` is
treated as future data and is never inspected for distributions or label
behavior (it has no labels anyway). Any per-feature statistic that will later
be reused at inference time (e.g. normalization constants, imputation values,
rare-category thresholds) is *computed here for understanding only* and must be
re-fit inside the modeling pipeline on the training fold to avoid leakage.

### 6.0 Data objects and conventions

- Each admission `id` = `"{subject_id}_{hadm_id}"`. It maps via `feat_dict[id]`
  to an array of shape `(n_days, 171)`, where `n_days` is the length of stay in
  days (variable per admission) and 171 is the fixed feature dimension.
- Column blocks (contiguous, in this order):
  demographics `[0:3]`, ICD groups `[3:94]` (91), labs `[94:130]` (36),
  medications `[130:171]` (41).
- All values are **integer-encoded**. `cat_idxs` / `cat_dims` from the pickle
  identify which columns are categorical and their cardinalities — these MUST
  be loaded and used; do not infer feature type from the values alone.
- Label `readmitted_within_30days` is stored as a string and must be cast to
  a 0/1 integer before any analysis. Confirm the exact positive token
  (`"1"`, `"True"`, `"Yes"`, etc.) before casting.

### 6.1 Integrity, keys, and split hygiene

**Decision fed:** whether we can trust the join, and how to split for our own
train/valid/test without patient leakage.

Steps:
1. Confirm every `id` in train/valid/test exists as a key in `feat_dict`, and
   report any missing keys or orphan keys (in `feat_dict` but in no split).
2. Confirm `id` uniqueness within each split and check for `id` overlap
   *across* splits (there should be none).
3. **Patient-level overlap:** extract `subject_id` from each `id` and check
   whether the same patient appears in more than one split. Because a patient
   can have multiple admissions, a naive random re-split would leak a patient
   across train/test. Quantify how many patients have >1 admission and how many
   admissions that accounts for. **Output → the re-split in later steps must be
   `GroupShuffleSplit`/`StratifiedGroupKFold` grouped on `subject_id`.**
4. Verify `feature_cols` has length 171 and that the block index ranges
   (demo/icd/lab/med) match the counts above. Guard against silent
   mis-alignment before any block-wise analysis.

### 6.2 The central question — sequence or static?

**Decision fed:** the single most consequential modeling choice. If per-day
rows are (near-)constant within an admission, the "sequence" adds no signal and
we collapse each admission to one vector → a tabular problem (LightGBM/XGBoost),
which suits the imbalance and the laptop GPU far better than an RNN/Transformer.
If rows genuinely vary over days, a sequence model (or richer temporal
aggregation) is justified.

Steps:
1. **Length-of-stay distribution:** compute `n_days` per admission; report
   min/median/mean/p95/max and a histogram. Note the fraction with `n_days == 1`
   (for which "sequence" is meaningless).
2. **Within-admission variability:** for each admission, for each column,
   compute whether the column is constant across days. Summarize:
   - fraction of admissions where the *entire* matrix is constant across rows;
   - per-block fraction of columns that vary (demo should be constant; labs/meds
     are the candidates for real temporal change);
   - distribution of "number of distinct rows" per admission.
3. **Where does variation live?** If variation exists, quantify which blocks
   drive it (expect labs and meds, not demographics/ICD which are
   admission-level). Inspect a few high-variability admissions manually.
4. **Deliverable:** an explicit recommendation — collapse-to-static vs.
   keep-sequence — with the supporting numbers. Also define, if collapsing, the
   aggregation per block to carry forward into step 6.7 (e.g. demo = first row;
   ICD = max/any over days; labs = last + mean + min + max; meds = max/any or
   count of days administered).

### 6.3 Target analysis

**Decision fed:** loss weighting, threshold strategy, and why AUROC (not
accuracy) is the metric.

Steps:
1. Overall positive rate on train+valid (expected ≈ 2379/13763 ≈ 17.3%, but
   confirm on the actual labeled rows we hold). Report per-split rates to check
   the provided split is stratified.
2. Confirm class imbalance magnitude and translate it into concrete choices:
   class weights / `scale_pos_weight`, stratified CV, and reporting AUROC +
   PR-AUC (PR-AUC is more sensitive under imbalance) while keeping AUROC as the
   lead metric per the spec.
3. **Patient-level leakage check on the target:** since patients recur, check
   whether readmission label is correlated within a patient's admissions (a
   patient readmitted once may be readmitted again). This reinforces grouped
   splitting and warns against treating admissions as i.i.d.

### 6.4 Demographics (cols 0–2)

**Decision fed:** encoding of demographics; fairness/sanity checks; candidate
strong features.

Steps:
1. Identify each of the 3 demo columns using `demo_cols` names. Col 0 appears to
   be **age** (integer years, values ~37–91 seen), col 1 binary (**sex**), col 2
   a 1–6 categorical (likely an admission/insurance/marital-type code — confirm
   against `cat_idxs`).
2. For age: distribution, range, and check for sentinel/clipped values (MIMIC
   caps ages ≥ 89). Bin and plot readmission rate by age band.
3. For categorical demo cols: value-count tables and readmission rate per
   category; flag rare categories (<1% support) for possible grouping.
4. Univariate association of each demo feature with the target (readmission rate
   per level; for age, a monotonicity check).

### 6.5 ICD comorbidity block (cols 3–93, 91 features)

**Decision fed:** whether ICD columns are binary flags or counts; feature
selection; interpretable risk drivers.

Steps:
1. Determine value semantics after collapsing to admission level: are these
   binary present/absent flags or day-counts? Report the value range per column.
2. **Prevalence:** fraction of admissions with each ICD group present. Rank most
   and least common. Drop or flag near-constant columns (present in <0.5% or
   >99.5%) that cannot help a model.
3. **Comorbidity burden:** count of distinct ICD groups per admission; plot its
   distribution and readmission rate vs. burden (a strong expected signal).
4. **Univariate lift:** readmission rate with vs. without each ICD group; rank
   groups by absolute lift / mutual information with the target. Surface the
   top ~15 as candidate strong predictors (e.g. heart failure I30–I52, renal
   N17–N19, etc.).
5. **Redundancy:** correlation/co-occurrence among ICD groups to anticipate
   multicollinearity (matters for linear baselines, not for trees).

### 6.6 Labs (cols 94–129, 36) and Medications (cols 130–170, 41)

**Decision fed:** scaling/encoding of labs; whether labs are binned ordinals or
counts; med representation; missingness handling.

Steps:
1. **Type check via `cat_idxs`:** determine which lab/med columns are flagged
   categorical vs. numeric. The small-integer lab values may be *pre-binned
   ordinal codes* rather than raw measurements — this changes whether we scale
   (numeric) or embed/one-hot (categorical). Resolve this explicitly per column.
2. **Labs:**
   - value-range and value-count per lab column (on the collapsed representation
     chosen in 6.2/6.7);
   - a "measured vs. not" indicator per lab — a value of 0 across all days
     likely means *not ordered*, which is itself informative (many body-fluid
     labs will be almost always absent). Quantify per-lab presence rate.
   - readmission rate for measured vs. not, for the most-present labs;
   - flag labs that are essentially never populated (candidate drops).
3. **Medications:**
   - per-drug-class administration prevalence across admissions;
   - number of distinct drug classes per admission (polypharmacy proxy) and its
     relation to readmission;
   - whether med values are binary/administered-day-counts; decide max/any vs.
     count aggregation.
4. **Block-level missingness/sparsity summary:** overall sparsity of the
   lab and med blocks (fraction of zeros), to inform whether sparse-aware models
   or presence-indicators are worthwhile.

### 6.7 Aggregation / representation design (bridge to modeling)

**Decision fed:** the exact feature matrix handed to the first models.

**This spec is now finalized against the EDA results (see §7).** The §6.2
variability profile showed that demographics, ICD, and medications are constant
across days (0.000, 0.000, and 0.004 mean varying columns respectively) while
**only the labs vary within an admission** (11.3 mean varying columns). The
`cat_idxs`/`cat_dims` map also revealed the 36 lab columns are **binary**
(dim 2), not multi-level ordinal bins. The design therefore is a **hybrid
static base + temporal lab summaries**, not a naive full collapse:

- **Demographics (3):** taken once from row 0. `age` numeric; `sex` binary;
  `ethnicity` 7-way categorical (native categorical handling in LightGBM, or
  one-hot for linear models).
- **ICD (91):** any-over-days (max) → 91 binary flags. Add one engineered
  `icd_burden` = count of distinct groups present. This block is sparse and
  weak (§7); keep it but do not expect much, and consider dropping groups with
  <0.4% prevalence for linear baselines.
- **Labs (36, binary, temporal):** replace the earlier {last, mean, min, max}
  + presence-flag plan. The presence flag is **degenerate and dropped** — every
  admission shows presence for the body-fluid labs, so it carries no
  information. Instead summarize each lab's daily binary sequence with:
  `last` (last-day state), `ever` (max over days), and `frac_days` (mean over
  days = fraction of stay abnormal). Prioritize the common blood labs
  (Hemoglobin, Hematocrit, Creatinine, Glucose, Platelets, Sodium, Potassium,
  etc.); the rarely-varying body-fluid labs can be reduced to `ever` only or
  dropped.
- **Medications (41, numeric):** max-over-days per class (41) plus an engineered
  `med_count` = number of distinct classes administered (the single strongest
  simple predictor per §7). Optionally sum-over-days for dose/frequency proxy.
- **Sequence model (later experiment, not the starting point):** because only
  the labs vary, a sequence model should model the **lab time series** (shape
  `(n_days, 36)`) with static demo/ICD/med context concatenated, rather than
  feeding all 171 constant-heavy columns through an RNN. It must beat the §6.8
  tabular baseline (AUROC 0.780) on grouped validation to justify its cost and
  the extra compute on the RTX A2000.
- **Leakage checklist:** every statistic used for imputation/scaling/rare-group
  collapsing is fit on the training fold only; all CV is grouped on
  `subject_id` (§6.1.3, §7).

### 6.8 First baseline sanity model (optional but recommended in EDA)

**Decision fed:** a floor AUROC and a feature-importance sanity check before
investing in tuning.

Fit a quick LightGBM on the static aggregation with grouped stratified CV
(group = `subject_id`), report mean±std validation AUROC and PR-AUC, and inspect
top feature importances / SHAP to confirm they align with the univariate signals
found in 6.4–6.6. This validates the whole pipeline end-to-end and gives a
number every later model must beat.

### 6.9 EDA deliverables checklist

All items completed; concrete answers recorded in §7.

- [x] Integrity report (key coverage, split overlap, patient recurrence counts).
- [x] Grouped-splitting decision documented (group = `subject_id`).
- [x] Sequence-vs-static verdict with supporting variability numbers.
- [x] Target/imbalance summary and metric plan (AUROC lead, PR-AUC secondary).
- [x] Per-block feature profiles (demo, ICD, labs, meds) with prevalence,
      ranges, missingness, and univariate target association.
- [x] Ranked candidate-predictor shortlist.
- [x] Categorical vs. numeric map from `cat_idxs`/`cat_dims` for every column.
- [x] Defined admission-level aggregation and the resulting feature matrix shape.
- [x] Baseline CV AUROC as the floor for subsequent modeling.

## 7. EDA Findings & Modeling Implications

This section records the concrete results of executing the §6 plan
(`01b_eda_execution.ipynb`) and the modeling decisions they drive. Numbers are
from **train + valid (11,022 labeled admissions, 8,833 unique patients)**;
`test` was not inspected.

### 7.1 Data structure and types (resolved)

- `cat_idxs`/`cat_dims` show the **categorical columns are: sex (dim 2),
  ethnicity (dim 7), and all 36 lab columns (dim 2 each)**. Labs are therefore
  **binary abnormal-flags, not multi-level ordinal bins**. Age and all 41
  medication columns are numeric.
- The label is stored as Python booleans (`True`/`False`), cast to 1/0.
- Feature width is a constant 171; block ranges (demo 0–2, ICD 3–93, lab 94–129,
  med 130–170) verified. `feature_cols` and `feat_cols` both appear across
  artifacts — code handles either key.

### 7.2 Integrity and splitting (decision: grouped CV, mandatory)

- Clean join: 0 missing keys, 0 duplicate ids, 0 id overlap across splits. 769
  `feat_dict` keys belong to no split (harmless superset).
- **Patient recurrence is substantial:** 1,529 patients have >1 admission,
  covering 3,718 admissions (~34% of labeled data); one patient appears 10
  times. The provided train/valid split already has **0 patient overlap**.
- **Decision:** all internal re-splitting and CV must group on `subject_id`
  (`StratifiedGroupKFold`). A random split would leak ~a third of patients
  across the boundary and inflate AUROC.

### 7.3 Sequence vs. static (decision: hybrid, tabular-first)

- LOS: median 11 days, mean 15, p95 37, max 231. Only 0.6% of admissions are
  fully constant across days.
- Variability lives **almost entirely in the labs**: mean varying columns per
  admission are demo 0.000, ICD 0.000, med 0.004, **lab 11.303**.
- **Decision:** demographics, ICD, and meds are admission-level constants →
  collapse once. Labs are the only temporal signal → summarize as time series
  (see §6.7). Primary model is **tabular gradient boosting**; a lab-focused
  sequence model is a later experiment that must beat the tabular floor.

### 7.4 Target and imbalance (decision: AUROC lead + scale_pos_weight)

- Positive rate **17.5%**, near-identical across train (17.6%) and valid (17.3%)
  — the provided split is properly stratified. `scale_pos_weight ≈ 4.71`.
- **Within-patient effect:** patients with another positive admission readmit at
  **27.5% vs. the 17.5% base** — prior readmission behavior is predictive and
  reinforces grouped splitting (this signal leaks under a random split).
- **Decision:** lead metric AUROC (per spec), PR-AUC as secondary under
  imbalance; class weighting via `scale_pos_weight`; grouped stratified CV.

### 7.5 Feature-block signal strength

- **Demographics — age is a strong monotonic driver:** readmission rises from
  10.2% (<40) to 21.7% (>85). Sex is weak (16.7% vs 18.2%). Ethnicity varies by
  opaque category (12–25%); keep as categorical, do not over-interpret.
- **ICD — weak and sparse:** median 0 comorbidity groups per admission
  (mean 0.42, max 8). Most common group (hypertension I10–I16, 11.6%) has
  slightly negative lift. Genuine but rare risk flags: respiratory J80–J84
  (35% readmit, 0.4% prevalence), urinary R30–R39 (29%), malnutrition E40–E46
  (23%). Useful mainly as `icd_burden` plus a few acute flags; **no ICD column
  reaches the model's top 20 importances.**
- **Labs — presence flag is dead, blood panels matter:** the "measured vs not"
  indicator is degenerate (body-fluid labs present for every admission →
  no information; **dropped**). The informative, actually-varying labs are the
  common blood panels (Hemoglobin 82.7%, Hematocrit 83.5%, Creatinine 92.9%,
  Glucose 93.3%); Hematocrit already appears in the model top 20.
- **Medications — strongest simple signal:** distinct-drug-class count
  (polypharmacy proxy) rises monotonically with readmission, **10.8% (lowest
  quintile) → 25.8% (highest)**. Med block is 75% sparse. Medication classes
  dominate the baseline importances.

### 7.6 Baseline (the floor to beat)

- LightGBM on the static aggregation, grouped stratified 5-fold CV:
  **AUROC 0.780 ± 0.012, PR-AUC 0.569**. Grouped, so not leakage-inflated;
  tight std indicates stable signal. This is a strong readmission baseline
  (published MIMIC models commonly sit at 0.68–0.75).
- **Top-20 importances** are led by age, then a cascade of medication classes
  (electrolytes, analgesics, diuretics, antibiotics, anticoagulants,
  gastrointestinal), then ethnicity and Hematocrit — consistent with the
  univariate analysis (illness-severity signal via medication breadth and age),
  which validates the pipeline end-to-end.

### 7.7 Prioritized next steps

1. Materialize the §6.7 hybrid feature matrix (static demo/ICD/med + temporal
   lab `last`/`ever`/`frac_days`, engineered `icd_burden` and `med_count`).
2. Tune LightGBM (and try XGBoost/CatBoost) under grouped stratified CV; target
   beating AUROC 0.780. Use native categorical handling for ethnicity + labs.
3. SHAP analysis for interpretability and to prune the weak ICD block.
4. Only then, if time allows, a lab-sequence model (labs as `(n_days, 36)` with
   static context) benchmarked against the tabular floor.
5. Final pipeline: mix train+valid, re-split grouped into train/valid/test,
   treat provided `test.csv` as future data, and fill predictions for submission.

## 8. Feature Engineering (Track 1: flat table) — done

Implemented in `02_feat_agg.ipynb`. Each admission's `(n_days, 171)` day-by-day
array is collapsed into one fixed-length row, following `feature_design.xlsx`.
Built on **train + valid only** — the project's `test` set is left untouched,
since its score is not visible during development and it functionally does not
exist for model building.

### 8.1 Feature blocks built

- **Demographics (3):** `age` (numeric), `gender` (0/1), `ethnicity` (7-way
  categorical). Read from day 0 (constant within a stay).
- **ICD (92):** `icd_count` (comorbidity burden = number of distinct diagnosis
  groups present) + 91 `icd_flag__*` present-on-any-day binaries. ICD confirmed
  effectively binary (99.5% of nonzero values are exactly 1; the rare 2s are a
  coding artifact and are binarized away via `> 0`).
- **MED (83):** `med_count` (distinct drug classes ever given) + 41 `med_flag__*`
  ever-given binaries + 41 `med_intensity__*` = average administrations per day
  (`sum over days ÷ n_days`). MED confirmed to be **counts** (values up to 354),
  so per-day averaging is used deliberately to avoid smuggling length-of-stay in
  via a raw sum.
- **LAB (108):** for each of 36 labs — `lab_last__*` (final-day flag),
  `lab_flag__*` (abnormal on any day), `lab_frac__*` (fraction of days abnormal).
  Labs confirmed to be **0/1 abnormal-flags** (every lab column takes only values
  {0,1}), so the three views summarize the one binary signal over time; the
  earlier "presence indicator" idea was dropped as degenerate.
- **Other (1):** `sequence_length` = length of stay in days (heavy right tail to
  watch; log-transformed for the linear baseline in modeling).

### 8.2 Dead-column pruning

After assembly, feature columns that are **constant across the training rows**
are dropped (a constant column cannot help distinguish patients). This is done:
(1) **last**, after `icd_count`/`med_count` were already computed from the
complete blocks, so no cross-column aggregate can break; and (2) judged on
**train rows only**, then the same columns dropped from valid, so the valid split
stays a fair test. Expected dead columns from the EDA sparsity check: ~5 ICD and
~1 MED (zero columns) and ~16 LAB (constant-non-zero "always-flagged" labs). The
notebook reports the dropped columns grouped by block for cross-checking.

Output: `data/features_track1.parquet`, one row per labeled admission, with
bookkeeping columns (`id`, `subject_id`, `split`, `y`) kept separate from the
model-input features.

## 9. Modeling (Track 1) — current step

Implemented in `03_model.ipynb`. Predict 30-day readmission from the flat feature
table. Lead metric **AUROC** (project spec), with **PR-AUC** reported alongside
as the imbalance-sensitive companion.

### 9.1 Validation design

- **Patient-grouped, stratified cross-validation:** `StratifiedGroupKFold`
  (5 folds) grouped on `subject_id`, so no patient appears in both the train and
  validation parts of a fold (recurrence ~34%, established in EDA), while keeping
  the ~18% positive rate even across folds. The notebook asserts zero patient
  overlap per fold.
- **Shared folds:** fold indices are computed once and reused for every model, so
  differences reflect the model, not the split.
- **Leakage control:** any preprocessing that learns from data (scaler, one-hot,
  resampling) is fit **inside each fold on the training portion only**.
- **Pooling decision:** train + valid are pooled for CV. Rationale: the dataset
  is small and the project's real hold-out is the ungraded `test` submission, so
  all labeled data is used for CV rather than carving out another hold-out.
  (Alternative — CV on `train` only, score `valid` once — is noted in the
  notebook as a valid swap.)

### 9.2 Model ladder (agreed sequence)

1. **Logistic Regression baseline** — scaled numerics, one-hot `ethnicity`,
   `class_weight="balanced"`, `log1p(sequence_length)`. Establishes the floor.
2. **Raw CatBoost** — defaults, no class weighting; a clean untouched reference.
   Native categorical handling for `gender`/`ethnicity` (no encoding needed).
3. **CatBoost + imbalance handling** — class weights (preferred, keeps all data)
   and downsampling at ratios 1:1/1:2/1:3 (done strictly inside-fold). Expectation
   set in planning: **AUROC is largely insensitive to rebalancing** (it measures
   ranking); PR-AUC is where any effect shows. Downsampling likely underperforms
   because it discards scarce negative data. "Keep all data with class weights (or
   stay raw)" is an acceptable outcome.
4. **Tune the chosen configuration** — small grid on the same folds over the
   parameters that matter most for CatBoost: `depth`, `learning_rate`,
   `l2_leaf_reg`. Search kept small on purpose (small data + grouped CV = risk of
   overfitting the folds with a large sweep).

Final output is a single comparison table (mean ± std AUROC and PR-AUC) across all
models, read against the ~0.78 EDA sanity floor.

### 9.3 Parameters catalogued for tuning

- **Logistic Regression:** `C` (inverse regularization, e.g. 0.01–10),
  `penalty` (`l2` vs `l1` for implicit feature selection). Kept minimal by design.
- **CatBoost:** `depth` (4–8), `learning_rate` (0.03–0.1 with more iterations +
  early stopping), `l2_leaf_reg` (regularization), `auto_class_weights` /
  `class_weights` (imbalance). `border_count`, `random_strength` left at defaults
  initially.
- **LightGBM (optional cross-check, not yet built):** `num_leaves`,
  `learning_rate` + `n_estimators`, `min_child_samples`, `subsample` /
  `colsample_bytree`, `reg_alpha` / `reg_lambda`, `scale_pos_weight ≈ 4.7`.

### 9.4 Next steps after modeling

- Feature importance / SHAP on the chosen model, cross-checked against EDA signal
  (age, `med_count`, key labs); prune weak features if warranted.
- Optional **Track 2 (sequence model)**: discharge-aligned lab window
  (last ~5 days × 36 labs) + static context into a GRU, which must beat the
  Track-1 CV number to justify its cost.
- **Final submission** (separate, careful step): retrain the chosen model on all
  labeled data, predict the project's `test.csv`; its AUROC is not visible, so the
  CV estimate stands as the best expectation of test performance.

## 10. Track 1 Results — recorded

All scored on identical patient-grouped 5-fold CV (`StratifiedGroupKFold` on
`subject_id`), on the pooled train + valid data (11,022 admissions, 8,833
patients, 17.5% positive). Lead metric AUROC; PR-AUC alongside.

| Model | AUROC | PR-AUC |
|---|---|---|
| **CatBoost (tuned: d7, lr0.02, l2=20)** | **0.8042** (single split) | **0.6347** |
| CatBoost (raw) | 0.8025 ± 0.0068 | 0.6308 |
| CatBoost (downsample 1:3) | 0.8005 | 0.6279 |
| CatBoost (downsample 1:2) | 0.8003 | 0.6216 |
| CatBoost (class weights) | 0.7964 | 0.6244 |
| CatBoost (downsample 1:1) | 0.7963 | 0.6109 |
| Logistic Regression (baseline) | 0.7770 ± 0.0036 | 0.5734 |

### 10.1 Decisions and findings

- **Chosen model: CatBoost, tuned d7 / lr0.02 / l2=20, no imbalance handling,
  full 247-feature set.** Validated across 7 CV seeds at **AUROC 0.8032 ± 0.0011**
  (range [0.8014, 0.8051]), PR-AUC 0.6269 ± 0.0018. This seed-averaged 0.8032 is
  the trustworthy headline; the single-split 0.8042 was a mildly favorable split.
- **Imbalance handling hurts.** Every rebalancing variant (class weights,
  downsampling 1:1/1:2/1:3) scored below raw on both AUROC and PR-AUC — confirming
  the planning-stage expectation that a ranking metric is insensitive-to-harmed by
  rebalancing, and that downsampling wastes scarce negatives. Decision: keep all
  data, no rebalancing.
- **Tuning found a real direction, small effect.** Slow learning + strong
  regularization (lr0.02, l2=20, deeper trees) consistently beat high-learning-rate
  configs (~0.803 vs ~0.79). The tuned edge over raw (~0.0017) is within one seed
  std, so tuned ≈ raw in practice; tuned is adopted as it costs nothing and has the
  best PR-AUC.
- **Model stability is excellent** (seed std 0.0011) — small differences are now
  meaningful, which is why the above comparisons are trustworthy.
- **The linear→tree gap is real:** +0.025 AUROC from logistic (0.777) to CatBoost
  (0.803), well outside noise. Both clear the ~0.78 EDA sanity floor.

### 10.2 Feature selection outcome (notebook 04)

- No reduced feature set matched the full-set 0.803 within the noise band.
  Dropping the 78 ICD flags alone held (0.8013); all other single cuts and all
  combinations dropped below the floor. SHAP top-15 cross-checked cleanly against
  the EDA (age, Hematocrit, electrolyte/antibiotic med-intensity, sequence_length,
  med_count on top), validating the pipeline.
- **Conclusion: keep the full feature set.** Feature selection confirmed the
  plateau rather than breaking it; the current flat representation is near its
  ceiling. Breaking the plateau requires a new representation → Track 2.

## 11. Track 2 — Sequence Model (GRU) — plan

**Motivation.** The flat table compresses each lab's day-by-day trajectory into a
few summaries (last / ever / frac). Labs are the only block that varies over time
(EDA §7), so their *trajectory shape* (improving vs. deteriorating toward
discharge) is information the flat model cannot see. A sequence model reads the
day-by-day labs directly and may capture this. It must beat the Track-1 validated
0.8032 to justify its cost.

**Design.**
- **Input representation:** per admission, a discharge-aligned window of the last
  N days (N≈7) of the full 171-feature daily array (or labs + static context),
  left-padded for shorter stays with a mask so padded steps are ignored.
- **Architecture:** GRU over the day sequence → final hidden state concatenated
  with static features (demographics, counts) → MLP head → sigmoid. GRU chosen
  over LSTM for fewer parameters on a small dataset; over a Transformer because
  ~11k admissions is small for attention.
- **Validation:** the SAME patient-grouped 5-fold CV and the same multi-seed
  stability check as Track 1, so the comparison to 0.8032 is apples-to-apples.
- **Imbalance:** consistent with Track 1, expect raw (no rebalancing) to be best
  for AUROC; may revisit only if PR-AUC is the focus.
- **Compute:** GPU (CUDA) for training speed.
- **Honest expectation:** deep models often struggle to beat well-tuned GBDTs on
  small tabular EHR data; the sequence model is a genuine test of whether lab
  trajectory shape carries signal, and a negative result is itself informative.

## 12. Track 2 Results & Ensemble — recorded

### 12.1 GRU results

- **v1 (all 171 columns through GRU):** AUROC 0.7904 ± 0.0070 — below the Track-1
  bar. Diagnosis: 158 of 171 columns are constant within a stay, diluting the ~36
  varying lab columns.
- **v2 (two-branch: lab sequence through GRU + static features bypass):** AUROC
  **0.8083 ± 0.0095** — above the Track-1 CatBoost (0.8032). Focusing the recurrence
  on the labs (the only temporal block) fixed the dilution. Confirms lab trajectory
  shape carries real signal beyond the flat summaries.

### 12.2 Ensemble (notebook 06)

Regenerated OOF predictions for both models under one shared 5-fold grouped split,
then combined (no meta-model / stacking).

- **Decorrelation gate:** Spearman(pred) = 0.721 — the models rank patients
  differently, so an ensemble has room to help (green light).
- **Results (OOF AUROC):** CatBoost 0.8012, GRU 0.8042, plain avg 0.8117,
  rank avg 0.8118, weighted (w_cat=0.60) 0.8121. All three blends beat both single
  models by ~0.008; the three agree closely, so the gain is trustworthy, not a
  lucky weight.
- **Chosen final model:** rank-average ensemble of CatBoost + lab-focused GRU
  (rank average is scale-invariant and matched the weighted blend). **Estimated
  AUROC ~0.812.**

### 12.3 Final submission (notebook 07)

- Both base models retrained on **all labeled data** (train + valid pooled); test
  features built with identical functions; predictions combined by rank average;
  written to `data/submission.csv`.
- The GRU is trained a fixed ~18 epochs (no held-out set when using all data),
  approximating where CV early-stopping landed.
- **Honest caveats:** (1) the ensemble choice/weight was tuned on OOF, so ~0.812 is
  a mild optimistic estimate; true test AUROC may be slightly lower. (2) test AUROC
  is not visible to us — OOF is our best estimate. (3) GRU fixed-epoch training is a
  reasonable shortcut, not fully rigorous early stopping.

## 13. Project Summary — model progression

| Stage | Model | CV AUROC |
|---|---|---|
| Baseline | Logistic Regression | 0.777 |
| Track 1 | CatBoost (tuned) | 0.803 |
| Track 2 | GRU (lab-focused, two-branch) | 0.808 |
| **Final** | **Rank-average ensemble** | **~0.812** |

Each step was validated on identical patient-grouped 5-fold CV; the +0.035 AUROC
from baseline to final ensemble is well outside the seed-noise band (~0.001–0.01).
**Final deliverable:** `data/submission.csv` from notebook 07.