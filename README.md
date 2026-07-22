# 30-Day Hospital Readmission Risk Prediction & Cost-Aware Intervention Optimizer

A complete, end-to-end classification project that (1) predicts which discharged
patients are likely to be **readmitted within 30 days**, and (2) turns those
predictions into a **cost-optimal intervention policy** — i.e., who should
actually be enrolled in a transitional-care program, given that the program
costs money and readmissions cost more.

> **Note on data:** No dataset was provided, so `src/generate_data.py` builds a
> statistically realistic **synthetic** discharge dataset (6,000 patients,
> ~15% 30-day readmission rate, in line with published Medicare all-cause
> readmission rates). Every column name mirrors what you'd find in a real EHR
> discharge extract, so you can drop in real data by keeping the same schema.

---

## 1. Project structure

```
readmission_project/
├── data/
│   └── hospital_data.csv          # synthetic patient discharge dataset
├── src/
│   ├── generate_data.py           # synthetic data generator
│   ├── cost_optimizer.py          # cost-aware threshold optimization logic
│   └── train_models.py            # trains & evaluates all 7 classifiers
├── models/
│   └── best_model.pkl             # pickled best pipeline (preprocessor + model)
├── results/
│   ├── metrics_summary.csv        # accuracy/precision/recall/F1/ROC-AUC/PR-AUC
│   ├── cost_summary.csv           # optimal threshold + $ savings per model
│   ├── roc_curves.png / pr_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance_rf.png / feature_importance_gb.png
│   ├── cost_curve_<model>.png     # expected-cost-vs-threshold, per model
│   └── run_config.json
├── requirements.txt
└── README.md
```

Run it yourself:
```bash
pip install -r requirements.txt
python src/generate_data.py     # (re)generate the dataset
python src/train_models.py      # train all models + produce all results
```

---

## 2. The dataset

20 features covering four groups, plus the binary target `readmitted_30d`:

| Group | Features |
|---|---|
| Demographics / coverage | `age`, `gender`, `insurance` |
| Admission details | `admission_type`, `discharge_disposition`, `length_of_stay` |
| Utilization history | `num_prior_admissions_12mo`, `num_er_visits_12mo` |
| Clinical burden | `num_diagnoses`, `num_medications`, `num_lab_procedures`, `num_procedures`, `diabetes`, `heart_failure`, `copd`, `ckd`, `depression`, `cancer`, `polypharmacy`, `pcp_followup_scheduled` |

The target is generated from a logistic latent-risk model driven by prior
utilization, comorbidity burden, discharge disposition, insurance type, and
whether follow-up care was already scheduled — plus irreducible noise, so no
model can (or should) reach near-perfect accuracy. That's intentional: it
mirrors the ceiling real readmission models hit in practice (published
literature typically reports ROC-AUC around 0.65–0.75 for this problem).

---

## 3. Modeling pipeline

**Preprocessing** (`ColumnTransformer` inside an sklearn `Pipeline`, fit only
on the training fold to avoid leakage):
- Numeric features → median imputation + standard scaling
- Categorical features → most-frequent imputation + one-hot encoding

**Split:** 75/25 train/test, stratified on the target.

**All 7 requested classifiers, each wrapped in the same preprocessing
pipeline for a fair comparison:**

| # | Model | Key settings |
|---|---|---|
| 1 | Logistic Regression | `class_weight="balanced"`, L2 penalty |
| 2 | K-Nearest Neighbors | k=25, distance-weighted |
| 3 | Decision Tree | max_depth=6, `class_weight="balanced"` |
| 4 | Random Forest | 400 trees, max_depth=8, `class_weight="balanced_subsample"` |
| 5 | Support Vector Machine | RBF kernel, C=2.0, `probability=True` |
| 6 | Naive Bayes | Gaussian NB |
| 7 | Gradient Boosting | 300 estimators, max_depth=3, lr=0.05 |

`class_weight="balanced"` (or subsample-balanced) is used wherever supported
because readmission is a minority class (~15%) — without it, models default
toward always predicting "no readmission."

---

## 4. Results (this run)

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.660 | 0.258 | 0.694 | 0.377 | **0.704** | 0.327 |
| Naive Bayes | 0.805 | 0.276 | 0.194 | 0.228 | 0.678 | 0.253 |
| Random Forest | 0.743 | 0.253 | 0.378 | 0.303 | 0.676 | 0.276 |
| Gradient Boosting | 0.853 | 0.538 | 0.032 | 0.060 | 0.676 | 0.289 |
| SVM (RBF) | 0.713 | 0.223 | 0.378 | 0.281 | 0.619 | 0.206 |
| Decision Tree | 0.549 | 0.197 | 0.662 | 0.303 | 0.616 | 0.211 |
| K-Nearest Neighbors | 0.852 | 0.000 | 0.000 | 0.000 | 0.606 | 0.217 |

**Logistic Regression is the top model by ROC-AUC (0.704)** and is the one
saved to `models/best_model.pkl`. Note the classic accuracy trap here:
Gradient Boosting and KNN post the *highest accuracy* (~0.85) simply by
predicting "no readmission" for almost everyone — useless for an
intervention program, which is exactly why **ROC-AUC / recall / cost
savings**, not accuracy, drive the model choice in this problem.

See `results/roc_curves.png`, `pr_curves.png`, and `confusion_matrices.png`
for the full visual comparison, and `feature_importance_rf.png` /
`feature_importance_gb.png` for which features drive risk (prior admissions,
ER visits, comorbidity flags, and lack of scheduled follow-up dominate, as
built into the data-generating process).

---

## 5. Cost-aware intervention optimization

Predicting risk is only half the problem — a care team has to decide **who
actually gets the intervention**. `src/cost_optimizer.py` encodes an explicit
cost model:

| Parameter | Default | Rationale |
|---|---|---|
| `COST_INTERVENTION` | $500/patient | Nurse call + med reconciliation + follow-up scheduling |
| `COST_READMISSION` | $13,500 | Average marginal cost of an avoidable 30-day readmission |
| `INTERVENTION_EFFECTIVENESS` | 28% relative risk reduction | Consistent with transitional-care trial literature (~20–35% RRR) |

For every threshold `t`, a patient is flagged if `P(readmit) >= t`. Expected
cost per patient is:
- Flagged: `COST_INTERVENTION + p × (1 − effectiveness) × COST_READMISSION`
- Not flagged: `p × COST_READMISSION`

We scan thresholds 0.01→0.99 and pick the one minimizing **total expected
cost across the test population**, then compare against two naive baselines:
"treat nobody" and "treat everybody."

**Cost-optimal policy per model (test set, 1,500 patients):**

| model | optimal threshold | % of patients flagged | $ saved vs. best naive strategy |
|---|---|---|---|
| Naive Bayes | 0.13 | 34% | **$292,449** |
| K-Nearest Neighbors | 0.13 | 42% | $185,214 |
| Gradient Boosting | 0.13 | 44% | $168,179 |
| SVM (RBF) | 0.13 | 53% | $100,288 |
| Decision Tree | 0.10 | 91% | $58,000 |
| Logistic Regression | 0.13 | 99% | $1,230 |
| Random Forest | 0.01 | 100% | $0 |

**Key takeaway:** the model with the best ROC-AUC is not automatically the
model with the best cost-optimized policy — it depends on how well-separated
and well-calibrated the predicted probabilities are around the decision
threshold, not just ranking quality. In this run, several models converge on
roughly the same low threshold (~0.13, close to the base rate) because the
intervention is cheap relative to a readmission (500 vs 13,500), so it pays
to flag broadly; models whose probability outputs cluster too tightly near
the base rate (e.g., Random Forest, Logistic Regression here) end up flagging
almost everyone and capture little savings over "treat all." **In practice
you would pick the single model that: (a) has strong ROC-AUC/PR-AUC, and (b)
produces the largest $ savings under your real cost assumptions** — swap in
your organization's actual intervention cost, readmission cost, and program
effectiveness in `cost_optimizer.py` and re-run.

Per-model cost curves are in `results/cost_curve_<model>.png`.

---

## 6. Interactive app & deployment

`app.py` is a Streamlit web app that loads `models/best_model.pkl` and
`results/run_config.json` (the cost-optimal threshold) and lets you enter a
patient's discharge details to get:
- Predicted 30-day readmission probability
- A cost-aware **enroll / don't enroll** recommendation, using the same
  threshold logic as `cost_optimizer.py`

**Run it locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens the app at `http://localhost:8501`.

**Deploy it for free (Streamlit Community Cloud):**
1. Push this whole folder to a GitHub repo (already done if you're reading
   this from GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with
   GitHub → "New app."
3. Select your repo, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically and
   gives you a public URL (`https://<something>.streamlit.app`).

No server, Docker, or extra config needed — the model file and config are
committed to the repo, so the app is fully self-contained.

---

## 7. How to extend this to production

1. **Swap in real data** — keep the column names in `generate_data.py`'s
   output schema, or edit `train_models.py`'s `load_data()` to point at your
   EHR extract / claims table.
2. **Calibrate probabilities** — wrap the chosen model in
   `sklearn.calibration.CalibratedClassifierCV` if you'll use the raw
   probabilities for cost decisions (tree ensembles in particular can be
   poorly calibrated out of the box).
3. **Cross-validate the cost threshold**, not just the model — pick both the
   model *and* threshold via nested CV so the reported savings aren't
   overfit to one test split.
4. **Update the cost assumptions** in `cost_optimizer.py` (`COST_INTERVENTION`,
   `COST_READMISSION`, `INTERVENTION_EFFECTIVENESS`) with your organization's
   real, ideally audited, figures — the optimizer is only as good as these
   inputs.
5. **Monitor for drift** — re-fit periodically; readmission drivers shift
   with policy changes (e.g., new CMS penalty rules), seasonal illness
   patterns, and changes in discharge protocol.
6. **Fairness check** — because `insurance` type and `discharge_disposition`
   are used as predictors, audit the model for disparate impact across
   payer/demographic groups before deploying an intervention-allocation
   policy operationally.

---

## 8. Limitations

- All results here are on **synthetic data** with intentionally moderate,
  realistic signal (ROC-AUC ceiling ~0.70–0.75) — treat the *pipeline and
  methodology* as the deliverable, not these specific numbers.
- Hyperparameters were set to sensible defaults, not exhaustively tuned via
  grid/Bayesian search — there's clear room to improve any individual model
  with `GridSearchCV` / `RandomizedSearchCV` on this preprocessing pipeline.
- The cost model assumes independence across patients (no capacity
  constraint on how many patients the care team can actually enroll) — a
  real deployment would add a **budget/capacity constraint** (e.g., "we can
  only enroll 200 patients/month," turning this into a ranking + top-K
  selection problem rather than a free threshold).
