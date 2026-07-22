"""
generate_data.py
----------------
Generates a realistic SYNTHETIC hospital discharge dataset for 30-day
readmission risk modeling. There is no real patient data here (none was
provided) -- this creates a statistically-structured stand-in so the full
pipeline (EDA -> modeling -> cost-aware optimization) can be built, tested,
and demonstrated end-to-end. Swap in a real extract (e.g. a MIMIC-IV or
CMS-style discharge table) by keeping the same column names.

Target: `readmitted_30d` (1 = readmitted to hospital within 30 days of discharge)
Base rate tuned to ~19%, in line with published 30-day readmission rates
for general medical admissions (Medicare all-cause readmission ~ 15-20%).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 6000

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate(n=N, seed=42):
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(63, 16, n), 18, 95).round(0)
    gender = rng.choice(["F", "M"], n, p=[0.52, 0.48])
    insurance = rng.choice(
        ["Medicare", "Medicaid", "Private", "Uninsured"], n, p=[0.55, 0.18, 0.22, 0.05]
    )

    # Clinical / utilization history
    num_prior_admissions_12mo = rng.poisson(0.9, n)
    num_er_visits_12mo = rng.poisson(1.1, n)
    length_of_stay = np.clip(rng.gamma(2.2, 1.8, n), 1, 30).round(0)
    num_diagnoses = np.clip(rng.poisson(5.5, n), 1, 20)
    num_medications = np.clip(rng.poisson(8, n), 0, 30)
    num_lab_procedures = np.clip(rng.normal(40, 18, n), 1, 120).round(0)
    num_procedures = np.clip(rng.poisson(1.5, n), 0, 10)

    # Comorbidities (binary flags)
    diabetes = rng.binomial(1, 0.30, n)
    heart_failure = rng.binomial(1, 0.18, n)
    copd = rng.binomial(1, 0.14, n)
    ckd = rng.binomial(1, 0.13, n)          # chronic kidney disease
    depression = rng.binomial(1, 0.16, n)
    cancer = rng.binomial(1, 0.08, n)

    comorbidity_score = (
        diabetes + heart_failure * 2 + copd * 1.5 + ckd * 1.5 + cancer * 2 + depression
    )

    discharge_disposition = rng.choice(
        ["Home", "Home Health Care", "Skilled Nursing Facility", "Against Medical Advice"],
        n, p=[0.62, 0.20, 0.15, 0.03]
    )

    admission_type = rng.choice(["Emergency", "Urgent", "Elective"], n, p=[0.55, 0.25, 0.20])

    polypharmacy = (num_medications >= 10).astype(int)
    has_pcp_followup_scheduled = rng.binomial(1, 0.55, n)  # follow-up already scheduled at discharge

    # ---- Latent risk score (drives the true probability of readmission) ----
    z = (
        -3.55
        + 0.028 * (age - 63)
        + 0.42 * num_prior_admissions_12mo
        + 0.33 * num_er_visits_12mo
        + 0.05 * length_of_stay
        + 0.10 * comorbidity_score
        + 0.35 * heart_failure
        + 0.25 * copd
        + 0.20 * ckd
        + 0.015 * num_medications
        + 0.30 * polypharmacy
        + 0.10 * (num_diagnoses - 5.5)
        - 0.45 * has_pcp_followup_scheduled
        + np.where(discharge_disposition == "Skilled Nursing Facility", 0.30, 0.0)
        + np.where(discharge_disposition == "Against Medical Advice", 0.75, 0.0)
        + np.where(insurance == "Medicaid", 0.20, 0.0)
        + np.where(insurance == "Uninsured", 0.35, 0.0)
        + np.where(admission_type == "Emergency", 0.20, 0.0)
        + rng.normal(0, 0.55, n)  # unexplained noise
    )
    prob = 1 / (1 + np.exp(-z))
    readmitted_30d = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age.astype(int),
        "gender": gender,
        "insurance": insurance,
        "admission_type": admission_type,
        "discharge_disposition": discharge_disposition,
        "length_of_stay": length_of_stay.astype(int),
        "num_prior_admissions_12mo": num_prior_admissions_12mo,
        "num_er_visits_12mo": num_er_visits_12mo,
        "num_diagnoses": num_diagnoses,
        "num_medications": num_medications,
        "num_lab_procedures": num_lab_procedures.astype(int),
        "num_procedures": num_procedures,
        "diabetes": diabetes,
        "heart_failure": heart_failure,
        "copd": copd,
        "ckd": ckd,
        "depression": depression,
        "cancer": cancer,
        "polypharmacy": polypharmacy,
        "pcp_followup_scheduled": has_pcp_followup_scheduled,
        "readmitted_30d": readmitted_30d,
    })

    # sprinkle a little missingness to make preprocessing realistic
    for col in ["num_lab_procedures", "pcp_followup_scheduled"]:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate()
    out_path = OUT_DIR / "hospital_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows -> {out_path}")
    print(f"Readmission base rate: {df['readmitted_30d'].mean():.3%}")
    print(df.dtypes)
