"""
app.py
------
Streamlit web app for the 30-Day Hospital Readmission Risk Predictor.

A care coordinator enters a patient's discharge details; the app returns:
  1. Predicted probability of 30-day readmission (using the best model
     selected in training, by ROC-AUC).
  2. A cost-aware intervention recommendation (enroll in transitional-care
     program or not), based on the optimal threshold learned in
     src/cost_optimizer.py.

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then point Streamlit Community Cloud
(share.streamlit.io) at app.py — no other setup needed.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
CONFIG_PATH = ROOT / "results" / "run_config.json"

st.set_page_config(
    page_title="30-Day Readmission Risk & Intervention Optimizer",
    page_icon="🏥",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return bundle["pipeline"], bundle["model_name"], config


pipeline, model_name, config = load_artifacts()
threshold = config["best_model_optimal_threshold"]
cost_intervention = config["cost_intervention"]
cost_readmission = config["cost_readmission"]
effectiveness = config["intervention_effectiveness"]

st.title("🏥 30-Day Readmission Risk & Intervention Optimizer")
st.caption(
    f"Model: **{model_name}** · Cost-optimal flag threshold: **{threshold:.2f}** "
    f"(intervention cost ${cost_intervention:,.0f} vs. avoidable readmission cost ${cost_readmission:,.0f})"
)
st.divider()

with st.form("patient_form"):
    st.subheader("Patient discharge details")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=65)
        gender = st.selectbox("Gender", ["F", "M"])
        insurance = st.selectbox("Insurance", ["Medicare", "Medicaid", "Private", "Uninsured"])
        admission_type = st.selectbox("Admission type", ["Emergency", "Urgent", "Elective"])
        discharge_disposition = st.selectbox(
            "Discharge disposition",
            ["Home", "Home Health Care", "Skilled Nursing Facility", "Against Medical Advice"],
        )
        length_of_stay = st.number_input("Length of stay (days)", min_value=1, max_value=60, value=4)
        num_prior_admissions_12mo = st.number_input("Prior admissions (last 12mo)", min_value=0, max_value=20, value=1)
        num_er_visits_12mo = st.number_input("ER visits (last 12mo)", min_value=0, max_value=20, value=1)
        num_diagnoses = st.number_input("Number of diagnoses", min_value=1, max_value=25, value=5)
        num_medications = st.number_input("Number of medications", min_value=0, max_value=40, value=8)

    with c2:
        num_lab_procedures = st.number_input("Number of lab procedures", min_value=0, max_value=150, value=40)
        num_procedures = st.number_input("Number of procedures", min_value=0, max_value=15, value=1)
        pcp_followup_scheduled = st.selectbox("Follow-up already scheduled at discharge?", ["Yes", "No"])
        st.markdown("**Comorbidities**")
        diabetes = st.checkbox("Diabetes")
        heart_failure = st.checkbox("Heart failure")
        copd = st.checkbox("COPD")
        ckd = st.checkbox("Chronic kidney disease")
        depression = st.checkbox("Depression")
        cancer = st.checkbox("Cancer")

    submitted = st.form_submit_button("Predict readmission risk", use_container_width=True)

if submitted:
    polypharmacy = 1 if num_medications >= 10 else 0

    patient = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "insurance": insurance,
        "admission_type": admission_type,
        "discharge_disposition": discharge_disposition,
        "length_of_stay": length_of_stay,
        "num_prior_admissions_12mo": num_prior_admissions_12mo,
        "num_er_visits_12mo": num_er_visits_12mo,
        "num_diagnoses": num_diagnoses,
        "num_medications": num_medications,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "diabetes": int(diabetes),
        "heart_failure": int(heart_failure),
        "copd": int(copd),
        "ckd": int(ckd),
        "depression": int(depression),
        "cancer": int(cancer),
        "polypharmacy": polypharmacy,
        "pcp_followup_scheduled": 1 if pcp_followup_scheduled == "Yes" else 0,
    }])

    prob = pipeline.predict_proba(patient)[0, 1]
    flagged = prob >= threshold

    st.divider()
    st.subheader("Result")

    col1, col2 = st.columns(2)
    col1.metric("Predicted 30-day readmission risk", f"{prob:.1%}")
    col2.metric("Cost-optimal decision threshold", f"{threshold:.0%}")

    if flagged:
        st.error(
            "🚩 **Recommend enrollment in transitional-care intervention.**\n\n"
            f"At this patient's predicted risk ({prob:.1%}), the expected cost of "
            f"*not* intervening (${prob * cost_readmission:,.0f}) exceeds the "
            f"expected cost of intervening "
            f"(${cost_intervention + prob * (1 - effectiveness) * cost_readmission:,.0f})."
        )
    else:
        st.success(
            "✅ **No intervention recommended** — predicted risk is below the "
            "cost-optimal threshold for this program. Standard discharge care applies."
        )

    with st.expander("How this recommendation is calculated"):
        st.markdown(
            f"""
- Intervention cost: **${cost_intervention:,.0f}** per enrolled patient
- Avoidable readmission cost: **${cost_readmission:,.0f}**
- Intervention effectiveness: **{effectiveness:.0%}** relative risk reduction
- Flag threshold **{threshold:.2f}** was chosen by minimizing total expected
  cost across the training population (see `src/cost_optimizer.py`) — it is
  *not* the default 0.5 classification cutoff.
            """
        )

st.divider()
st.caption(
    "Model trained on synthetic data for demonstration purposes. "
    "Replace `data/hospital_data.csv` with real (de-identified, IRB-approved) "
    "discharge data and re-run `src/train_models.py` before any clinical use."
)
