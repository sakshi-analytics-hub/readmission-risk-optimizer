"""
cost_optimizer.py
------------------
Turns a model's predicted readmission probabilities into a business decision:
"Which discharged patients should receive the (limited, costly) transitional
care intervention?"

Cost model (editable -- these are illustrative, defensible defaults):
  - COST_INTERVENTION: cost of enrolling one patient in a transitional-care /
    care-management program (nurse call, med reconciliation, follow-up visit).
  - COST_READMISSION: average marginal cost of an avoidable 30-day readmission
    to the hospital/payer (bed-day costs, procedures, potential CMS penalty).
  - INTERVENTION_EFFECTIVENESS: relative risk reduction the intervention
    provides IF the patient would have been readmitted (evidence from
    transitional-care trials typically shows 20-35% relative reduction).

For each candidate probability threshold t, a patient is flagged for
intervention if predicted P(readmit) >= t. Expected cost per patient:

  If flagged (predict positive):
      cost = COST_INTERVENTION + p * (1 - EFFECTIVENESS) * COST_READMISSION
  If not flagged (predict negative):
      cost = p * COST_READMISSION

We sum this over the (held-out) population for every threshold and pick the
threshold that minimizes total expected cost. We compare that against two
naive baselines: "treat nobody" and "treat everybody".
"""

import numpy as np
import pandas as pd

COST_INTERVENTION = 500.0          # $ per patient enrolled in the program
COST_READMISSION = 13500.0         # $ marginal cost of an avoidable readmission
INTERVENTION_EFFECTIVENESS = 0.28  # relative risk reduction from the program


def expected_cost_curve(y_prob, thresholds=None,
                         cost_intervention=COST_INTERVENTION,
                         cost_readmission=COST_READMISSION,
                         effectiveness=INTERVENTION_EFFECTIVENESS):
    """Total expected cost across the population for a range of thresholds."""
    y_prob = np.asarray(y_prob)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for t in thresholds:
        flagged = y_prob >= t
        cost_flagged = flagged.sum() * cost_intervention + \
            (y_prob[flagged] * (1 - effectiveness) * cost_readmission).sum()
        cost_unflagged = (y_prob[~flagged] * cost_readmission).sum()
        total = cost_flagged + cost_unflagged
        rows.append({
            "threshold": t,
            "n_flagged": int(flagged.sum()),
            "pct_flagged": flagged.mean(),
            "expected_total_cost": total,
            "expected_cost_per_patient": total / len(y_prob),
        })
    return pd.DataFrame(rows)


def optimal_threshold(y_prob, **cost_kwargs):
    curve = expected_cost_curve(y_prob, **cost_kwargs)
    best = curve.loc[curve["expected_total_cost"].idxmin()]
    return best, curve


def baseline_costs(y_prob, cost_intervention=COST_INTERVENTION,
                    cost_readmission=COST_READMISSION,
                    effectiveness=INTERVENTION_EFFECTIVENESS):
    """Treat-nobody and treat-everybody baselines for comparison."""
    y_prob = np.asarray(y_prob)
    n = len(y_prob)

    treat_none = (y_prob * cost_readmission).sum()
    treat_all = n * cost_intervention + (y_prob * (1 - effectiveness) * cost_readmission).sum()

    return {
        "treat_none_total_cost": treat_none,
        "treat_none_per_patient": treat_none / n,
        "treat_all_total_cost": treat_all,
        "treat_all_per_patient": treat_all / n,
    }


def summarize_savings(y_prob, **cost_kwargs):
    best_row, curve = optimal_threshold(y_prob, **cost_kwargs)
    base = baseline_costs(y_prob, **cost_kwargs)
    naive_best = min(base["treat_none_total_cost"], base["treat_all_total_cost"])
    savings_vs_naive = naive_best - best_row["expected_total_cost"]
    return {
        "optimal_threshold": best_row["threshold"],
        "pct_flagged_at_optimum": best_row["pct_flagged"],
        "expected_total_cost_at_optimum": best_row["expected_total_cost"],
        **base,
        "savings_vs_best_naive_strategy": savings_vs_naive,
    }, curve
