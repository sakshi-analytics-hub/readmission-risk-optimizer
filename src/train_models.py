"""
train_models.py
----------------
End-to-end training + evaluation of 7 classifiers for 30-day hospital
readmission prediction, plus cost-aware intervention threshold selection
per model.

Models: Logistic Regression, K-Nearest Neighbors, Decision Tree,
Random Forest, Support Vector Machine (RBF), Gaussian/Bernoulli Naive Bayes,
Gradient Boosting.

Outputs (written to ../results/):
  - metrics_summary.csv        : accuracy/precision/recall/F1/ROC-AUC/PR-AUC per model
  - roc_curves.png             : ROC curves, all models overlaid
  - pr_curves.png               : Precision-Recall curves, all models overlaid
  - confusion_matrices.png     : confusion matrix grid
  - feature_importance_rf.png  : Random Forest feature importances
  - feature_importance_gb.png  : Gradient Boosting feature importances
  - cost_curve_<model>.png     : expected-cost-vs-threshold curve per model
  - cost_summary.csv           : optimal threshold + savings per model
  - best_model.pkl             : pickled best model (by ROC-AUC) + preprocessor
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)

from cost_optimizer import summarize_savings, COST_INTERVENTION, COST_READMISSION, INTERVENTION_EFFECTIVENESS

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hospital_data.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

TARGET = "readmitted_30d"
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Load + split
# --------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])
    return preprocessor, numeric_features, categorical_features


# --------------------------------------------------------------------------
# 2. Model zoo
# --------------------------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=25, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=2.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


# --------------------------------------------------------------------------
# 3. Train + evaluate everything
# --------------------------------------------------------------------------
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor, num_feats, cat_feats = build_preprocessor(X)

    models = get_models()
    metrics_rows = []
    roc_data = {}
    pr_data = {}
    fitted_pipelines = {}
    cost_summary_rows = []

    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", clf),
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        metrics_rows.append({
            "model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc,
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)
        p, r, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (p, r)

        # ---- cost-aware threshold optimization on this model's test-set probs
        summary, curve = summarize_savings(y_prob)
        summary["model"] = name
        cost_summary_rows.append(summary)

        plt.figure(figsize=(6, 4))
        plt.plot(curve["threshold"], curve["expected_cost_per_patient"], color="#c0392b")
        plt.axvline(summary["optimal_threshold"], color="#2c3e50", linestyle="--",
                    label=f"optimal t={summary['optimal_threshold']:.2f}")
        plt.xlabel("Intervention threshold (flag if P(readmit) >= t)")
        plt.ylabel("Expected cost per patient ($)")
        plt.title(f"Cost-aware threshold curve — {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"cost_curve_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=130)
        plt.close()

        print(f"[{name}] ROC-AUC={roc_auc:.3f}  F1={f1:.3f}  "
              f"optimal_t={summary['optimal_threshold']:.2f}  "
              f"savings_vs_naive=${summary['savings_vs_best_naive_strategy']:,.0f}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    cost_df = pd.DataFrame(cost_summary_rows).sort_values("savings_vs_best_naive_strategy", ascending=False)
    cost_df.to_csv(RESULTS_DIR / "cost_summary.csv", index=False)

    # ---- ROC curve plot (all models) ----
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr) in roc_data.items():
        auc = metrics_df.loc[metrics_df.model == name, "roc_auc"].values[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — 30-Day Readmission Models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close()

    # ---- PR curve plot (all models) ----
    plt.figure(figsize=(7, 6))
    for name, (p, r) in pr_data.items():
        ap = metrics_df.loc[metrics_df.model == name, "pr_auc"].values[0]
        plt.plot(r, p, label=f"{name} (AP={ap:.3f})")
    baseline = y_test.mean()
    plt.axhline(baseline, color="k", linestyle="--", alpha=0.4, label=f"baseline prevalence={baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — 30-Day Readmission Models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pr_curves.png", dpi=150)
    plt.close()

    # ---- confusion matrices grid ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, (name, pipe) in enumerate(fitted_pipelines.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No Readmit", "Readmit"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No Readmit", "Readmit"], fontsize=8)
        for r_ in range(2):
            for c_ in range(2):
                ax.text(c_, r_, cm[r_, c_], ha="center", va="center",
                        color="white" if cm[r_, c_] > cm.max() / 2 else "black")
    for j in range(len(fitted_pipelines), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=140)
    plt.close()

    # ---- feature importances (tree models) ----
    def plot_importance(pipe, title, fname, top_n=15):
        model = pipe.named_steps["model"]
        pre = pipe.named_steps["preprocess"]
        feat_names = pre.get_feature_names_out()
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(order)), importances[order][::-1], color="#2c7fb8")
        plt.yticks(range(len(order)), [feat_names[i] for i in order][::-1], fontsize=8)
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / fname, dpi=140)
        plt.close()

    plot_importance(fitted_pipelines["Random Forest"], "Random Forest — Feature Importance", "feature_importance_rf.png")
    plot_importance(fitted_pipelines["Gradient Boosting"], "Gradient Boosting — Feature Importance", "feature_importance_gb.png")

    # ---- save best model (by ROC-AUC) ----
    best_name = metrics_df.iloc[0]["model"]
    best_pipe = fitted_pipelines[best_name]
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump({"model_name": best_name, "pipeline": best_pipe}, f)

    best_model_threshold = float(
        cost_df.loc[cost_df["model"] == best_name, "optimal_threshold"].values[0]
    )

    with open(RESULTS_DIR / "run_config.json", "w") as f:
        json.dump({
            "best_model_by_roc_auc": best_name,
            "best_model_optimal_threshold": best_model_threshold,
            "cost_intervention": COST_INTERVENTION,
            "cost_readmission": COST_READMISSION,
            "intervention_effectiveness": INTERVENTION_EFFECTIVENESS,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "base_rate_test": float(y_test.mean()),
        }, f, indent=2)

    print("\n=== METRICS SUMMARY ===")
    print(metrics_df.to_string(index=False))
    print("\n=== COST-AWARE INTERVENTION SUMMARY ===")
    print(cost_df.to_string(index=False))
    print(f"\nBest model by ROC-AUC: {best_name}  -> saved to {MODELS_DIR/'best_model.pkl'}")


if __name__ == "__main__":
    main()
