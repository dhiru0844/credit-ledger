"""
Train the credit-risk default model.

    python train.py

Design rules (each one fixes a problem found in the previous version):
  1. The test set is split off FIRST and touched exactly ONCE, at the very end.
  2. Model selection and the decision threshold are chosen using only
     cross-validation / out-of-fold predictions on the TRAINING data.
  3. No class weighting -> predicted probabilities stay honest (well calibrated).
     The imbalance is handled by the decision threshold instead.
  3b. The best model is chosen by cross-validation among ALL candidates (logistic
     regression included); a simple model is allowed to win.
  4. Bad values are set to NaN instead of dropping rows; XGBoost handles NaN natively.
  5. Model + threshold + feature list + library versions are saved as ONE artifact,
     so they can never get out of sync.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "credit_risk_dataset.csv"
MODEL_PATH = ROOT / "models" / "credit_risk_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
REPORT_PATH = ROOT / "reports" / "evaluation.png"

SEED = 42
TARGET = "loan_status"
NUMERIC = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CATEGORICAL = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]
FEATURES = NUMERIC + CATEGORICAL

# F-beta used to pick the threshold. beta=1 -> balance precision & recall.
# beta>1 favours catching defaulters (recall); beta<1 favours not wrongly rejecting.
THRESHOLD_BETA = 1.0


# --------------------------------------------------------------------------- data
def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)

    # Impossible values become missing (row is kept) instead of deleting the row.
    df.loc[~df["person_age"].between(18, 100), "person_age"] = np.nan
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = np.nan
    df.loc[df["person_emp_length"] > df["person_age"], "person_emp_length"] = np.nan
    df = df.dropna(subset=["person_age"])  # an applicant with no valid age is unusable

    # Derive this feature exactly the way the API derives it, so train == serve.
    df["loan_percent_income"] = (df["loan_amnt"] / df["person_income"]).round(2)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------- pipelines
def one_hot():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def xgb_pipeline(**params) -> Pipeline:
    pre = ColumnTransformer(
        [("num", "passthrough", NUMERIC), ("cat", one_hot(), CATEGORICAL)]
    )  # NaN in numeric columns is handled natively by XGBoost
    model = XGBClassifier(
        random_state=SEED, n_jobs=-1, eval_metric="logloss", tree_method="hist", **params
    )
    return Pipeline([("pre", pre), ("model", model)])


def logreg_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", one_hot(), CATEGORICAL),
    ])
    return Pipeline([("pre", pre),
                     ("model", LogisticRegression(max_iter=2000, random_state=SEED))])


# --------------------------------------------------------------------------- utils
def cv_scores(pipe, X, y, cv):
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return oof, roc_auc_score(y, oof), average_precision_score(y, oof)


def pick_threshold(y_true, proba, beta=THRESHOLD_BETA) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    b2 = beta ** 2
    fbeta = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)
    return float(thr[np.argmax(fbeta)])


def ks_statistic(y_true, proba) -> float:
    """Kolmogorov-Smirnov: max gap between defaulter and non-defaulter score curves."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------------------- main
def main():
    df = load_and_clean(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    print(f"Clean rows: {len(df):,} | default rate: {y.mean():.1%}")

    # 1) lock the test set away
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 2) compare candidates with cross-validation on TRAIN only
    candidates = {
        "logistic_regression": logreg_pipeline(),
        "xgboost_default": xgb_pipeline(n_estimators=300, max_depth=5, learning_rate=0.1),
    }
    results = {}
    print("\nCross-validation on the training set (5-fold):")
    for name, pipe in candidates.items():
        _, auc, ap = cv_scores(pipe, X_tr, y_tr, cv)
        results[name] = {"cv_roc_auc": round(auc, 4), "cv_pr_auc": round(ap, 4)}
        print(f"  {name:22s} ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}")

    # 3) modest hyper-parameter search (inside train only)
    search = RandomizedSearchCV(
        xgb_pipeline(),
        {
            "model__n_estimators": randint(200, 700),
            "model__max_depth": randint(2, 7),
            "model__learning_rate": uniform(0.02, 0.13),
            "model__subsample": uniform(0.7, 0.3),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": uniform(1, 4),
        },
        n_iter=30, cv=cv, scoring="average_precision", n_jobs=1, random_state=SEED,
    )
    search.fit(X_tr, y_tr)
    results["xgboost_tuned"] = {
        "cv_pr_auc": round(search.best_score_, 4),
        "params": {k.replace("model__", ""): (float(v) if isinstance(v, float) else int(v))
                   for k, v in search.best_params_.items()},
    }
    print(f"  {'xgboost_tuned':22s} PR-AUC={search.best_score_:.4f}  <- best of 30 random configs")

    # pick the best candidate by CV PR-AUC (tuning is allowed to lose; so is XGBoost)
    fitted = {
        "logistic_regression": candidates["logistic_regression"],
        "xgboost_default": candidates["xgboost_default"],
        "xgboost_tuned": search.best_estimator_,
    }
    best_name = max(fitted, key=lambda k: results[k]["cv_pr_auc"])
    final_pipe = fitted[best_name]
    print(f"  -> selected by CV PR-AUC: {best_name}")

    # 4) choose the threshold from OUT-OF-FOLD predictions on TRAIN (not on test)
    oof, _, _ = cv_scores(final_pipe, X_tr, y_tr, cv)
    threshold = pick_threshold(y_tr, oof)
    print(f"\nThreshold (max F{THRESHOLD_BETA:g} on out-of-fold train predictions): {threshold:.3f}")

    # 5) fit on the full training set, then evaluate on the test set ONCE
    final_pipe.fit(X_tr, y_tr)
    p_te = final_pipe.predict_proba(X_te)[:, 1]
    test = {
        "roc_auc": round(roc_auc_score(y_te, p_te), 4),
        "pr_auc": round(average_precision_score(y_te, p_te), 4),
        "gini": round(2 * roc_auc_score(y_te, p_te) - 1, 4),
        "ks": round(ks_statistic(y_te, p_te), 4),
        "brier": round(brier_score_loss(y_te, p_te), 4),
        "at_chosen_threshold": classification_metrics(y_te, p_te, threshold),
        "at_0.50": classification_metrics(y_te, p_te, 0.5),
    }
    base_rate = float(y_te.mean())
    print("\nHELD-OUT TEST SET (used once):")
    print(f"  ROC-AUC={test['roc_auc']}  Gini={test['gini']}  KS={test['ks']}  "
          f"PR-AUC={test['pr_auc']}  Brier={test['brier']}")
    print(f"  default rate in test set: {base_rate:.1%} -> always-predict-no-default accuracy "
          f"would be {1 - base_rate:.3f}, PR-AUC of a random model would be {base_rate:.3f}")
    for k in ("at_chosen_threshold", "at_0.50"):
        m = test[k]
        print(f"  {k:20s} thr={m['threshold']:.2f} acc={m['accuracy']:.3f} "
              f"prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f}  {m['confusion']}")

    # 6) save ONE artifact
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": final_pipe,
        "threshold": threshold,
        "features": FEATURES,
        "model_name": best_name,
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(
        {"selected_model": best_name, "cv": results, "test": test,
         "n_train": len(X_tr), "n_test": len(X_te)}, indent=2))
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")

    make_report(y_te.values, p_te, threshold, final_pipe)


def make_report(y_true, proba, threshold, pipe):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
    except ImportError:
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    prec, rec, _ = precision_recall_curve(y_true, proba)
    ax[0, 0].plot(rec, prec)
    pt = classification_metrics(y_true, proba, threshold)
    ax[0, 0].scatter([pt["recall"]], [pt["precision"]], c="red", zorder=3, label=f"threshold {threshold:.2f}")
    ax[0, 0].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[0, 0].legend()

    cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
    ax[0, 1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[0, 1].text(j, i, v, ha="center", va="center", fontsize=14)
    ax[0, 1].set(title="Confusion matrix (test)", xticks=[0, 1], yticks=[0, 1],
                 xlabel="Predicted", ylabel="Actual")

    frac, mean = calibration_curve(y_true, proba, n_bins=10)
    ax[1, 0].plot([0, 1], [0, 1], "k--", label="perfect")
    ax[1, 0].plot(mean, frac, "o-", label="model")
    ax[1, 0].set(title="Calibration (test)", xlabel="Predicted probability", ylabel="Observed default rate")
    ax[1, 0].legend()

    names = pipe.named_steps["pre"].get_feature_names_out()
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        vals, title = est.feature_importances_, "Top 10 features (XGBoost gain)"
    else:
        vals, title = np.abs(est.coef_[0]), "Top 10 features (|coefficient|, standardised)"
    imp = pd.Series(vals, index=names).nlargest(10)[::-1]
    imp.index = [n.split("__", 1)[1] for n in imp.index]
    ax[1, 1].barh(imp.index, imp.values)
    ax[1, 1].set(title=title)

    fig.tight_layout()
    REPORT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(REPORT_PATH, dpi=110)
    print(f"Saved {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
