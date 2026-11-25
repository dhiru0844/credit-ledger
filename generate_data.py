"""
Generate a realistic SYNTHETIC retail-loan dataset (amounts in Indian rupees).

    python generate_data.py          -> data/credit_risk_dataset.csv

IMPORTANT - what this is and is not
-----------------------------------
This is simulated data. Nobody's real loan history is in it. The relationships below are
my design assumptions about how retail credit behaves, chosen to match general patterns
seen in real lending. They are NOT fitted to any specific bank's data. So the model's
scores measure how well it recovers these assumptions plus noise, not real-world
performance. To validate for real use, retrain on genuine historical loan data.

Design goals (each fixes a flaw of the old Kaggle-style file)
-------------------------------------------------------------
* Risk rises SMOOTHLY with loan-to-income (no cliff, no 100%-default groups).
* Plenty of irreducible noise: two similar applicants can have different outcomes.
  Result: a good model reaches AUC ~0.8, like real credit models, not ~0.95.
* Default rate ~11 % overall (unsecured personal lending is typically single digits to
  mid-teens, depending on market and segment).
* Loan grade / interest rate are the LENDER'S own risk assessment (a noisy proxy for
  bureau score and behaviour) - they carry information but are not the cause.
* Rupee scale: median income ~Rs 5.5 lakh, loans Rs 10,000 .. Rs 30 lakh.
* Some missing values (employment length, interest rate), as in real data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2024
N_ROWS = 60_000
TARGET_DEFAULT_RATE = 0.11
OUT = Path(__file__).parent / "data" / "credit_risk_dataset.csv"

MEDIAN_INCOME = 550_000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=N_ROWS, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- who the applicant is -------------------------------------------------------
    age = np.clip(np.round(21 + rng.gamma(3.0, 4.6, n)), 21, 65).astype(int)
    emp = np.floor(np.clip(rng.gamma(2.0, 3.4, n), 0, age - 20)).astype(float)

    log_income = (np.log(MEDIAN_INCOME) + 0.02 * (np.minimum(age, 50) - 34)
                  + 0.015 * (emp - 6) + rng.normal(0, 0.55, n))
    income = np.clip(np.round(np.exp(log_income), -3), 60_000, 6_000_000)

    z_inc = np.log(income) - np.log(MEDIAN_INCOME)
    own = -3.0 + 0.06 * (age - 30) + 0.4 * z_inc
    mort = -1.3 + 0.03 * (age - 30) + 0.5 * z_inc
    other = np.full(n, -2.6)
    logits = np.column_stack([np.zeros(n), own, mort, other])        # RENT is baseline
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    cum = probs.cumsum(1)
    home = np.array(["RENT", "OWN", "MORTGAGE", "OTHER"])[(rng.random(n)[:, None] > cum).sum(1)]

    intents = np.array(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    intent = rng.choice(intents, n, p=[0.26, 0.17, 0.16, 0.12, 0.13, 0.16])

    # ---- hidden creditworthiness (bureau score / repayment behaviour) ---------------
    u = rng.normal(0, 1, n)            # higher = better. Never given to the model.

    hist = np.minimum(2 + np.floor(rng.gamma(2.0, 3.0, n)), age - 18).astype(int)
    prior = (rng.random(n) < sigmoid(-2.3 - 0.9 * u)).astype(int)

    # ---- how much they borrow -------------------------------------------------------
    size_mult = {"PERSONAL": 0.9, "EDUCATION": 1.3, "MEDICAL": 0.9, "VENTURE": 1.2,
                 "HOMEIMPROVEMENT": 1.4, "DEBTCONSOLIDATION": 1.0}
    ratio_raw = np.clip(rng.lognormal(np.log(0.20), 0.65, n)
                        * np.vectorize(size_mult.get)(intent), 0.02, 1.0)
    loan = np.clip(np.round(income * ratio_raw / 5_000) * 5_000, 10_000, 3_000_000)
    loan = np.minimum(loan, income)                       # never more than 100 % of income
    ratio = np.round(loan / income, 2)

    # ---- true probability of default (the hidden rule the model must discover) ------
    home_eff = pd.Series(home).map({"RENT": 0.25, "OTHER": 0.45, "MORTGAGE": 0.0, "OWN": -0.35}).values
    intent_eff = pd.Series(intent).map({"VENTURE": 0.45, "MEDICAL": 0.30, "DEBTCONSOLIDATION": 0.35,
                                        "PERSONAL": 0.10, "EDUCATION": -0.10,
                                        "HOMEIMPROVEMENT": -0.20}).values
    core = (1.10 * prior - 0.85 * u
            + 3.2 * ratio + 2.0 * ratio ** 2           # smooth, convex burden effect
            - 0.30 * z_inc
            - 0.035 * np.minimum(emp, 20) - 0.03 * np.minimum(hist, 20)
            + 0.04 * np.maximum(0, 28 - age)           # very young borrowers a bit riskier
            + home_eff + intent_eff
            + rng.normal(0, 0.6, n))                   # life events nobody can predict

    lo, hi = -12.0, 4.0                                # calibrate intercept to target rate
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(core + mid).mean() > TARGET_DEFAULT_RATE:
            hi = mid
        else:
            lo = mid
    p_default = sigmoid(core + (lo + hi) / 2)
    default = (rng.random(n) < p_default).astype(int)

    # ---- the lender's grade and interest rate (noisy proxy, not the cause) ----------
    grade_score = -0.9 * u + 0.5 * prior + 0.4 * ratio + rng.normal(0, 0.8, n)
    cuts = np.quantile(grade_score, np.cumsum([0.24, 0.28, 0.22, 0.14, 0.08, 0.03]))
    grade_idx = np.searchsorted(cuts, grade_score)
    grade = np.array(list("ABCDEFG"))[grade_idx]
    base_rate = np.array([10.5, 12.5, 14.5, 17.0, 19.5, 22.0, 24.0])[grade_idx]
    rate = np.round(np.clip(base_rate + 1.5 * ratio + rng.normal(0, 0.8, n), 8.0, 30.0), 2)

    # ---- real data has holes ---------------------------------------------------------
    emp_out = emp.copy()
    emp_out[rng.random(n) < 0.03] = np.nan
    rate[rng.random(n) < 0.05] = np.nan

    return pd.DataFrame({
        "person_age": age,
        "person_income": income.astype(int),
        "person_home_ownership": home,
        "person_emp_length": emp_out,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": loan.astype(int),
        "loan_int_rate": rate,
        "loan_status": default,
        "loan_percent_income": ratio,
        "cb_person_default_on_file": np.where(prior == 1, "Y", "N"),
        "cb_person_cred_hist_length": hist,
    })


if __name__ == "__main__":
    df = generate()
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT}  |  default rate {df.loan_status.mean():.1%}")
