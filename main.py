"""
Credit-risk API.   Run:  uvicorn main:app --reload
Docs:  http://127.0.0.1:8000/docs      UI:  http://127.0.0.1:8000/
"""
import re
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from train import DATA_PATH, load_and_clean

ROOT = Path(__file__).parent
ARTIFACT_PATH = ROOT / "models" / "credit_risk_model.joblib"

# Optional policy rule (not machine learning): a loan at or above this share of annual
# income is always flagged High Risk, whatever the model says. DISABLED by default
# because the model now learns the loan/income effect smoothly; set e.g. 0.8 to enable.
MAX_LOAN_TO_INCOME = None

# Inputs where we warn if the value is outside what the model saw in training.
MONEY_COLUMNS = {"person_income", "loan_amnt"}
RANGE_CHECKED = {
    "person_income": "Annual income",
    "loan_amnt": "Loan amount",
    "loan_percent_income": "Loan-to-income ratio",
    "loan_int_rate": "Interest rate",
    "person_emp_length": "Employment length",
    "cb_person_cred_hist_length": "Credit history length",
}

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(f"{ARTIFACT_PATH} not found. Run `python train.py` first.")
    artifact = joblib.load(ARTIFACT_PATH)

    # A model pickled under different library versions can load silently and
    # behave wrongly -- warn loudly instead.
    current = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
    for lib, ver in current.items():
        trained = artifact["versions"].get(lib)
        if trained != ver:
            warnings.warn(f"{lib}: model trained with {trained}, running {ver}. "
                          f"Re-run `python train.py` to be safe.")

    state.update(artifact)
    # Training-data ranges, used to warn when an input is outside what the model has seen.
    train_df = load_and_clean(DATA_PATH)
    state["ranges"] = {c: (float(train_df[c].min()), float(train_df[c].max()))
                       for c in RANGE_CHECKED}
    if MAX_LOAN_TO_INCOME is not None:
        over = train_df[train_df["loan_percent_income"] >= MAX_LOAN_TO_INCOME]
        state["policy_default_rate"] = float(over["loan_status"].mean())
    print(f"Loaded {artifact['model_name']} | threshold={artifact['threshold']:.3f}")
    yield
    state.clear()


app = FastAPI(title="Credit Risk API", version="3.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    """Return `detail` as a readable string (the frontend shows it directly)."""
    msg = "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=422, content={"detail": msg})


class LoanApplication(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0, description="Annual income")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: Optional[float] = Field(default=None, ge=0, le=60,
                                               description="Years employed; may be omitted")
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                         "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0)
    loan_int_rate: Optional[float] = Field(default=None, ge=0, le=40,
                                           description="Percent; may be omitted")
    # Accepted for backward compatibility with the old frontend but IGNORED:
    # the server always recomputes it as loan_amnt / person_income.
    loan_percent_income: Optional[float] = None
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0, le=60)


def _inr(x: float) -> str:
    """Rupee amount with Indian digit grouping, e.g. 1250000 -> Rs 12,50,000."""
    digits = f"{abs(x):.0f}"
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        digits = f"{head},{tail}"
    return f"₹{digits}"


def _fmt(col: str, x: float) -> str:
    return _inr(x) if col in MONEY_COLUMNS else f"{x:g}"


@app.get("/health")
def health():
    return {"status": "ok", "model": state.get("model_name"), "threshold": state.get("threshold")}


@app.post("/predict")
def predict(app_in: LoanApplication):
    row = app_in.model_dump()
    ratio = round(row["loan_amnt"] / row["person_income"], 2)
    row["loan_percent_income"] = ratio
    # None -> NaN: XGBoost handles missing numeric values natively
    frame = pd.DataFrame([row])[state["features"]].astype(
        {c: float for c in ("person_emp_length", "loan_int_rate")}
    )
    frame = frame.replace({None: np.nan})

    try:
        probability = float(state["model"].predict_proba(frame)[:, 1][0])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    threshold = float(state["threshold"])
    model_high = probability >= threshold
    notes = []

    # Safety check 1: is the applicant inside the range the model was trained on?
    for col, label in RANGE_CHECKED.items():
        value = row[col]
        lo, hi = state["ranges"][col]
        if value is not None and not (lo <= value <= hi):
            notes.append(
                f"{label} ({_fmt(col, value)}) is outside the range this model was trained on "
                f"({_fmt(col, lo)} to {_fmt(col, hi)}), so its score may be unreliable."
            )

    # Safety check 2: policy rule on loan size relative to income
    policy_high = MAX_LOAN_TO_INCOME is not None and ratio >= MAX_LOAN_TO_INCOME
    if policy_high:
        notes.append(
            f"Loan is {ratio:.0%} of annual income. Applicants asking for {MAX_LOAN_TO_INCOME:.0%} "
            f"or more defaulted {state['policy_default_rate']:.0%} of the time in the training "
            f"data, so this is flagged High Risk regardless of the model score."
        )

    high_risk = model_high or policy_high
    return {
        "default_probability": probability,          # the model's own estimate, unmodified
        "default_prediction": int(high_risk),
        "threshold": threshold,
        "Result": "High Risk" if high_risk else "Low Risk",
        "decision_source": "policy_rule" if (policy_high and not model_high) else "model",
        "notes": notes,
    }


# Frontend (must be mounted last so it doesn't shadow the API routes)
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
