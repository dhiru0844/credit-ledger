"""Tests.  Run:  pytest -q   (after `python generate_data.py` and `python train.py`)"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402

# Amounts are in rupees, matching data/credit_risk_dataset.csv
LOW_RISK = dict(person_age=35, person_income=1_200_000, person_home_ownership="MORTGAGE",
                person_emp_length=10, loan_intent="HOMEIMPROVEMENT", loan_grade="A",
                loan_amnt=150_000, loan_int_rate=10.5, cb_person_default_on_file="N",
                cb_person_cred_hist_length=12)
HIGH_RISK = dict(person_age=22, person_income=240_000, person_home_ownership="RENT",
                 person_emp_length=1, loan_intent="MEDICAL", loan_grade="F",
                 loan_amnt=200_000, loan_int_rate=22, cb_person_default_on_file="Y",
                 cb_person_cred_hist_length=2)


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:      # runs lifespan -> loads the model
        yield c


def predict(client, payload):
    return client.post("/predict", json=payload).json()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_low_risk_applicant(client):
    r = predict(client, LOW_RISK)
    assert r["Result"] == "Low Risk" and r["default_probability"] < 0.05


def test_high_risk_applicant(client):
    r = predict(client, HIGH_RISK)
    assert r["Result"] == "High Risk" and r["default_probability"] > 0.5


def test_risk_rises_with_loan_size(client):
    """Same person, bigger loan -> higher risk (the effect that looked wrong in v1)."""
    base = {**LOW_RISK, "person_income": 100_000, "person_home_ownership": "OWN"}
    small = predict(client, {**base, "loan_amnt": 10_000})["default_probability"]
    large = predict(client, {**base, "loan_amnt": 80_000})["default_probability"]
    assert large > 5 * small


def test_missing_optional_fields_ok(client):
    p = {k: v for k, v in LOW_RISK.items() if k not in ("person_emp_length", "loan_int_rate")}
    assert client.post("/predict", json=p).status_code == 200


def test_old_frontend_payload_still_works(client):
    assert client.post("/predict", json={**LOW_RISK, "loan_percent_income": 0.12}).status_code == 200


@pytest.mark.parametrize("bad", [{"person_age": 17}, {"person_age": 144},
                                 {"person_income": 0}, {"loan_grade": "Z"},
                                 {"person_emp_length": 123}])
def test_invalid_input_rejected(client, bad):
    assert client.post("/predict", json={**LOW_RISK, **bad}).status_code == 422


def test_normal_applicant_has_no_notes(client):
    assert predict(client, LOW_RISK)["notes"] == []


def test_out_of_range_loan_is_warned_in_rupees(client):
    r = predict(client, {**LOW_RISK, "loan_amnt": 10_000_000})
    assert any("outside the range" in n and "₹1,00,00,000" in n for n in r["notes"])


def test_loan_larger_than_income_is_warned(client):
    r = predict(client, {**LOW_RISK, "loan_amnt": 10_000_000})
    assert any("Loan-to-income ratio" in n for n in r["notes"])


def test_optional_policy_rule(client, monkeypatch):
    """The rule is off by default; when enabled it forces High Risk."""
    safe = {**LOW_RISK, "person_income": 1_000_000, "loan_amnt": 600_000}   # ratio 0.60
    monkeypatch.setattr(main, "MAX_LOAN_TO_INCOME", 0.5)
    monkeypatch.setitem(main.state, "policy_default_rate", 0.4)
    r = predict(client, safe)
    assert r["Result"] == "High Risk"
    assert any("60%" in n for n in r["notes"])
