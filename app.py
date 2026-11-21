import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

with open(os.path.join(BASE_DIR, "decision.pkl"), "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# Label encoding (alphabetical, as produced by sklearn LabelEncoder)
GENDER = {"female": 0, "male": 1}
EDUCATION = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}
HOME = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
INTENT = {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2,
          "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5}
DEFAULTS = {"No": 0, "Yes": 1}

# field -> (min, max, label)
NUMERIC = {
    "person_age": (18, 100, "Age"),
    "person_income": (1, 10_000_000, "Annual income"),
    "person_emp_exp": (0, 80, "Work experience"),
    "loan_amnt": (1, 5_000_000, "Loan amount"),
    "loan_int_rate": (0, 40, "Interest rate"),
    "cb_person_cred_hist_length": (0, 80, "Credit history length"),
    "credit_score": (300, 900, "Credit score"),
}
CATEGORICAL = {
    "person_gender": GENDER,
    "person_education": EDUCATION,
    "person_home_ownership": HOME,
    "loan_intent": INTENT,
    "previous_loan_defaults_on_file": DEFAULTS,
}


def parse(payload):
    row, errors = {}, {}
    for field, (lo, hi, label) in NUMERIC.items():
        try:
            value = float(payload.get(field))
        except (TypeError, ValueError):
            errors[field] = f"{label} must be a number."
            continue
        if not lo <= value <= hi:
            errors[field] = f"{label} must be between {lo:,} and {hi:,}."
        row[field] = value
    for field, mapping in CATEGORICAL.items():
        choice = payload.get(field)
        if choice not in mapping:
            errors[field] = "Choose one of the listed options."
        else:
            row[field] = mapping[choice]
    if not errors:
        row["loan_percent_income"] = round(row["loan_amnt"] / row["person_income"], 2)
    return row, errors


@app.route("/")
def index():
    return render_template(
        "index.html",
        education=list(EDUCATION),
        home=list(HOME),
        intent=list(INTENT),
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    row, errors = parse(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    frame = pd.DataFrame([row])[FEATURES]
    approved = int(model.predict(frame)[0]) == 1
    proba = model.predict_proba(frame)[0]
    confidence = float(proba[1] if approved else proba[0])

    return jsonify({
        "ok": True,
        "approved": approved,
        "confidence": round(confidence * 100),
        "loan_percent_income": round(row["loan_percent_income"] * 100),
    })


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
