# VIVA Q&A — AI-Based Loan Default Prediction
### Project: Credit Risk Prediction System (XGBoost)

---

## Q1: How does the model predict that a person will default?

The model outputs a **probability score between 0 and 1** — the Probability of Default (PD).

| Probability | Risk Label | Decision |
|---|---|---|
| prob < 0.40 | LOW RISK | **Repay** |
| 0.40 ≤ prob < 0.60 | MEDIUM RISK | **Review** |
| 0.60 ≤ prob < 0.80 | HIGH RISK | **Default** |
| prob ≥ 0.80 | VERY HIGH RISK | **Default** |

The decision threshold is not hardcoded at 0.5. It is determined by **Youden's J statistic** (maximising TPR − FPR on the test set) and saved to `model_metrics.json`. This gives better recall for default detection.

Additionally, a **Business Override Rule** auto-flags as High Risk if:
```
Loan Amount > 5 × Annual Income  (regardless of model probability)
```

---

## Q2: Why XGBoost and not other models?

| Property | Reason |
|---|---|
| Handles tabular data natively | Credit data is tabular, not image/text — XGBoost is optimal here |
| Built-in feature importance | Direct SHAP compatibility via TreeExplainer |
| Handles class imbalance | `scale_pos_weight` parameter adjusts for minority class |
| Robust to outliers | Tree-based splitting is not sensitive to outlier values |
| Fast training + inference | Can be deployed on modest hardware |

We use **a single XGBoost model** rather than comparing multiple models. Multi-model comparison adds complexity without guaranteeing that the final saved model is XGBoost — which is the requirement.

---

## Q3: What features are used and why?

| Feature | Why it matters |
|---|---|
| `loan_amnt` | Larger loans = harder to repay |
| `int_rate` | High interest = larger repayment burden |
| `annual_inc` | Low income = cannot afford repayments |
| `dti` | Already over-leveraged |
| `fico_range_low` | Credit score = historical repayment behaviour |
| `installment` | Monthly burden vs income |
| `revol_bal` | High revolving balance = financial stress |
| `revol_util` | High utilization = near credit limit |
| `delinq_2yrs` | Past missed payments = strong predictor |
| `pub_rec` | Bankruptcies / civil judgments = very high risk |
| `inq_last_6mths` | Many inquiries = desperate for credit |
| `grade` / `sub_grade` | Lender's own internal risk grading |
| `purpose` | Debt consolidation applicants historically default more |
| `home_ownership` | Asset backing reduces default risk |
| `emp_length` | Longer employment = income stability |

### Engineered features (derived from raw inputs)

```python
loan_to_income           = loan_amnt / annual_inc
installment_to_income    = installment / annual_inc
credit_utilization       = revol_bal / (revol_bal + bc_open_to_buy)
payment_capacity         = annual_inc - (installment × 12)   # free cash flow
credit_stress            = dti × loan_amnt                   # combined leverage
high_dti_flag            = 1 if dti > 20 else 0
low_fico_flag            = 1 if fico < 600 else 0
recent_inquiries_flag    = 1 if inq_last_6mths > 3 else 0
fico_avg                 = (fico_range_low + fico_range_high) / 2
risk_score               = int_rate × dti
```

---

## Q4: How does SMOTE work and why is it applied only on the training split?

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic examples of the minority class (default = 1) to balance the training distribution.

**Why training-split only:** Applying SMOTE before the train-test split causes **data leakage** — the test set statistics contaminate the scaler/oversampler, making evaluation results artificially optimistic. The correct order is:
```
Split → SMOTE on X_train/y_train only → train model → evaluate on untouched X_test/y_test
```

---

## Q5: How does SHAP explain the prediction?

SHAP (SHapley Additive exPlanations) assigns each feature a value that represents its contribution to the model output for that specific prediction.

- **Positive SHAP value** → pushes probability higher (toward default)
- **Negative SHAP value** → pushes probability lower (toward repay)
- Uses **TreeExplainer** for XGBoost — computationally exact, not approximate

The top 5 SHAP drivers are shown on the result page for every prediction.

---

## Q6: How does the drift detection work?

**Population Stability Index (PSI)** measures how much a feature's distribution has shifted between the training data (reference) and new incoming prediction data.

| PSI | Status |
|---|---|
| < 0.10 | No Drift |
| 0.10 – 0.25 | Moderate Drift |
| ≥ 0.25 | High Drift → triggers retraining |

PSI is checked every 10th prediction. If High Drift is detected, `retrain_model()` is called automatically.

---

## Q7: What is the governance / audit trail?

Every prediction is appended to `logs/audit_log.json` by `governance.py`. Each record contains:
- Timestamp, raw input, prediction, probability
- Threshold used, risk level, override flag
- SHAP top features and advice strings

The log is append-only (capped at 10,000 entries) and uses an absolute file path so it works regardless of which directory the Flask app is launched from.

---

## Q8: How is the feedback loop implemented?

After every 100 predictions, `feedback_loop.py` builds a training dataset from the prediction history and appends it to the processed CSV. The model is then retrained automatically.

**Important caveat:** The ground-truth label used is the model's own prediction (`prediction_numeric`), not the actual repayment outcome (which would only be known months later). In production, this should be replaced with real outcome labels once they become available.

---

## Q9: How is the application deployed?

- **Local:** `python -m webapp.app` (Flask development server)
- **Docker:** `docker compose up --build` (Gunicorn + 2 workers)
- **Gunicorn command:** `gunicorn --bind 0.0.0.0:5000 --workers 2 webapp.app:app`

The Gunicorn command references `webapp.app:app` (module path), not `app:app` — the latter only works when CWD is the `webapp/` directory.
