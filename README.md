# AegisBank — AI-Based Loan Default Prediction System

> Predicting loan defaults in real time using XGBoost, SHAP explainability, drift detection, and a Flask web interface.

---

## 📌 One Line Description

An explainable AI system that predicts whether a loan applicant will default, built with XGBoost, SHAP, Flask, and Docker.

---

## 🎯 Problem It Solves

- Banks lose billions approving loans that never get repaid
- Traditional credit scoring is static, slow, and often unfair
- Black box AI models cannot explain their decisions
- Models degrade over time without monitoring
- This system solves all four problems in one application

---

## ⭐ What Makes This Different

- Not just a prediction — gives the **TOP 5 REASONS** behind every decision
- Conservative threshold (0.40) tuned for banking reality
- Auto-calculates DTI, installment, revolving balance from user inputs
- Business override rule catches extreme cases the model might miss
- Every decision is logged for regulatory compliance
- Model retrains itself automatically when data drift is detected

---

## 🛠️ Tech Stack

| Layer          | Technology                        |
|----------------|-----------------------------------|
| ML Model       | XGBoost (Gradient Boosted Trees)  |
| Explainability | SHAP (TreeSHAP)                   |
| Imbalance Fix  | SMOTE + scale_pos_weight          |
| Backend        | Python 3.9+, Flask                |
| Frontend       | HTML5, CSS3, JavaScript           |
| Monitoring     | PSI-based drift detection         |
| Deployment     | Docker, Docker Compose            |
| Data           | LendingClub loan dataset          |

---

## 🚀 Key Features

- ✅ Real-time loan default probability prediction
- ✅ SHAP explainability — top 5 reasons shown for every decision
- ✅ Auto-calculated DTI, installment, revolving balance on the form
- ✅ Risk bands — LOW / MEDIUM / HIGH with color-coded results
- ✅ Business override rule — loan > 5x income = auto HIGH RISK
- ✅ Drift detection — PSI monitors incoming data every 10 predictions
- ✅ Auto-retraining — model updates itself on drift or every 100 predictions
- ✅ Governance logging — full audit trail for every decision
- ✅ Prediction history — searchable log of all past assessments
- ✅ Individual borrower reports — printable per-applicant risk reports
- ✅ Dashboard — confusion matrix, ROC-AUC, accuracy metrics live
- ✅ Fully containerized — runs with one Docker command

---

## 📊 Model Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | 68.68%  |
| ROC-AUC    | 74.06%  |
| Recall     | 64.77%  |
| F1-Score   | 45.25%  |
| Best Model | XGBoost |

> **Note:** Model is tuned for high recall over precision. In banking, missing a real defaulter is far more costly than wrongly rejecting a good borrower.

### Confusion Matrix

```
                     Predicted: Repay   Predicted: Default
Actual: Repay           16,719 (TN)          7,284 (FP)
Actual: Default          2,113 (FN)          3,884 (TP)
```

---

## 🧠 How the Prediction Works

```
User fills form
    ↓
Auto-calculates installment (EMI formula), DTI, revolving balance
    ↓
Backend engineers 9 additional features
    ↓
XGBoost outputs Probability of Default (0 to 1)
    ↓
Threshold applied (≤0.40 LOW, 0.40–0.60 MEDIUM, >0.60 HIGH)
    ↓
Business override checked (loan > 5× income = HIGH RISK)
    ↓
SHAP computes top 5 reasons
    ↓
Result shown to user with verdict + explanation
```

---

## 🎯 Risk Threshold Table

| Probability  | Risk Label  | Decision                        |
|--------------|-------------|---------------------------------|
| ≤ 0.40       | 🟢 LOW      | Loan likely to be repaid        |
| 0.40 – 0.60  | 🟡 MEDIUM   | Manual review recommended       |
| > 0.60       | 🔴 HIGH     | High probability of default     |
| Any          | 🔴 OVERRIDE | Loan amount > 5× annual income  |

---

## 🔧 Engineered Features

| Feature               | Formula                              | Purpose               |
|-----------------------|--------------------------------------|-----------------------|
| loan_to_income        | loan_amnt / annual_inc               | Affordability ratio   |
| installment_to_income | installment / annual_inc             | Monthly burden        |
| payment_capacity      | annual_inc − (installment × 12)      | Free cash flow        |
| credit_stress         | dti × loan_amnt                      | Combined leverage     |
| credit_utilization    | revol_bal / (revol_bal + bc_open)    | Credit stress         |
| high_dti_flag         | 1 if dti > 20 else 0                 | Binary risk flag      |
| low_fico_flag         | 1 if fico < 600 else 0               | Credit risk flag      |
| recent_inquiries_flag | 1 if inq_last_6mths > 3 else 0       | Credit-seeking flag   |
| risk_score            | int_rate × dti                       | Combined risk index   |

---

## 🏗️ System Architecture

```
Browser (User)
    │
    │  GET /  →  index.html (loan assessment form)
    │
    │  POST /predict
    ▼
Flask app.py
    ├── 1.  Validate input (loan_amnt, annual_inc, fico)
    ├── 2.  preprocess_input()       → 1-row DataFrame
    ├── 3.  create_features_live()   → engineered features
    ├── 4.  add_economic_features()  → macro context
    ├── 5.  reindex to MODEL_FEATURES → align columns
    ├── 6.  SHAP explain_single()    → top 5 feature drivers
    ├── 7.  MODEL.predict_proba()[0][1] → PD probability
    ├── 8.  Threshold logic          → verdict + risk label
    ├── 9.  Calculate LGD, EAD, Expected Loss
    ├── 10. Save to prediction_history.json
    ├── 11. log_decision()           → governance audit log
    ├── 12. Feedback loop check      → retrain if ≥ 100 entries
    ├── 13. Drift detection          → compare vs reference data
    └── 14. render result.html       → show risk to user
```

---

## 📁 Project Structure

```
credit-risk-prediction/
│
├── data/
│   ├── raw/
│   │   └── loan_dataset.csv           ← Place LendingClub CSV here
│   └── processed/
│       └── cleaned_data.csv           ← Auto-generated after preprocessing
│
├── models/
│   └── loan_default_model.pkl         ← Saved best model (XGBoost)
│
├── outputs/
│   ├── prediction_history.json        ← Live log of all predictions
│   └── fairness_report.txt            ← Auto-generated fairness metrics
│
├── reports/
│   └── <uuid>.txt                     ← Individual borrower risk reports
│
├── src/
│   ├── data_preprocessing.py          ← Clean + engineer + save processed CSV
│   ├── train_model.py                 ← Train LR + RF + XGBoost, save best
│   ├── evaluate_model.py              ← Evaluate saved model, update metrics
│   ├── shap_explainer.py              ← SHAP plots, fairness report
│   └── generate_performance_plots.py  ← ROC curve, confusion matrix plots
│
├── utils/
│   ├── config.py                      ← All paths, thresholds, XGBoost params
│   ├── preprocessor.py                ← Extract & save feature list from model
│   └── model_features.pkl             ← Auto-generated feature name list
│
├── webapp/
│   ├── app.py                         ← Main Flask application (all routes)
│   ├── retrain.py                     ← Triggers retraining as subprocess
│   ├── templates/
│   │   ├── base.html                  ← Shared navigation + layout
│   │   ├── index.html                 ← Loan assessment input form
│   │   ├── result.html                ← Prediction result page
│   │   ├── dashboard.html             ← Model metrics + confusion matrix
│   │   ├── history.html               ← Searchable prediction log
│   │   ├── reports.html               ← All borrower report cards
│   │   └── report_detail.html         ← Individual printable report
│   └── static/
│       ├── css/style.css              ← Application styling
│       └── js/script.js               ← Frontend JavaScript + auto-calculations
│
├── monitoring/
│   └── drift_detection.py             ← PSI-based feature drift monitor
│
├── explainability/                    ← SHAP output plots directory
├── logs/                              ← Governance audit logs
├── notebooks/                         ← Exploratory notebooks
│
├── feedback_loop.py                   ← Converts prediction history → training data
├── governance.py                      ← Logs every decision for compliance
├── model_metrics.json                 ← Auto-generated training metrics
├── Dockerfile                         ← Docker container definition
├── docker-compose.yml                 ← Docker Compose configuration
├── requirements.txt                   ← Python dependencies
└── README.md
```

---

## ⚙️ How to Run Locally

### Prerequisites

- Python 3.9+
- pip

### Step 1 — Clone the repo

```bash
git clone https://github.com/Ashwani4545/credit-risk-prediction.git
cd credit-risk-prediction
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Place your dataset

Put your LendingClub-format CSV at:

```
data/raw/loan_dataset.csv
```

Required target column: `loan_status` → `0` = Repay, `1` = Default

### Step 4 — Preprocess the data

```bash
python -m src.data_preprocessing
```

### Step 5 — Train the model

```bash
python -m src.train_model
```

This trains Logistic Regression, Random Forest, and XGBoost. Best model by simulated profit score is saved to `models/loan_default_model.pkl`.

### Step 6 — Save the feature list

```bash
python -m utils.preprocessor
```

### Step 7 — (Optional) Generate SHAP plots

```bash
python -m src.shap_explainer
```

### Step 8 — Run the web application

```bash
python webapp/app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 🐳 How to Run with Docker

```bash
docker-compose up --build
```

Open: **http://localhost:5000**

---

## 🌐 Web Application Pages

| Route          | Method | What it does                              |
|----------------|--------|-------------------------------------------|
| `/`            | GET    | Loan assessment input form                |
| `/predict`     | POST   | Runs model, shows result + SHAP reasons   |
| `/dashboard`   | GET    | Model metrics, confusion matrix, charts   |
| `/history`     | GET    | Searchable log of all past predictions    |
| `/reports`     | GET    | All individual borrower report cards      |
| `/reports/<id>`| GET    | Single printable borrower risk report     |
| `/api/metrics` | GET    | JSON endpoint — live model metrics        |
| `/api/history` | GET    | JSON history with search filter support   |
| `/health`      | GET    | Healthcheck endpoint                      |

---

## 🔄 Data Flow

```
data/raw/loan_dataset.csv
        ↓
src/data_preprocessing.py      → cleans, engineers features
        ↓
data/processed/cleaned_data.csv
        ↓
src/train_model.py              → trains 3 models, picks best
        ↓
models/loan_default_model.pkl   → saved XGBoost model
utils/model_features.pkl        → saved feature column list
model_metrics.json              → accuracy, AUC, confusion matrix
        ↓
webapp/app.py                   → Flask server loads model at startup
        ↓
User submits form → /predict route
        ↓
outputs/prediction_history.json → every prediction stored
        ↓
feedback_loop.py                → history becomes new training data
        ↓
monitoring/drift_detection.py   → detects distribution shift
        ↓
webapp/retrain.py               → triggers retraining if drift found
                                   or every 100 new predictions
```

---

## 📋 Key Design Decisions

- **Model selection by profit** — not just ROC-AUC. Correctly rejecting one defaulter saves the full loan amount.
- **SMOTE + scale_pos_weight** — dual-layer imbalance handling for better recall on the minority (default) class.
- **Feature alignment** — `model_features.pkl` saves the exact ordered feature list at training time. At inference, all inputs are `reindex()`-ed to this list to prevent column mismatch.
- **Column sanitization** — XGBoost-safe column names applied identically in both training and inference pipelines.
- **Governance logging** — every prediction logged with trace ID, timestamp, input features, and decision for compliance.
- **Drift detection** — PSI compares live predictions against first 10,000 training rows every 10th prediction.
- **Conservative threshold (0.40)** — in banking, missing a defaulter (False Negative) is more costly than wrongly rejecting a good borrower (False Positive).
- **Business override rule** — if `loan_amount > 5 × annual_income`, auto-flagged HIGH RISK regardless of model output.
- **Auto-calculations on form** — installment is computed using the EMI formula, DTI and revolving balance derived automatically, so users only need to enter core financial details.

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) explains every single prediction.

- Uses **TreeSHAP** — automatically selected for XGBoost, computationally exact
- Computes the marginal contribution of every feature to the model output
- Returns **top 5 features** driving each credit decision
- Positive SHAP value → pushes probability toward default
- Negative SHAP value → pushes probability toward repay

```
final_score  = sum of all XGBoost tree outputs (log-odds)
probability  = sigmoid(final_score) = 1 / (1 + e^(-final_score))
SHAP value   = feature's share of (final_score − baseline_score)
```

---

## 🔄 Drift Detection & Auto-Retraining

**Population Stability Index (PSI)** measures how much a feature's distribution has shifted between training data and new incoming predictions.

| PSI Value     | Status                              |
|---------------|-------------------------------------|
| < 0.10        | No Drift — model is stable          |
| 0.10 – 0.25   | Moderate Drift — monitor closely    |
| ≥ 0.25        | High Drift — retraining triggered   |

- PSI checked every 10th prediction
- On High Drift, `retrain.py` runs `src.train_model` as a subprocess
- After every 100 predictions, retraining is also triggered automatically
- Best model re-selected after every retraining run

---

## 📝 Governance & Audit Trail

Every prediction is appended to `logs/audit_log.json`. Each record contains:

- Timestamp and unique trace ID
- Raw input features
- Model prediction and probability
- Threshold used and risk level
- Override flag if triggered
- Top SHAP features and advice strings

The log is append-only and uses absolute file paths so it works regardless of launch directory.

---

## ⚠️ Known Limitations

- Economic features (`inflation_rate`, `unemployment_rate`) are hardcoded constants and carry no real signal — future version will use a live API
- Alternative credit features (`mobile_usage_score`, `digital_txn_count`, `utility_payment_score`) are set to 0 at inference since they are not collected from users
- Feedback loop uses the model's own predictions as training labels — in production, real repayment outcomes (known months later) would replace these
- Prediction history is stored as a flat JSON file — for production, replace with PostgreSQL or SQLite database

---

## 🔮 Future Scope

- Connect live macroeconomic data API (RBI, World Bank, FRED)
- Replace JSON history with PostgreSQL database
- REST API for integration with banking core systems
- Role-based access control — loan officer / admin / auditor
- Mobile-responsive UI improvements
- Advanced fairness auditing (demographic parity, equalized odds)
- MSME credit risk analytics extension
- Deploy as a Risk Intelligence SaaS platform for banks and NBFCs

---

## 🧪 Test Cases for Demo

**Test Case 1 — Should predict LOW RISK:**
```
Loan Amount   : 500,000
Annual Income : 2,000,000
Interest Rate : 7%
FICO Score    : 750
DTI           : 8    (auto-calculated)
Grade         : A
```

**Test Case 2 — Should predict HIGH RISK (also triggers override rule):**
```
Loan Amount    : 1,500,000
Annual Income  : 300,000
Interest Rate  : 22%
FICO Score     : 550
DTI            : 35   (auto-calculated)
Grade          : E
Delinquencies  : 3
Public Records : 1
```

> The second case triggers the business override rule (loan > 5× income), guaranteeing a HIGH RISK result regardless of model probability.

---

## 🎓 Academic Note

This project is developed for academic and research purposes as part of an AI/ML course project.

It demonstrates the application of explainable machine learning in the financial domain, covering the full pipeline from data preprocessing to real-time web deployment with governance and model monitoring.

**Built with:** Flask · XGBoost · SHAP · Python · Docker

---

## 👨‍💻 Author

**Ashwani**
GitHub: [@Ashwani4545](https://github.com/Ashwani4545)
