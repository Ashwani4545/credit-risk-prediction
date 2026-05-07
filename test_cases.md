# Loan Default Prediction — Test Cases

## Test Case 1: Low Risk / Approve
Expected: Probability ~25–35%, Risk = "LOW RISK", Verdict = "Repay/Approve"

```
loan_amnt: 100000
annual_inc: 600000
int_rate: 8.5
installment: 2500
dti: 12
fico_range_low: 730
fico_range_high: 750
term: 36
grade: A
home_ownership: MORTGAGE
verification_status: Verified
purpose: credit_card
revol_bal: 5000
bc_open_to_buy: 20000
open_acc: 10
total_acc: 40
inq_last_6mths: 0
```

---

## Test Case 2: Medium Risk / Review (REVISED)
Expected: Probability ~50–58%, Risk = "MEDIUM RISK", Verdict = "Review" (borderline, at threshold)

```
loan_amnt: 120000
annual_inc: 350000
int_rate: 10.5
installment: 3200
dti: 18
fico_range_low: 680
fico_range_high: 700
term: 36
grade: B
home_ownership: MORTGAGE
verification_status: Verified
purpose: debt_consolidation
revol_bal: 8000
bc_open_to_buy: 12000
open_acc: 8
total_acc: 30
inq_last_6mths: 1
```

---

## Test Case 3: High Risk / Default/Reject
Expected: Probability ~70–75%, Risk = "HIGH RISK", Verdict = "Default/Reject"

```
loan_amnt: 80000
annual_inc: 150000
int_rate: 16.0
installment: 2800
dti: 32
fico_range_low: 600
fico_range_high: 620
term: 60
grade: D
home_ownership: RENT
verification_status: Source Verified
purpose: small_business
revol_bal: 25000
bc_open_to_buy: 2000
open_acc: 5
total_acc: 15
inq_last_6mths: 4
```

---

## Test Case 4: Edge Case — Loan >> Income (Manual-Review Flag)
Expected: Probability ~40–50%, Risk = "MEDIUM RISK", Verdict = "Repay" + Manual Review Warning

```
loan_amnt: 500000
annual_inc: 60000
int_rate: 9.5
installment: 12000
dti: 85
fico_range_low: 700
fico_range_high: 720
term: 60
grade: B
home_ownership: OWN
verification_status: Verified
purpose: home_improvement
revol_bal: 5000
bc_open_to_buy: 10000
open_acc: 7
total_acc: 22
inq_last_6mths: 1
```

---

## Test Case 5: Credit-Invisible (No FICO)
Expected: Probability ~45–55%, Risk = "MEDIUM RISK", Verdict = "Review" (alternative data used)

```
loan_amnt: 25000
annual_inc: 50000
int_rate: 13.0
installment: 600
dti: 16
fico_range_low: 0
fico_range_high: 0
term: 36
grade: E
home_ownership: RENT
verification_status: Not Verified
purpose: car
revol_bal: 3000
bc_open_to_buy: 1000
open_acc: 3
total_acc: 8
inq_last_6mths: 2
```

---

## How to Use

**Via Web Form:** Fill http://127.0.0.1:5000/ with one test case and click "Analyze".

**Via cURL (POST to `/predict`):**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "loan_amnt=100000&annual_inc=600000&int_rate=8.5&installment=2500&dti=12&fico_range_low=730&fico_range_high=750&term=36&grade=A&home_ownership=MORTGAGE&verification_status=Verified&purpose=credit_card&revol_bal=5000&bc_open_to_buy=20000&open_acc=10&total_acc=40&inq_last_6mths=0&borrower_name=Test1"
```

## RISK_LEVELS Mapping

| Probability Range | Risk Label | Decision Threshold (0.56) |
|---|---|---|
| < 0.40 | LOW RISK | Repay/Approve |
| 0.40–0.59 | MEDIUM RISK | Repay/Approve (if < 0.56) or Review (if ≥ 0.56) |
| 0.60–0.79 | HIGH RISK | Review or Default |
| ≥ 0.80 | VERY HIGH RISK | Default/Reject |

---

## Expected Outcomes

- **Test 1:** ✅ Approve — Good credit, low DTI, high income
- **Test 2:** 📋 Review — Moderate risk, near threshold
- **Test 3:** ❌ Reject — Poor credit, high DTI, low income
- **Test 4:** ⚠️ Repay + Warning — Loan too large relative to income
- **Test 5:** 🤔 Review — No FICO, alternative data fallback
