import pandas as pd
from webapp.app import preprocess_input, _NUMERIC_FIELDS, _CATEGORICAL_FIELDS
import json

form_data = {
    'borrower_name': 'John', 'loan_amnt': '100000', 'int_rate': '15',
    'installment': '3500', 'annual_inc': '50000', 'dti': '25',
    'fico_range_low': '580', 'fico_range_high': '600',
    'open_acc': '5', 'revol_bal': '8000', 'revol_util': '60',
    'total_acc': '12', 'delinq_2yrs': '1', 'inq_last_6mths': '2',
    'pub_rec': '0', 'bc_open_to_buy': '2000', 'bc_util': '70',
    'grade': 'D', 'purpose': 'debt_consolidation', 'term': ' 60 months',
}

df = preprocess_input(form_data)
set_cols = [c for c in df.columns if df[c].iloc[0] != 0.0]
print("Set columns:")
for c in set_cols:
    print(c, df[c].iloc[0])

print("\nTerm/Purpose columns in model features:")
for c in df.columns:
    if 'term' in c.lower() or 'purpose' in c.lower() or 'grade' in c.lower():
        print(c)
