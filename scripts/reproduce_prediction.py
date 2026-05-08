import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import get_model_frame_debug, _load_model
import json

inp = {
    'income':65000,'loan_amount':12300,'credit_score':685,'int_rate':11.53,
    'dti':18.37,'revol_util':52,'emp_length':2,'open_acc':11,'delinq_2yrs':0,
    'term':'36 months','purpose':'debt_consolidation','home_ownership':'mortgage'
}

frame, raw = get_model_frame_debug(inp)
model = _load_model()
import pandas as pd
prob = float(model.predict_proba(pd.DataFrame([frame]))[0][1])

out = {
    'raw_input': raw,
    'sample_features': {k: frame[k] for k in list(frame)[:60]},
    'probability': prob,
}
print(json.dumps(out, indent=2))

# also call the public predict() to show full result payload (including overrides, SHAP)
from src.predict import predict
result = predict({
    'income':65000,'loan_amount':12300,'credit_score':685,'int_rate':11.53,
    'dti':18.37,'revol_util':52,'emp_length':2,'open_acc':11,'delinq_2yrs':0,
    'term':'36 months','purpose':'debt_consolidation','home_ownership':'mortgage'
})
print(json.dumps({'predict_result': result}, indent=2))
