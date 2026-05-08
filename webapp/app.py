# webapp/app.py
# Run: python webapp/app.py

from flask import Flask, render_template, request, jsonify
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict as get_prediction
from src.shap_explainer import get_local_shap
from monitoring.drift_detection import run_monitoring_summary

app = Flask(__name__)


# ─────────────────────────────────────
# HOME
# ─────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


# ─────────────────────────────────────
# PREDICT (form POST → result page)
# ─────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        result = get_prediction(data)

        prediction   = result["prediction"]
        probability  = result["default_probability"]
        shap_values  = result.get("shap_values", {})
        advice       = result.get("advice", [])

        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(probability * 100, 1),
            shap_values=shap_values,
            advice=advice,
            form_data=data,
        )

    except Exception as e:
        return render_template('error.html', error=str(e)), 500


# ─────────────────────────────────────
# PREDICT (AJAX JSON endpoint)
# ─────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        result = get_prediction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    try:
        drift_summary = run_monitoring_summary()
    except Exception:
        drift_summary = {"status": "unavailable", "features": []}
    return render_template('dashboard.html', drift=drift_summary)


# ─────────────────────────────────────
# ABOUT
# ─────────────────────────────────────
@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
