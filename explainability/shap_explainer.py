# explainability/shap_explainer.py
"""
REMOVED — This file was a duplicate of src/shap_explainer.py with a
conflicting implementation (different class methods, different fairness
functions, different SHAP initialisation).

The canonical implementation is at:
    src/shap_explainer.py

Import from there:
    from src.shap_explainer import LoanModelExplainer, get_local_shap

This file exists only to prevent ImportError for any legacy references.
It re-exports everything from the canonical location.
"""

from src.shap_explainer import LoanModelExplainer, get_local_shap  # noqa: F401

__all__ = ["LoanModelExplainer", "get_local_shap"]
