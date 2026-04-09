"""
Hybrid Neuro-Symbolic Clinical Decision Support System

This package implements a multi-disease diagnosis system combining
rule-based reasoning with machine learning.
"""

__version__ = "1.0.0"
__author__ = "Clinical AI Team"

from .rule_engine import RuleEngine, Rule
from .ml_models import RandomForestDiagnostic, SkinLesionCNN
from .fusion import NeuroSymbolicFusion
from .explainability import ExplainabilityModule

__all__ = [
    "RuleEngine",
    "Rule",
    "RandomForestDiagnostic",
    "SkinLesionCNN",
    "NeuroSymbolicFusion",
    "ExplainabilityModule",
]
