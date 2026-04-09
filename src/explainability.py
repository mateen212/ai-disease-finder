"""
Explainability Module

Provides feature-level explanations using SHAP values and rule traces
to make the system's decisions transparent and interpretable.
"""

import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainabilityModule:
    """
    Provides comprehensive explanations for clinical decisions.
    
    Combines:
    - SHAP values for ML model feature importance
    - Rule traces from the inference engine
    - Feature contribution analysis
    """
    
    def __init__(self, config_file: str = "config/model_config.yaml"):
        """
        Initialize the explainability module.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        explain_config = self.config.get('explainability', {})
        shap_config = explain_config.get('shap', {})
        
        self.max_features_display = shap_config.get('max_features_display', 10)
        self.num_shap_samples = shap_config.get('num_samples', 100)
        
        self.explainer = None
        self.background_data = None
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not available. Some features will be limited.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def initialize_shap_explainer(
        self,
        model: Any,
        background_data: pd.DataFrame,
        model_type: str = "tree"
    ):
        """
        Initialize SHAP explainer for a trained model.
        
        Args:
            model: Trained ML model (sklearn or similar)
            background_data: Sample data for SHAP background
            model_type: Type of model ("tree", "linear", "kernel")
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return
        
        # Sample background data if too large
        if len(background_data) > self.num_shap_samples:
            self.background_data = background_data.sample(
                n=self.num_shap_samples,
                random_state=42
            )
        else:
            self.background_data = background_data
        
        # Create explainer based on model type
        try:
            if model_type == "tree":
                # For tree-based models (Random Forest, XGBoost, etc.)
                self.explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                self.explainer = shap.LinearExplainer(model, self.background_data)
            else:
                # Kernel explainer (model-agnostic but slower)
                self.explainer = shap.KernelExplainer(
                    model.predict_proba,
                    self.background_data
                )
            
            logger.info(f"SHAP explainer initialized: {model_type}")
        
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
    
    def explain_prediction(
        self,
        patient_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP-based explanation for a prediction.
        
        Args:
            patient_data: Single patient data as DataFrame (1 row)
            feature_names: Optional list of feature names
        
        Returns:
            Dictionary with SHAP values and explanation
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {
                'shap_values': None,
                'feature_contributions': {},
                'explanation': "SHAP explainer not available"
            }
        
        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(patient_data)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # For multi-class, use the class with highest prediction
                # or average across classes
                shap_values_array = np.array(shap_values)
                # Take mean absolute contribution across classes
                shap_values_summary = np.mean(np.abs(shap_values_array), axis=0)
            else:
                shap_values_summary = shap_values
            
            # Get feature names
            if feature_names is None:
                feature_names = patient_data.columns.tolist()
            
            # Create feature contribution dictionary
            contributions = {}
            for i, feature in enumerate(feature_names):
                if len(shap_values_summary.shape) == 2:
                    value = float(shap_values_summary[0, i])
                else:
                    value = float(shap_values_summary[i])
                contributions[feature] = value
            
            # Sort by absolute contribution
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top features
            top_features = sorted_contributions[:self.max_features_display]
            
            # Generate explanation text
            explanation = self._generate_shap_explanation(
                top_features,
                patient_data.iloc[0].to_dict()
            )
            
            return {
                'shap_values': shap_values,
                'feature_contributions': contributions,
                'top_features': dict(top_features),
                'explanation': explanation
            }
        
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return {
                'shap_values': None,
                'feature_contributions': {},
                'explanation': f"Error: {str(e)}"
            }
    
    def _generate_shap_explanation(
        self,
        top_features: List[Tuple[str, float]],
        patient_values: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation from SHAP values.
        
        Args:
            top_features: List of (feature_name, shap_value) tuples
            patient_values: Dictionary of patient's feature values
        
        Returns:
            Explanation string
        """
        explanation = ["Key factors influencing this diagnosis:\n"]
        
        for i, (feature, shap_value) in enumerate(top_features, 1):
            # Get patient's value for this feature
            patient_val = patient_values.get(feature, "N/A")
            
            # Determine direction of contribution
            if shap_value > 0:
                direction = "increases"
                symbol = "↑"
            else:
                direction = "decreases"
                symbol = "↓"
            
            # Clean feature name for display
            display_name = feature.replace('_', ' ').title()
            
            explanation.append(
                f"{i}. {display_name} (value: {patient_val:.2f if isinstance(patient_val, (int, float)) else patient_val}): "
                f"{symbol} {direction} probability (contribution: {abs(shap_value):.3f})"
            )
        
        return "\n".join(explanation)
    
    def combine_explanations(
        self,
        shap_explanation: Dict[str, Any],
        rule_trace: List[Dict[str, Any]],
        fusion_result: Dict[str, Any]
    ) -> str:
        """
        Combine SHAP explanations with rule traces for comprehensive explanation.
        
        Args:
            shap_explanation: SHAP explanation dictionary
            rule_trace: List of fired rules from rule engine
            fusion_result: Fusion results dictionary
        
        Returns:
            Combined explanation string
        """
        explanation = []
        
        explanation.append("="*60)
        explanation.append("CLINICAL DECISION SUPPORT SYSTEM - EXPLANATION REPORT")
        explanation.append("="*60)
        
        # Primary diagnosis
        primary_disease, confidence = fusion_result.get('primary_diagnosis', ('unknown', 0))
        explanation.append(f"\nPrimary Diagnosis: {primary_disease.upper()}")
        explanation.append(f"Confidence: {confidence:.1%}")
        explanation.append(f"Risk Level: {fusion_result.get('overall_risk', 'unknown').upper()}")
        
        # Rule-based reasoning
        explanation.append("\n" + "-"*60)
        explanation.append("SYMBOLIC REASONING (Rule-Based)")
        explanation.append("-"*60)
        
        if rule_trace:
            explanation.append(f"\n{len(rule_trace)} diagnostic rule(s) activated:\n")
            for rule in rule_trace:
                explanation.append(f"✓ {rule['name']}")
                explanation.append(f"  Description: {rule['description']}")
                explanation.append(f"  Disease: {rule['disease']}")
                explanation.append(f"  Confidence: {rule['confidence']}")
                explanation.append(f"  Probability Boost: +{rule['boost']:.2f}\n")
        else:
            explanation.append("\nNo specific diagnostic rules were triggered.")
        
        # Machine Learning explanation
        explanation.append("-"*60)
        explanation.append("NEURAL NETWORK REASONING (ML Model)")
        explanation.append("-"*60)
        
        if shap_explanation and shap_explanation.get('explanation'):
            explanation.append("\n" + shap_explanation['explanation'])
        else:
            explanation.append("\nML explanation not available")
        
        # Component contributions
        explanation.append("\n" + "-"*60)
        explanation.append("NEURO-SYMBOLIC FUSION")
        explanation.append("-"*60)
        
        contributions = fusion_result.get('component_contributions', {})
        explanation.append("\nComponent-wise confidence scores:")
        
        for component, scores in contributions.items():
            if scores and any(s > 0 for s in scores.values()):
                explanation.append(f"\n{component.replace('_', ' ').title()}:")
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for disease, score in sorted_scores[:3]:
                    if score > 0:
                        explanation.append(f"  • {disease}: {score:.1%}")
        
        # Final integrated score
        explanation.append("\n" + "-"*60)
        explanation.append("INTEGRATED DISEASE PROBABILITIES")
        explanation.append("-"*60 + "\n")
        
        disease_scores = fusion_result.get('disease_scores', {})
        if disease_scores:
            sorted_scores = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
            for disease, score in sorted_scores:
                bar_length = int(score * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                explanation.append(f"{disease:15s} {bar} {score:.1%}")
        
        explanation.append("\n" + "="*60)
        
        return "\n".join(explanation)
    
    def create_feature_importance_plot(
        self,
        feature_contributions: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Create a visualization of feature importance.
        
        Args:
            feature_contributions: Dictionary of feature SHAP values
            save_path: Optional path to save plot
        """
        # Sort by absolute value
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:self.max_features_display]
        
        features, values = zip(*sorted_features)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if v < 0 else 'green' for v in values]
        ax.barh(range(len(features)), values, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Importance for Diagnosis')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_patient_report(
        self,
        patient_data: Dict[str, Any],
        fusion_result: Dict[str, Any],
        shap_explanation: Dict[str, Any],
        rule_trace: List[Dict[str, Any]],
        recommendations: List[str]
    ) -> str:
        """
        Generate comprehensive patient report.
        
        Args:
            patient_data: Patient's clinical data
            fusion_result: Fusion results
            shap_explanation: SHAP explanation
            rule_trace: Rule traces
            recommendations: Clinical recommendations
        
        Returns:
            Formatted patient report
        """
        report = []
        
        report.append("╔" + "═"*78 + "╗")
        report.append("║" + " "*20 + "CLINICAL DIAGNOSTIC REPORT" + " "*32 + "║")
        report.append("╚" + "═"*78 + "╝")
        
        # Patient summary
        report.append("\n📋 PATIENT SUMMARY")
        report.append("─"*80)
        demographics = patient_data.get('demographics', {})
        report.append(f"Age: {demographics.get('age', 'N/A')}")
        report.append(f"Gender: {demographics.get('gender', 'N/A')}")
        
        # Key symptoms
        report.append("\n🔍 PRESENTING SYMPTOMS")
        report.append("─"*80)
        symptoms = patient_data.get('symptoms', {})
        positive_symptoms = [k.replace('_', ' ').title() for k, v in symptoms.items() if v]
        if positive_symptoms:
            for symptom in positive_symptoms:
                report.append(f"  ✓ {symptom}")
        else:
            report.append("  No significant symptoms reported")
        
        # Vital signs
        report.append("\n🩺 VITAL SIGNS")
        report.append("─"*80)
        vitals = patient_data.get('vitals', {})
        for vital, value in vitals.items():
            display_name = vital.replace('_', ' ').title()
            report.append(f"  {display_name}: {value}")
        
        # Lab results
        labs = patient_data.get('labs', {})
        if labs:
            report.append("\n🧪 LABORATORY RESULTS")
            report.append("─"*80)
            for lab, value in labs.items():
                display_name = lab.replace('_', ' ').title()
                report.append(f"  {display_name}: {value}")
        
        # Diagnosis
        report.append("\n🎯 DIAGNOSIS")
        report.append("─"*80)
        primary_disease, confidence = fusion_result.get('primary_diagnosis', ('unknown', 0))
        risk = fusion_result.get('overall_risk', 'unknown')
        
        report.append(f"Primary: {primary_disease.upper()}")
        report.append(f"Confidence: {confidence:.1%}")
        report.append(f"Risk Level: {risk.upper()}")
        
        # Detailed explanation
        report.append("\n📊 EXPLANATION")
        report.append("─"*80)
        combined_explanation = self.combine_explanations(
            shap_explanation,
            rule_trace,
            fusion_result
        )
        report.append(combined_explanation)
        
        # Recommendations
        report.append("\n💊 RECOMMENDATIONS")
        report.append("─"*80)
        for rec in recommendations:
            report.append(rec)
        
        report.append("\n" + "═"*80)
        report.append("⚠️  DISCLAIMER: This report is generated by an AI system for")
        report.append("   educational purposes. It should NOT replace professional")
        report.append("   medical judgment. Always consult healthcare providers.")
        report.append("═"*80)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    explainer = ExplainabilityModule()
    
    # Mock data
    patient_data = {
        'symptoms': {'fever': True, 'headache': True, 'rash': True},
        'vitals': {'temperature': 39.5, 'oxygen_saturation': 96},
        'labs': {'platelet_count': 95000, 'wbc_count': 3200},
        'demographics': {'age': 35, 'gender': 'M'}
    }
    
    fusion_result = {
        'primary_diagnosis': ('dengue', 0.82),
        'overall_risk': 'high',
        'disease_scores': {'dengue': 0.82, 'covid19': 0.12, 'pneumonia': 0.06},
        'component_contributions': {
            'rule_based': {'dengue': 0.7},
            'random_forest': {'dengue': 0.75}
        }
    }
    
    rule_trace = [
        {
            'name': 'Dengue_Classic',
            'description': 'Classic dengue presentation',
            'disease': 'dengue',
            'confidence': 'high',
            'boost': 0.4
        }
    ]
    
    recommendations = [
        "Seek medical evaluation within 24 hours",
        "Monitor platelet count daily",
        "Ensure adequate hydration"
    ]
    
    shap_explanation = {
        'explanation': 'Fever, low platelets, and rash are key contributors'
    }
    
    report = explainer.generate_patient_report(
        patient_data,
        fusion_result,
        shap_explanation,
        rule_trace,
        recommendations
    )
    
    print(report)
