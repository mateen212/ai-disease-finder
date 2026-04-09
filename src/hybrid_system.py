"""
Hybrid Diagnostic System

Integrates Rule Engine, Random Forest, CNN, and Fusion components
into a unified diagnostic system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import torch

from src.rule_engine import RuleEngine
from src.ml_models import RandomForestDiagnostic, SkinLesionCNN
from src.fusion import NeuroSymbolicFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridDiagnosticSystem:
    """
    Main diagnostic system that integrates all components.
    
    Combines:
    - Rule-based reasoning (expert rules)
    - Random Forest (ML classifier on clinical data)
    - CNN (deep learning for skin lesions)
    - Neuro-symbolic fusion (weighted combination)
    """
    
    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing Hybrid Diagnostic System...")
        
        # Initialize components
        self.rule_engine = RuleEngine()
        self.random_forest = RandomForestDiagnostic()
        self.cnn = SkinLesionCNN()
        self.fusion = NeuroSymbolicFusion()
        
        # Load trained models
        self._load_models()
        
        # Define disease mappings
        self.clinical_diseases = ['dengue', 'covid19', 'pneumonia', 'malaria', 'influenza']
        self.skin_diseases = [
            'melanoma', 'melanocytic_nevus', 'basal_cell_carcinoma',
            'actinic_keratosis', 'benign_keratosis', 'dermatofibroma',
            'vascular_lesion'
        ]
        
        logger.info("✓ Hybrid System initialized")
    
    def _load_models(self):
        """Load trained ML models"""
        # Load Random Forest
        rf_path = Path("models/random_forest_clinical.pkl")
        if rf_path.exists():
            try:
                self.random_forest.load(str(rf_path))
                logger.info(f"✓ Loaded Random Forest from {rf_path}")
            except Exception as e:
                logger.warning(f"Could not load Random Forest: {e}")
        
        # Load CNN
        cnn_path = Path("models/cnn_skin_lesion.pth")
        if cnn_path.exists():
            try:
                self.cnn.load(str(cnn_path))
                logger.info(f"✓ Loaded CNN from {cnn_path}")
            except Exception as e:
                logger.warning(f"Could not load CNN: {e}")
    
    def diagnose(
        self,
        patient_data: Dict[str, Any],
        image: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid diagnosis using all available components.
        
        Args:
            patient_data: Dictionary with patient information:
                - symptoms: dict of symptom flags
                - vitals: dict of vital signs
                - labs: dict of laboratory values
                - demographics: dict with age, sex, etc.
            image: Optional skin lesion image tensor for CNN
        
        Returns:
            Dictionary with diagnosis results:
                - diagnosis: Primary diagnosis
                - confidence: Overall confidence score
                - probabilities: Disease probabilities
                - component_predictions: Individual component results
                - explanation: Reasoning explanation
        """
        # Get predictions from each component
        rule_result = self._get_rule_prediction(patient_data)
        rf_result = self._get_rf_prediction(patient_data)
        cnn_result = self._get_cnn_prediction(image) if image is not None else None
        
        # Prepare scores for fusion
        rule_scores = rule_result['disease_scores']
        rf_scores = rf_result['probabilities']
        cnn_scores = cnn_result['probabilities'] if cnn_result else {}
        
        # Fuse predictions
        fusion_result = self.fusion.fuse_predictions(
            rule_scores=rule_scores,
            rf_scores=rf_scores,
            cnn_scores=cnn_scores,
            rule_metadata={'rule_count': rule_result['rule_count']}
        )
        
        # Get top prediction from fused scores
        probabilities = fusion_result['disease_scores']  # Fixed: was 'combined_scores'
        if not probabilities:
            probabilities = fusion_result['all_scores']  # Fallback to unfiltered scores
        
        top_disease = max(probabilities.items(), key=lambda x: x[1])
        
        # Build result
        result = {
            'diagnosis': top_disease[0],
            'confidence': top_disease[1],
            'probabilities': probabilities,
            'component_predictions': {
                'Rule Engine': {
                    'top_disease': rule_result.get('top_disease', 'none'),
                    'top_score': rule_result.get('top_score', 0.0),
                    'fired_rules': rule_result.get('rule_count', 0)
                },
                'Random Forest': {
                    'top_disease': rf_result.get('top_disease', 'none'),
                    'top_score': rf_result.get('top_score', 0.0)
                }
            },
            'explanation': self._generate_explanation(
                rule_result, rf_result, cnn_result, top_disease[0]
            )
        }
        
        # Add CNN results if available
        if cnn_result:
            result['component_predictions']['CNN'] = {
                'top_disease': cnn_result.get('top_disease', 'none'),
                'top_score': cnn_result.get('top_score', 0.0)
            }
        
        return result
    
    def _get_rule_prediction(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from rule engine"""
        try:
            result = self.rule_engine.evaluate_rules(patient_data)
            
            # Find top disease
            disease_scores = result.get('disease_scores', {})
            if disease_scores:
                top = max(disease_scores.items(), key=lambda x: x[1])
                result['top_disease'] = top[0]
                result['top_score'] = top[1]
            else:
                result['top_disease'] = 'none'
                result['top_score'] = 0.0
            
            return result
        except Exception as e:
            logger.error(f"Rule engine error: {e}")
            return {
                'disease_scores': {},
                'rule_count': 0,
                'top_disease': 'none',
                'top_score': 0.0
            }
    
    def _get_rf_prediction(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from Random Forest"""
        try:
            if not self.random_forest.is_trained:
                return {
                    'probabilities': {},
                    'top_disease': 'none',
                    'top_score': 0.0
                }
            
            # Convert patient data to DataFrame
            features = self._extract_features(patient_data)
            
            # Get predictions
            probabilities = self.random_forest.predict_proba(features)
            
            # Find top disease
            top = max(probabilities.items(), key=lambda x: x[1])
            
            return {
                'probabilities': probabilities,
                'top_disease': top[0],
                'top_score': top[1]
            }
        except Exception as e:
            logger.error(f"Random Forest error: {e}")
            return {
                'probabilities': {},
                'top_disease': 'none',
                'top_score': 0.0
            }
    
    def _get_cnn_prediction(self, image: torch.Tensor) -> Dict[str, Any]:
        """Get prediction from CNN"""
        try:
            if not self.cnn.is_trained:
                return {
                    'probabilities': {},
                    'top_disease': 'none',
                    'top_score': 0.0
                }
            
            # Get predictions
            probabilities = self.cnn.predict_proba(image)
            
            # Find top disease
            top = max(probabilities.items(), key=lambda x: x[1])
            
            return {
                'probabilities': probabilities,
                'top_disease': top[0],
                'top_score': top[1]
            }
        except Exception as e:
            logger.error(f"CNN error: {e}")
            return {
                'probabilities': {},
                'top_disease': 'none',
                'top_score': 0.0
            }
    
    def _extract_features(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract features from patient data for Random Forest.
        
        Converts nested dictionary to flat feature DataFrame.
        """
        features = {}
        
        # Symptoms (boolean features)
        symptoms = patient_data.get('symptoms', {})
        for symptom, value in symptoms.items():
            features[symptom] = 1 if value else 0
        
        # Vitals (numeric features)
        vitals = patient_data.get('vitals', {})
        for vital, value in vitals.items():
            features[vital] = float(value) if value is not None else 0.0
        
        # Labs (numeric features)
        labs = patient_data.get('labs', {})
        for lab, value in labs.items():
            features[lab] = float(value) if value is not None else 0.0
        
        # Demographics
        demographics = patient_data.get('demographics', {})
        features['age'] = float(demographics.get('age', 30))
        
        # Convert sex to numeric (Male=1, Female=0, Other=2)
        sex = demographics.get('sex', 'Male')
        features['sex'] = 1 if sex == 'Male' else (0 if sex == 'Female' else 2)
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all expected features are present (fill missing with 0)
        if self.random_forest.feature_names:
            for feature in self.random_forest.feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Reorder to match training
            df = df[self.random_forest.feature_names]
        
        return df
    
    def _generate_explanation(
        self,
        rule_result: Dict[str, Any],
        rf_result: Dict[str, Any],
        cnn_result: Optional[Dict[str, Any]],
        final_diagnosis: str
    ) -> str:
        """Generate human-readable explanation of the diagnosis"""
        
        explanation = f"**Diagnosis: {final_diagnosis.upper()}**\n\n"
        
        # Rule engine contribution
        rule_count = rule_result.get('rule_count', 0)
        rule_disease = rule_result.get('top_disease', 'none')
        explanation += f"• **Rule Engine**: {rule_count} rules fired"
        if rule_disease != 'none':
            explanation += f", suggesting {rule_disease}"
        explanation += "\n"
        
        # Random Forest contribution
        rf_disease = rf_result.get('top_disease', 'none')
        rf_score = rf_result.get('top_score', 0.0)
        explanation += f"• **Random Forest**: "
        if rf_disease != 'none':
            explanation += f"Predicts {rf_disease} ({rf_score*100:.1f}% confidence)"
        else:
            explanation += "No prediction"
        explanation += "\n"
        
        # CNN contribution (if available)
        if cnn_result:
            cnn_disease = cnn_result.get('top_disease', 'none')
            cnn_score = cnn_result.get('top_score', 0.0)
            explanation += f"• **CNN**: "
            if cnn_disease != 'none':
                explanation += f"Identifies {cnn_disease} ({cnn_score*100:.1f}% confidence)"
            else:
                explanation += "No prediction"
            explanation += "\n"
        
        explanation += f"\n**Combined prediction** using weighted fusion favors {final_diagnosis}."
        
        return explanation


# Example usage
if __name__ == "__main__":
    system = HybridDiagnosticSystem()
    
    # Example patient data
    patient = {
        'symptoms': {
            'fever': True,
            'cough': True,
            'fatigue': True,
            'shortness_of_breath': True,
            'loss_of_taste': True
        },
        'vitals': {
            'temperature': 38.5,
            'heart_rate': 95,
            'respiratory_rate': 22,
            'oxygen_saturation': 94,
            'blood_pressure_systolic': 125,
            'blood_pressure_diastolic': 82
        },
        'labs': {
            'wbc_count': 6500,
            'platelet_count': 220000,
            'hemoglobin': 14.2,
            'crp': 45,
            'ferritin': 280
        },
        'demographics': {
            'age': 45,
            'sex': 'Male'
        }
    }
    
    result = system.diagnose(patient)
    
    print("\n=== Diagnosis Result ===")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nExplanation:\n{result['explanation']}")
