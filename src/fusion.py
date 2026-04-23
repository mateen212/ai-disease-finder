"""
Neuro-Symbolic Fusion Module

Combines rule-based reasoning with neural network predictions to
produce final disease diagnoses and risk assessments.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuroSymbolicFusion:
    """
    Fuses symbolic rule-based outputs with neural network predictions.
    
    Implements multiple fusion strategies:
    - Weighted averaging
    - Maximum confidence
    - Stacking/ensemble
    """
    
    def __init__(self, config_file: str = "config/model_config.yaml"):
        """
        Initialize the fusion module.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        fusion_config = self.config.get('fusion', {})
        
        self.weights = fusion_config.get('weights', {
            'rule_based': 0.3,
            'random_forest': 0.5,
            'cnn': 0.2
        })
        
        self.strategy = fusion_config.get('strategy', 'weighted_average')
        self.min_confidence = fusion_config.get('min_confidence', 0.1)
        
        # Dynamic boosting configuration
        boosting_config = fusion_config.get('dynamic_boosting', {})
        self.dynamic_boosting_enabled = boosting_config.get('enabled', False)
        self.high_confidence_threshold = boosting_config.get('high_confidence_threshold', 0.7)
        self.high_confidence_rule_weight = boosting_config.get('high_confidence_rule_weight', 0.75)
        
        # Disease categories
        diseases_config = self.config.get('diseases', {})
        self.clinical_diseases = diseases_config.get('clinical', ['dengue', 'covid19', 'pneumonia'])
        self.skin_diseases = diseases_config.get('skin', ['melanoma', 'eczema', 'psoriasis', 'acne'])
        
        logger.info(f"Fusion strategy: {self.strategy}")
        logger.info(f"Weights: {self.weights}")
        if self.dynamic_boosting_enabled:
            logger.info(f"Dynamic boosting enabled: threshold={self.high_confidence_threshold}, boosted_weight={self.high_confidence_rule_weight}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def fuse_predictions(
        self,
        rule_scores: Dict[str, float],
        rf_scores: Dict[str, float],
        cnn_scores: Optional[Dict[str, float]] = None,
        rule_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fuse predictions from all components.
        
        Args:
            rule_scores: Disease scores from rule engine
            rf_scores: Disease probabilities from Random Forest
            cnn_scores: Disease probabilities from CNN (for skin conditions)
            rule_metadata: Additional rule information (risk levels, fired rules)
        
        Returns:
            Dictionary with fused predictions and metadata
        """
        # Initialize combined scores
        all_diseases = set(self.clinical_diseases + self.skin_diseases)
        combined_scores = {disease: 0.0 for disease in all_diseases}
        
        # Track which components contributed
        component_contributions = {
            'rule_based': {},
            'random_forest': {},
            'cnn': {}
        }
        
        # Determine if we should use dynamic boosting
        # When high-confidence rules fire, increase rule weight
        max_rule_score = max(rule_scores.values()) if rule_scores else 0.0
        use_boosted_weights = (
            self.dynamic_boosting_enabled and 
            max_rule_score >= self.high_confidence_threshold
        )
        
        if use_boosted_weights:
            logger.info(f"🔥 Dynamic boosting activated: Rule score {max_rule_score:.2f} >= threshold {self.high_confidence_threshold}")
            current_rule_weight = self.high_confidence_rule_weight
            current_rf_weight = 1.0 - current_rule_weight
        else:
            current_rule_weight = self.weights['rule_based']
            current_rf_weight = self.weights['random_forest']
        
        # Process clinical diseases (rule + RF)
        for disease in self.clinical_diseases:
            rule_score = rule_scores.get(disease, 0.0)
            rf_score = rf_scores.get(disease, 0.0)
            
            if self.strategy == 'weighted_average':
                # Weighted average of rule and RF scores
                combined = (
                    current_rule_weight * rule_score +
                    current_rf_weight * rf_score
                )
                # Normalize by sum of weights (since CNN doesn't apply)
                weight_sum = current_rule_weight + current_rf_weight
                combined_scores[disease] = combined / weight_sum
            
            elif self.strategy == 'max':
                # Take maximum confidence
                combined_scores[disease] = max(rule_score, rf_score)
            
            else:  # Default to weighted average
                combined = (
                    current_rule_weight * rule_score +
                    current_rf_weight * rf_score
                )
                weight_sum = current_rule_weight + current_rf_weight
                combined_scores[disease] = combined / weight_sum
            
            # Track contributions
            component_contributions['rule_based'][disease] = rule_score
            component_contributions['random_forest'][disease] = rf_score
        
        # Process skin diseases (RF + CNN if available)
        if cnn_scores:
            for disease in self.skin_diseases:
                rf_score = rf_scores.get(disease, 0.0)
                cnn_score = cnn_scores.get(disease, 0.0)
                
                if self.strategy == 'weighted_average':
                    combined = (
                        self.weights['random_forest'] * rf_score +
                        self.weights['cnn'] * cnn_score
                    )
                    weight_sum = self.weights['random_forest'] + self.weights['cnn']
                    combined_scores[disease] = combined / weight_sum
                
                elif self.strategy == 'max':
                    combined_scores[disease] = max(rf_score, cnn_score)
                
                else:
                    combined = (
                        self.weights['random_forest'] * rf_score +
                        self.weights['cnn'] * cnn_score
                    )
                    weight_sum = self.weights['random_forest'] + self.weights['cnn']
                    combined_scores[disease] = combined / weight_sum
                
                component_contributions['random_forest'][disease] = rf_score
                component_contributions['cnn'][disease] = cnn_score
        
        # Filter out low-confidence predictions
        filtered_scores = {
            disease: score 
            for disease, score in combined_scores.items()
            if score >= self.min_confidence
        }
        
        # Extract risk assessments from rule metadata
        risk_assessments = {}
        if rule_metadata and 'risk_assessments' in rule_metadata:
            risk_assessments = rule_metadata['risk_assessments']
        
        # Determine overall risk level (highest among predicted diseases)
        overall_risk = self._determine_overall_risk(filtered_scores, risk_assessments)
        
        # Create final result
        result = {
            'disease_scores': filtered_scores,
            'all_scores': combined_scores,
            'risk_assessments': risk_assessments,
            'overall_risk': overall_risk,
            'component_contributions': component_contributions,
            'fusion_strategy': self.strategy,
            'weights_used': self.weights
        }
        
        # Add top predictions
        if filtered_scores:
            sorted_diseases = sorted(
                filtered_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            result['top_predictions'] = sorted_diseases[:3]
            result['primary_diagnosis'] = sorted_diseases[0]
        else:
            result['top_predictions'] = []
            result['primary_diagnosis'] = ('none', 0.0)
        
        return result
    
    def _determine_overall_risk(
        self,
        disease_scores: Dict[str, float],
        risk_assessments: Dict[str, str]
    ) -> str:
        """
        Determine overall risk level based on disease scores and individual risks.
        
        Args:
            disease_scores: Disease probability scores
            risk_assessments: Individual disease risk levels
        
        Returns:
            Overall risk level string
        """
        if not disease_scores:
            return "low"
        
        # Get risk level for highest-scoring disease
        top_disease = max(disease_scores.items(), key=lambda x: x[1])[0]
        disease_risk = risk_assessments.get(top_disease, "moderate")
        
        # Adjust based on score magnitude
        top_score = disease_scores[top_disease]
        
        severity_levels = ['low', 'moderate', 'high', 'severe']
        
        if disease_risk == 'severe' or top_score > 0.85:
            return 'severe'
        elif disease_risk == 'high' or top_score > 0.6:
            return 'high'
        elif disease_risk == 'moderate' or top_score > 0.3:
            return 'moderate'
        else:
            return 'low'
    
    def adjust_weights(
        self,
        rule_weight: float,
        rf_weight: float,
        cnn_weight: float
    ):
        """
        Dynamically adjust fusion weights.
        
        Args:
            rule_weight: Weight for rule-based component
            rf_weight: Weight for Random Forest
            cnn_weight: Weight for CNN
        """
        # Normalize weights to sum to 1
        total = rule_weight + rf_weight + cnn_weight
        
        self.weights = {
            'rule_based': rule_weight / total,
            'random_forest': rf_weight / total,
            'cnn': cnn_weight / total
        }
        
        logger.info(f"Weights adjusted: {self.weights}")
    
    def explain_fusion(self, fusion_result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of fusion process.
        
        Args:
            fusion_result: Result dictionary from fuse_predictions
        
        Returns:
            Explanation string
        """
        explanation = []
        
        explanation.append("=== Neuro-Symbolic Fusion Analysis ===\n")
        
        # Primary diagnosis
        primary = fusion_result['primary_diagnosis']
        explanation.append(
            f"Primary Diagnosis: {primary[0].upper()} "
            f"(Confidence: {primary[1]:.2%})"
        )
        
        # Risk level
        overall_risk = fusion_result['overall_risk']
        explanation.append(f"Overall Risk Level: {overall_risk.upper()}\n")
        
        # Component contributions
        explanation.append("Component Contributions:")
        contributions = fusion_result['component_contributions']
        
        for component, scores in contributions.items():
            if scores:
                explanation.append(f"\n  {component.replace('_', ' ').title()}:")
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for disease, score in sorted_scores[:3]:
                    if score > 0:
                        explanation.append(f"    - {disease}: {score:.2%}")
        
        # Fusion strategy
        explanation.append(f"\nFusion Strategy: {fusion_result['fusion_strategy']}")
        weights = fusion_result['weights_used']
        explanation.append(
            f"Weights: Rule={weights['rule_based']:.2f}, "
            f"RF={weights['random_forest']:.2f}, "
            f"CNN={weights['cnn']:.2f}"
        )
        
        return "\n".join(explanation)
    
    def generate_recommendations(
        self,
        fusion_result: Dict[str, Any]
    ) -> List[str]:
        """
        Generate clinical recommendations based on fusion results.
        
        Args:
            fusion_result: Result dictionary from fuse_predictions
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        primary_disease, confidence = fusion_result['primary_diagnosis']
        risk_level = fusion_result['overall_risk']
        
        # General recommendations based on risk level
        if risk_level == 'severe':
            recommendations.append("⚠️ URGENT: Immediate medical attention required")
            recommendations.append("Consider emergency department evaluation")
        elif risk_level == 'high':
            recommendations.append("Seek medical evaluation within 24 hours")
            recommendations.append("Consider hospitalization if symptoms worsen")
        elif risk_level == 'moderate':
            recommendations.append("Schedule medical consultation within 48-72 hours")
            recommendations.append("Monitor symptoms closely")
        else:
            recommendations.append("Continue home care and symptom monitoring")
        
        # Disease-specific recommendations
        if primary_disease == 'dengue' and confidence > 0.5:
            recommendations.append("Dengue-specific:")
            recommendations.append("  - Perform dengue NS1 antigen and IgM/IgG serology")
            recommendations.append("  - Monitor platelet count daily")
            recommendations.append("  - Ensure adequate hydration")
            recommendations.append("  - Watch for warning signs: abdominal pain, bleeding")
        
        elif primary_disease == 'covid19' and confidence > 0.5:
            recommendations.append("COVID-19-specific:")
            recommendations.append("  - Perform RT-PCR or rapid antigen test")
            recommendations.append("  - Monitor oxygen saturation")
            recommendations.append("  - Isolate from household members")
            recommendations.append("  - Consider antiviral treatment if high risk")
        
        elif primary_disease == 'pneumonia' and confidence > 0.5:
            recommendations.append("Pneumonia-specific:")
            recommendations.append("  - Obtain chest X-ray")
            recommendations.append("  - Consider sputum culture")
            recommendations.append("  - Monitor respiratory rate and oxygen levels")
            recommendations.append("  - May require antibiotic therapy")
        
        elif primary_disease in ['melanoma', 'eczema', 'psoriasis', 'acne']:
            recommendations.append(f"{primary_disease.title()}-specific:")
            recommendations.append("  - Consult dermatologist for definitive diagnosis")
            if primary_disease == 'melanoma':
                recommendations.append("  - Consider biopsy for suspicious lesions")
                recommendations.append("  - Early detection is critical")
        
        # General monitoring
        recommendations.append("\nGeneral Monitoring:")
        recommendations.append("  - Keep record of symptoms and vital signs")
        recommendations.append("  - Seek re-evaluation if symptoms worsen")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    fusion = NeuroSymbolicFusion()
    
    # Mock predictions
    rule_scores = {
        'dengue': 0.7,
        'covid19': 0.2,
        'pneumonia': 0.1
    }
    
    rf_scores = {
        'dengue': 0.65,
        'covid19': 0.25,
        'pneumonia': 0.10
    }
    
    rule_metadata = {
        'risk_assessments': {
            'dengue': 'high',
            'covid19': 'moderate',
            'pneumonia': 'low'
        }
    }
    
    # Fuse predictions
    result = fusion.fuse_predictions(rule_scores, rf_scores, rule_metadata=rule_metadata)
    
    print(fusion.explain_fusion(result))
    print("\n" + "="*50 + "\n")
    print("Recommendations:")
    for rec in fusion.generate_recommendations(result):
        print(rec)
