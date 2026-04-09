"""
Main Application - Hybrid Neuro-Symbolic Clinical Decision Support System

This is the main entry point for using the trained system to diagnose patients.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.rule_engine import RuleEngine
from src.data_preprocessing import DataPreprocessor
from src.ml_models import RandomForestDiagnostic, SkinLesionCNN
from src.fusion import NeuroSymbolicFusion
from src.explainability import ExplainabilityModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridClinicalDSS:
    """
    Main Hybrid Neuro-Symbolic Clinical Decision Support System.
    
    Integrates rule-based reasoning, Random Forest, and CNN for
    multi-disease diagnosis with full explainability.
    """
    
    def __init__(
        self,
        rf_model_path: str = "models/random_forest_clinical.pkl",
        cnn_model_path: str = "models/cnn_skin_lesion_final.pth"
    ):
        """
        Initialize the hybrid system.
        
        Args:
            rf_model_path: Path to trained Random Forest model
            cnn_model_path: Path to trained CNN model
        """
        logger.info("Initializing Hybrid Clinical Decision Support System...")
        
        # Initialize components
        self.rule_engine = RuleEngine()
        self.preprocessor = DataPreprocessor()
        self.fusion = NeuroSymbolicFusion()
        self.explainer = ExplainabilityModule()
        
        # Load trained models
        self.rf_model = None
        self.cnn_model = None
        
        # Load Random Forest
        if Path(rf_model_path).exists():
            self.rf_model = RandomForestDiagnostic()
            self.rf_model.load(rf_model_path)
            logger.info(f"Random Forest model loaded from {rf_model_path}")
            
            # Initialize SHAP explainer
            # Note: We'd need background data for this, skipping for now
            # self.explainer.initialize_shap_explainer(self.rf_model.model, background_data)
        else:
            logger.warning(f"Random Forest model not found at {rf_model_path}")
            logger.warning("Train the model first using: python train.py --train-rf")
        
        # Load CNN
        if Path(cnn_model_path).exists():
            self.cnn_model = SkinLesionCNN()
            self.cnn_model.load(cnn_model_path)
            logger.info(f"CNN model loaded from {cnn_model_path}")
        else:
            logger.warning(f"CNN model not found at {cnn_model_path}")
            logger.info("CNN will not be used for diagnosis")
        
        logger.info("System initialization complete")
    
    def diagnose(
        self,
        patient_data: dict,
        skin_image: str = None
    ) -> dict:
        """
        Perform comprehensive diagnosis on patient data.
        
        Args:
            patient_data: Dictionary with patient clinical data
            skin_image: Optional path to skin lesion image
        
        Returns:
            Dictionary with diagnosis results and explanations
        """
        logger.info("Starting diagnosis...")
        
        # 1. Rule-based inference
        logger.info("Step 1: Evaluating diagnostic rules...")
        rule_results = self.rule_engine.evaluate_rules(patient_data)
        rule_scores = rule_results['disease_scores']
        rule_trace = self.rule_engine.get_rule_trace()
        
        logger.info(f"Rules fired: {rule_results['rule_count']}")
        logger.info(f"Rule scores: {rule_scores}")
        
        # 2. Random Forest prediction
        rf_scores = {}
        shap_explanation = {}
        
        if self.rf_model and self.rf_model.is_trained:
            logger.info("Step 2: Random Forest prediction...")
            
            # Prepare features
            patient_df = self.preprocessor.prepare_clinical_features(patient_data)
            
            # Ensure columns match training data
            missing_cols = set(self.rf_model.feature_names) - set(patient_df.columns)
            for col in missing_cols:
                patient_df[col] = 0
            
            patient_df = patient_df[self.rf_model.feature_names]
            
            # Predict
            rf_scores = self.rf_model.predict_proba(patient_df)
            logger.info(f"Random Forest scores: {rf_scores}")
            
            # SHAP explanation
            try:
                shap_explanation = self.explainer.explain_prediction(
                    patient_df,
                    self.rf_model.feature_names
                )
            except Exception as e:
                logger.warning(f"Could not compute SHAP explanation: {e}")
                shap_explanation = {'explanation': 'SHAP explanation unavailable'}
        else:
            logger.warning("Random Forest model not available, skipping ML prediction")
            # Use rule scores as fallback
            rf_scores = rule_scores.copy()
        
        # 3. CNN prediction (if skin image provided)
        cnn_scores = None
        
        if skin_image and self.cnn_model and self.cnn_model.is_trained:
            logger.info("Step 3: CNN skin lesion analysis...")
            
            try:
                image_tensor = self.preprocessor.preprocess_image(skin_image)
                cnn_scores = self.cnn_model.predict_proba(image_tensor)
                logger.info(f"CNN scores: {cnn_scores}")
            except Exception as e:
                logger.error(f"Error processing skin image: {e}")
        
        # 4. Neuro-symbolic fusion
        logger.info("Step 4: Fusing predictions...")
        
        fusion_result = self.fusion.fuse_predictions(
            rule_scores,
            rf_scores,
            cnn_scores,
            rule_metadata=rule_results
        )
        
        logger.info(f"Final diagnosis: {fusion_result['primary_diagnosis']}")
        
        # 5. Generate recommendations
        recommendations = self.fusion.generate_recommendations(fusion_result)
        
        # 6. Create comprehensive explanation
        combined_explanation = self.explainer.combine_explanations(
            shap_explanation,
            rule_trace,
            fusion_result
        )
        
        # 7. Generate patient report
        patient_report = self.explainer.generate_patient_report(
            patient_data,
            fusion_result,
            shap_explanation,
            rule_trace,
            recommendations
        )
        
        # Compile results
        results = {
            'patient_data': patient_data,
            'diagnosis': fusion_result['primary_diagnosis'],
            'all_disease_scores': fusion_result['disease_scores'],
            'risk_level': fusion_result['overall_risk'],
            'rule_results': {
                'scores': rule_scores,
                'fired_rules': len(rule_trace),
                'trace': rule_trace
            },
            'ml_results': {
                'rf_scores': rf_scores,
                'cnn_scores': cnn_scores
            },
            'fusion_result': fusion_result,
            'recommendations': recommendations,
            'explanation': combined_explanation,
            'patient_report': patient_report
        }
        
        logger.info("Diagnosis complete")
        
        return results
    
    def diagnose_from_file(
        self,
        patient_file: str,
        skin_image: str = None
    ) -> dict:
        """
        Diagnose patient from JSON file.
        
        Args:
            patient_file: Path to JSON file with patient data
            skin_image: Optional path to skin lesion image
        
        Returns:
            Diagnosis results dictionary
        """
        with open(patient_file, 'r') as f:
            patient_data = json.load(f)
        
        return self.diagnose(patient_data, skin_image)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Neuro-Symbolic Clinical Decision Support System"
    )
    
    parser.add_argument(
        '--patient-data',
        type=str,
        help='Path to JSON file with patient data'
    )
    parser.add_argument(
        '--skin-image',
        type=str,
        default=None,
        help='Path to skin lesion image (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save diagnosis report'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with sample patient data'
    )
    parser.add_argument(
        '--rf-model',
        type=str,
        default='models/random_forest_clinical.pkl',
        help='Path to Random Forest model'
    )
    parser.add_argument(
        '--cnn-model',
        type=str,
        default='models/cnn_skin_lesion_final.pth',
        help='Path to CNN model'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = HybridClinicalDSS(
        rf_model_path=args.rf_model,
        cnn_model_path=args.cnn_model
    )
    
    # Demo mode
    if args.demo:
        logger.info("Running demo with sample patient data...")
        
        sample_patient = {
            'symptoms': {
                'fever': True,
                'headache': True,
                'rash': True,
                'nausea': True,
                'retro_orbital_pain': True,
                'cough': False,
                'fatigue': False,
                'loss_of_taste': False
            },
            'vitals': {
                'temperature': 39.5,
                'oxygen_saturation': 96,
                'heart_rate': 88
            },
            'labs': {
                'platelet_count': 95000,
                'wbc_count': 3200,
                'lymphocyte_percentage': 25
            },
            'demographics': {
                'age': 35,
                'gender': 'M',
                'travel_history': True
            }
        }
        
        results = system.diagnose(sample_patient)
        
        # Print report
        print("\n" + "="*80)
        print(results['patient_report'])
        print("="*80 + "\n")
        
        # Save if output specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(results['patient_report'])
            logger.info(f"Report saved to {args.output}")
    
    # File-based diagnosis
    elif args.patient_data:
        if not Path(args.patient_data).exists():
            logger.error(f"Patient data file not found: {args.patient_data}")
            return
        
        results = system.diagnose_from_file(args.patient_data, args.skin_image)
        
        # Print report
        print("\n" + results['patient_report'] + "\n")
        
        # Save if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Full results saved to {args.output}")
            
            # Also save readable report
            report_path = args.output.replace('.json', '_report.txt')
            with open(report_path, 'w') as f:
                f.write(results['patient_report'])
            logger.info(f"Report saved to {report_path}")
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run demo")
        print("  python main.py --demo")
        print("\n  # Diagnose from file")
        print("  python main.py --patient-data patient.json --output results.json")
        print("\n  # With skin image")
        print("  python main.py --patient-data patient.json --skin-image lesion.jpg")


if __name__ == "__main__":
    main()
