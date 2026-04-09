"""
Training Script for Hybrid Neuro-Symbolic Clinical Decision Support System

This script handles:
- Data downloading and preprocessing
- Model training (Random Forest and CNN)
- Model evaluation and saving
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_preprocessing import DataPreprocessor
from src.ml_models import RandomForestDiagnostic, SkinLesionCNN, SkinLesionDataset
from src.rule_engine import RuleEngine
from src.evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'outputs', 'logs']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    logger.info("Directories created/verified")


def download_data(preprocessor):
    """Download datasets from Kaggle and other sources"""
    logger.info("="*60)
    logger.info("DATA DOWNLOAD")
    logger.info("="*60)
    logger.info("Run the download script to get all datasets:")
    logger.info("  python download_datasets.py")
    logger.info("")
    logger.info("Or download manually from Kaggle:")
    logger.info("  1. COVID-19: https://www.kaggle.com/meirnizri/covid19-dataset")
    logger.info("  2. Dengue: https://www.kaggle.com/mdtuser/dengue-dataset")
    logger.info("  3. Skin: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000")
    logger.info("")
    
    # Check if data exists
    data_dirs = ['data/covid19', 'data/dengue', 'data/skin_lesions', 'data/clinical']
    has_data = any(Path(d).exists() and any(Path(d).iterdir()) for d in data_dirs)
    
    if not has_data:
        logger.warning("No downloaded data found. Generating synthetic data for training...")
        
        # Generate synthetic clinical data
        synthetic_data = preprocessor.create_synthetic_clinical_data(
            n_samples=2000,
            save_path="data/clinical_training_data.csv"
        )
        logger.info(f"Generated {len(synthetic_data)} synthetic samples")
    else:
        logger.info("✓ Found downloaded datasets, will use real data")


def train_random_forest(args):
    """Train Random Forest classifier on clinical data"""
    logger.info("="*60)
    logger.info("TRAINING RANDOM FOREST CLASSIFIER")
    logger.info("="*60)
    
    # Initialize
    preprocessor = DataPreprocessor()
    rf_model = RandomForestDiagnostic()
    
    # Load or generate data
    data_file = "data/clinical_training_data.csv"
    if not Path(data_file).exists():
        logger.info("Clinical data not found, generating synthetic data...")
        data = preprocessor.create_synthetic_clinical_data(
            n_samples=2000,
            save_path=data_file
        )
    else:
        data = pd.read_csv(data_file)
    
    logger.info(f"Loaded {len(data)} samples")
    logger.info(f"Class distribution:\n{data['diagnosis'].value_counts()}")
    
    # Prepare dataset
    X_train, X_test, y_train, y_test = preprocessor.prepare_clinical_dataset(
        data,
        target_column='diagnosis',
        test_size=0.2
    )
    
    # Train model
    metrics = rf_model.train(X_train, y_train, X_test, y_test)
    logger.info(f"Training completed: {metrics}")
    
    # Evaluate
    evaluator = ModelEvaluator()
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.model.predict_proba(X_test)
    
    class_names = rf_model.classes.tolist()
    eval_metrics = evaluator.evaluate_classifier(
        y_test.values,
        y_pred,
        y_prob,
        class_names=class_names,
        model_name="Random Forest"
    )
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"Accuracy: {eval_metrics['accuracy']:.3f}")
    logger.info(f"F1 Score: {eval_metrics['f1_weighted']:.3f}")
    
    # Save model
    model_path = "models/random_forest_clinical.pkl"
    rf_model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save confusion matrix plot
    evaluator.plot_confusion_matrix(
        eval_metrics['confusion_matrix'],
        class_names,
        title="Random Forest - Confusion Matrix",
        save_path="outputs/rf_confusion_matrix.png"
    )
    
    # Feature importance
    importance_df = rf_model.get_feature_importances()
    logger.info(f"\nTop 10 Important Features:")
    logger.info(importance_df.head(10).to_string())
    importance_df.to_csv("outputs/rf_feature_importance.csv", index=False)
    
    return rf_model, eval_metrics


def train_cnn(args):
    """Train CNN classifier on skin lesion images"""
    logger.info("="*60)
    logger.info("TRAINING CNN FOR SKIN LESION CLASSIFICATION")
    logger.info("="*60)
    
    # Check if image data exists
    image_dir = "data/skin_lesions/images"
    labels_file = "data/skin_lesions/labels.csv"
    
    if not Path(image_dir).exists() or not Path(labels_file).exists():
        logger.warning("Skin lesion data not found. Skipping CNN training.")
        logger.info("To train CNN, download skin lesion datasets from Kaggle")
        logger.info("Expected structure:")
        logger.info("  data/skin_lesions/images/")
        logger.info("  data/skin_lesions/labels.csv")
        return None, None
    
    # Initialize
    preprocessor = DataPreprocessor()
    cnn_model = SkinLesionCNN()
    
    # Check for existing checkpoint to resume
    checkpoint_path = "models/cnn_skin_lesion_checkpoint.pth"
    start_epoch = 0
    best_val_acc = 0.0
    
    if Path(checkpoint_path).exists():
        logger.info(f"Found existing checkpoint: {checkpoint_path}")
        try:
            checkpoint_info = cnn_model.load(checkpoint_path)
            start_epoch = checkpoint_info['epoch'] + 1  # Start from next epoch
            best_val_acc = checkpoint_info['best_val_acc']
            logger.info(f"✓ Resuming from epoch {start_epoch+1}, best_val_acc={best_val_acc:.2f}%")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_val_acc = 0.0
    
    # Prepare dataset
    train_paths, test_paths, train_labels, test_labels = preprocessor.prepare_image_dataset(
        image_dir,
        labels_file,
        test_size=0.2
    )
    
    # Create data loaders
    train_dataset = SkinLesionDataset(
        train_paths,
        train_labels,
        transform=preprocessor.train_transform
    )
    
    test_dataset = SkinLesionDataset(
        test_paths,
        test_labels,
        transform=preprocessor.test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cnn_model.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cnn_model.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Train model
    history = cnn_model.train(
        train_loader,
        test_loader,
        save_best=True,
        model_save_path="models/cnn_skin_lesion.pth",
        start_epoch=start_epoch,
        best_val_acc=best_val_acc
    )
    
    # Evaluate
    test_loss, test_acc = cnn_model.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Save model
    cnn_model.save("models/cnn_skin_lesion_final.pth")
    
    return cnn_model, history


def test_rule_engine():
    """Test rule engine on known cases"""
    logger.info("="*60)
    logger.info("TESTING RULE ENGINE")
    logger.info("="*60)
    
    rule_engine = RuleEngine()
    
    # Test cases
    test_cases = [
        {
            'name': 'Classic Dengue',
            'patient_data': {
                'symptoms': {
                    'fever': True,
                    'headache': True,
                    'rash': True,
                    'nausea': True,
                    'retro_orbital_pain': True
                },
                'vitals': {
                    'temperature': 39.5,
                    'oxygen_saturation': 96
                },
                'labs': {
                    'platelet_count': 95000,
                    'wbc_count': 3200
                }
            },
            'expected_disease': 'dengue'
        },
        {
            'name': 'COVID-19 Case',
            'patient_data': {
                'symptoms': {
                    'fever': True,
                    'cough': True,
                    'fatigue': True,
                    'loss_of_taste': True,
                    'shortness_of_breath': True
                },
                'vitals': {
                    'temperature': 38.5,
                    'oxygen_saturation': 93
                },
                'labs': {
                    'platelet_count': 200000,
                    'wbc_count': 5500
                }
            },
            'expected_disease': 'covid19'
        }
    ]
    
    correct = 0
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        
        results = rule_engine.evaluate_rules(test_case['patient_data'])
        disease_scores = results['disease_scores']
        
        logger.info(f"Fired {results['rule_count']} rules")
        logger.info(f"Scores: {disease_scores}")
        
        if disease_scores:
            predicted = max(disease_scores.items(), key=lambda x: x[1])[0]
            expected = test_case['expected_disease']
            
            if predicted == expected:
                logger.info(f"✓ Correct prediction: {predicted}")
                correct += 1
            else:
                logger.info(f"✗ Incorrect: predicted {predicted}, expected {expected}")
        
        rule_engine.reset()
    
    accuracy = correct / len(test_cases)
    logger.info(f"\nRule Engine Accuracy on test cases: {accuracy:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Neuro-Symbolic Clinical Decision Support System"
    )
    parser.add_argument(
        '--download-data',
        action='store_true',
        help='Download datasets from Kaggle'
    )
    parser.add_argument(
        '--train-rf',
        action='store_true',
        help='Train Random Forest classifier'
    )
    parser.add_argument(
        '--train-cnn',
        action='store_true',
        help='Train CNN for skin lesions'
    )
    parser.add_argument(
        '--test-rules',
        action='store_true',
        help='Test rule engine'
    )
    parser.add_argument(
        '--train-all',
        action='store_true',
        help='Train all components'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # If no specific action, default to train-all
    if not any([args.download_data, args.train_rf, args.train_cnn, 
                args.test_rules, args.train_all]):
        args.train_all = True
    
    # Download data
    if args.download_data or args.train_all:
        preprocessor = DataPreprocessor()
        download_data(preprocessor)
    
    # Test rule engine
    if args.test_rules or args.train_all:
        test_rule_engine()
    
    # Train Random Forest
    if args.train_rf or args.train_all:
        rf_model, rf_metrics = train_random_forest(args)
    
    # Train CNN
    if args.train_cnn or args.train_all:
        cnn_model, cnn_history = train_cnn(args)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    logger.info("\nTrained models saved in: models/")
    logger.info("Evaluation results saved in: outputs/")
    logger.info("\nTo use the system, run: python main.py --help")


if __name__ == "__main__":
    main()
