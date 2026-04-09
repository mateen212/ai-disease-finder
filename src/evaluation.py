"""
Evaluation Module

Provides utilities for evaluating the hybrid system's performance
across different diseases and components.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation utilities for the hybrid system.
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        self.results = {}
    
    def evaluate_classifier(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
        class_names: List[str] = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a classifier.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            class_names: List of class names
            model_name: Name of the model
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted metrics (account for class imbalance)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # AUC (if probabilities provided)
        if y_prob is not None:
            try:
                # Handle multi-class
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(
                        y_true, y_prob, 
                        multi_class='ovr', 
                        average='macro'
                    )
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics['auc'] = None
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def compare_models(
        self,
        models_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            models_results: Dictionary mapping model names to their results
        
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for model_name, metrics in models_results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision_weighted', 0),
                'Recall': metrics.get('recall_weighted', 0),
                'F1 Score': metrics.get('f1_weighted', 0),
                'AUC': metrics.get('auc', 0) or 0
            }
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('F1 Score', ascending=False)
        
        return df
    
    def plot_confusion_matrix(
        self,
        confusion_mat: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        save_path: str = None
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            confusion_mat: Confusion matrix
            class_names: List of class names
            title: Plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        title: str = "ROC Curves",
        save_path: str = None
    ):
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: List of class names
            title: Plot title
            save_path: Optional path to save plot
        """
        from sklearn.preprocessing import label_binarize
        
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            
            plt.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f'{class_name} (AUC = {auc:.3f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot bar chart comparing models.
        
        Args:
            comparison_df: DataFrame from compare_models()
            save_path: Optional path to save plot
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.15
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = width * (i - 2)
            ax.bar(
                x + offset,
                comparison_df[metric],
                width,
                label=metric,
                color=color,
                alpha=0.8
            )
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def evaluate_rule_engine(
        self,
        test_cases: List[Dict[str, Any]],
        rule_engine: Any
    ) -> Dict[str, Any]:
        """
        Evaluate rule engine coverage and accuracy.
        
        Args:
            test_cases: List of test case dictionaries
            rule_engine: RuleEngine instance
        
        Returns:
            Dictionary with evaluation results
        """
        correct = 0
        total = len(test_cases)
        coverage = 0
        
        results = {
            'test_cases': [],
            'accuracy': 0,
            'coverage': 0
        }
        
        for test_case in test_cases:
            patient_data = test_case['patient_data']
            expected_disease = test_case['expected_disease']
            
            # Evaluate rules
            rule_results = rule_engine.evaluate_rules(patient_data)
            disease_scores = rule_results['disease_scores']
            
            # Check if any rules fired
            if rule_results['rule_count'] > 0:
                coverage += 1
            
            # Check if top prediction matches expected
            if disease_scores:
                predicted = max(disease_scores.items(), key=lambda x: x[1])[0]
                if predicted == expected_disease:
                    correct += 1
                
                results['test_cases'].append({
                    'expected': expected_disease,
                    'predicted': predicted,
                    'scores': disease_scores,
                    'rules_fired': rule_results['rule_count']
                })
        
        results['accuracy'] = correct / total if total > 0 else 0
        results['coverage'] = coverage / total if total > 0 else 0
        
        logger.info(f"Rule Engine - Accuracy: {results['accuracy']:.2%}, Coverage: {results['coverage']:.2%}")
        
        return results
    
    def generate_evaluation_report(
        self,
        all_results: Dict[str, Dict[str, Any]],
        save_path: str = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            all_results: Dictionary with all evaluation results
            save_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = []
        
        report.append("="*80)
        report.append("HYBRID NEURO-SYMBOLIC SYSTEM - EVALUATION REPORT")
        report.append("="*80)
        
        # Overall system performance
        if 'hybrid_system' in all_results:
            report.append("\n📊 OVERALL SYSTEM PERFORMANCE")
            report.append("-"*80)
            metrics = all_results['hybrid_system']
            report.append(f"Accuracy:  {metrics.get('accuracy', 0):.3f}")
            report.append(f"Precision: {metrics.get('precision_weighted', 0):.3f}")
            report.append(f"Recall:    {metrics.get('recall_weighted', 0):.3f}")
            report.append(f"F1 Score:  {metrics.get('f1_weighted', 0):.3f}")
            if metrics.get('auc'):
                report.append(f"AUC:       {metrics['auc']:.3f}")
        
        # Component-wise performance
        report.append("\n🔧 COMPONENT PERFORMANCE")
        report.append("-"*80)
        
        for component in ['rule_engine', 'random_forest', 'cnn']:
            if component in all_results:
                report.append(f"\n{component.replace('_', ' ').title()}:")
                metrics = all_results[component]
                report.append(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}")
                report.append(f"  F1 Score:  {metrics.get('f1_weighted', 0):.3f}")
        
        # Classification report
        if 'hybrid_system' in all_results:
            report.append("\n📋 DETAILED CLASSIFICATION REPORT")
            report.append("-"*80)
            report.append(all_results['hybrid_system'].get('classification_report', 'N/A'))
        
        report.append("\n" + "="*80)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_str


# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # Mock data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    y_prob = np.random.rand(9, 3)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    class_names = ['dengue', 'covid19', 'pneumonia']
    
    metrics = evaluator.evaluate_classifier(
        y_true, y_pred, y_prob,
        class_names=class_names,
        model_name="Test Model"
    )
    
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1_weighted']:.3f}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")
