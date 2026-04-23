"""
Machine Learning Models Module

Implements Random Forest classifier for clinical data and
CNN for skin lesion classification.
"""

import os
import yaml
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestDiagnostic:
    """
    Random Forest classifier for multi-disease diagnosis from clinical data.
    
    Handles symptoms, vitals, lab values, and demographics to predict
    disease probabilities.
    """
    
    def __init__(self, config_file: str = "config/model_config.yaml"):
        """
        Initialize the Random Forest model.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        rf_config = self.config.get('random_forest', {})
        
        self.model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 20),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            max_features=rf_config.get('max_features', 'sqrt'),
            bootstrap=rf_config.get('bootstrap', True),
            class_weight=rf_config.get('class_weight', 'balanced'),
            random_state=rf_config.get('random_state', 42),
            n_jobs=rf_config.get('n_jobs', -1)
        )
        
        self.feature_names = None
        self.classes = None
        self.is_trained = False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training Random Forest model...")
        
        # Store feature names and classes
        self.feature_names = X_train.columns.tolist()
        self.classes = np.unique(y_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        metrics = {
            'train_accuracy': train_acc,
            'n_features': len(self.feature_names),
            'n_classes': len(self.classes)
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            metrics['val_accuracy'] = val_acc
            
            logger.info(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        else:
            logger.info(f"Train Accuracy: {train_acc:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict disease classes.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predicted classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Predict disease probabilities.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Dictionary mapping disease names to probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            disease: float(prob) 
            for disease, prob in zip(self.classes, probabilities)
        }
    
    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'classes': self.classes
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.classes = data['classes']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for skin lesion images"""
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SkinLesionCNN:
    """
    CNN for skin lesion classification.
    
    Uses pretrained models (EfficientNet, ResNet, etc.) and fine-tunes
    for melanoma, eczema, psoriasis, and acne classification.
    """
    
    def __init__(self, config_file: str = "config/model_config.yaml"):
        """
        Initialize the CNN model.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        cnn_config = self.config.get('cnn', {})
        
        self.architecture = cnn_config.get('architecture', 'efficientnet_b0')
        self.num_classes = cnn_config.get('num_classes', 4)
        self.input_size = cnn_config.get('input_size', 512)
        self.batch_size = cnn_config.get('batch_size', 32)
        self.learning_rate = cnn_config.get('learning_rate', 0.001)
        self.epochs = cnn_config.get('epochs', 50)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model(cnn_config.get('pretrained', True))
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        self.class_names = None
        self.is_trained = False
        
        logger.info(f"CNN initialized: {self.architecture} on {self.device}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _create_model(self, pretrained: bool = True) -> nn.Module:
        """
        Create CNN model using timm library.
        
        Args:
            pretrained: Whether to use pretrained weights
        
        Returns:
            PyTorch model
        """
        # Create model using timm
        model = timm.create_model(
            self.architecture,
            pretrained=pretrained,
            num_classes=self.num_classes
        )
        
        return model
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_best: bool = True,
        model_save_path: str = "models/cnn_best.pth",
        start_epoch: int = 0,
        best_val_acc: float = 0.0
    ) -> Dict[str, List[float]]:
        """
        Train the CNN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            save_best: Whether to save best model
            model_save_path: Path to save model
            start_epoch: Epoch to start/resume from
            best_val_acc: Best validation accuracy so far (for resume)
        
        Returns:
            Dictionary with training history
        """
        if start_epoch > 0:
            logger.info(f"Resuming training from epoch {start_epoch+1}/{self.epochs}")
        else:
            logger.info("Training CNN model...")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(start_epoch, self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / train_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logger.info(
                    f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, "
                    f"Train Acc={epoch_train_acc:.2f}%, Val Loss={val_loss:.4f}, "
                    f"Val Acc={val_acc:.2f}%"
                )
                
                # Save best model
                if save_best and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save(model_save_path, epoch=epoch, best_val_acc=best_val_acc)
                    logger.info(f"Best model saved with val_acc={val_acc:.2f}%")
                
                # Save checkpoint for resume (every epoch)
                try:
                    checkpoint_path = model_save_path.replace('.pth', '_checkpoint.pth')
                    self.save(checkpoint_path, epoch=epoch, best_val_acc=best_val_acc)
                    logger.info(f"✓ Checkpoint saved for epoch {epoch+1}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")
            else:
                logger.info(
                    f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, "
                    f"Train Acc={epoch_train_acc:.2f}%"
                )
        
        self.is_trained = True
        return history
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, image: torch.Tensor) -> int:
        """
        Predict class for a single image.
        
        Args:
            image: Preprocessed image tensor
        
        Returns:
            Predicted class index
        """
        self.model.eval()
        
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            image = image.to(self.device)
            outputs = self.model(image)
            _, predicted = outputs.max(1)
        
        return predicted.item()
    
    def predict_proba(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Predict probabilities for all classes.
        
        Args:
            image: Preprocessed image tensor
        
        Returns:
            Dictionary mapping class names to probabilities
        """
        self.model.eval()
        
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            image = image.to(self.device)
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Default class names if not set
        if self.class_names is None:
            self.class_names = ['melanoma', 'eczema', 'psoriasis', 'acne']
        
        return {
            class_name: float(prob)
            for class_name, prob in zip(self.class_names, probabilities.cpu().numpy())
        }
    
    def save(self, filepath: str, epoch: int = None, best_val_acc: float = None):
        """Save the trained model with checkpoint info"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'class_names': self.class_names
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if best_val_acc is not None:
            checkpoint['best_val_acc'] = best_val_acc
        
        torch.save(checkpoint, filepath)
        
        logger.info(f"CNN model saved to {filepath}")
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load a trained model and return checkpoint info"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.class_names = checkpoint.get('class_names')
        self.is_trained = True
        
        epoch = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        logger.info(f"CNN model loaded from {filepath}")
        logger.info(f"Loaded checkpoint: epoch={epoch}, best_val_acc={best_val_acc:.2f}%")
        
        return {
            'epoch': epoch,
            'best_val_acc': best_val_acc
        }


# Example usage
if __name__ == "__main__":
    # Test Random Forest
    print("\n=== Testing Random Forest ===")
    rf_model = RandomForestDiagnostic()
    
    # Create dummy data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=20, n_classes=3, random_state=42)
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)
    
    metrics = rf_model.train(X_train_df, y_train_series)
    print(f"Training metrics: {metrics}")
    
    # Test CNN
    print("\n=== Testing CNN ===")
    cnn_model = SkinLesionCNN()
    print(f"CNN architecture: {cnn_model.architecture}")
    print(f"Device: {cnn_model.device}")
