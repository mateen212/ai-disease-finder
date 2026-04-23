"""
Data Preprocessing Module

Handles data downloading from Kaggle, cleaning, feature engineering,
and preparation for ML models.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import torch
from torchvision import transforms
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing tasks for clinical and image data.
    """
    
    def __init__(self, config_file: str = "config/model_config.yaml"):
        """
        Initialize the preprocessor.
        
        Args:
            config_file: Path to model configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.feature_groups = self.config.get('feature_groups', {})
        
        # Scalers and encoders
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
        # Image transforms
        self._setup_image_transforms()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _setup_image_transforms(self):
        """Setup image preprocessing and augmentation transforms"""
        cnn_config = self.config.get('cnn', {})
        input_size = cnn_config.get('input_size', 512)
        norm_mean = cnn_config['normalization']['mean']
        norm_std = cnn_config['normalization']['std']
        
        # Training transforms (with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
        
        # Validation/test transforms (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    
    def download_kaggle_datasets(self, data_dir: str = "data"):
        """
        Download datasets from Kaggle using the Kaggle API.
        
        Args:
            data_dir: Directory to save downloaded datasets
        """
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        kaggle_datasets = self.config.get('kaggle_datasets', {})
        
        try:
            import kaggle
            
            for category, dataset_ids in kaggle_datasets.items():
                category_dir = data_path / category
                category_dir.mkdir(exist_ok=True)
                
                for dataset_id in dataset_ids:
                    logger.info(f"Downloading {dataset_id}...")
                    try:
                        kaggle.api.dataset_download_files(
                            dataset_id,
                            path=str(category_dir),
                            unzip=True,
                            quiet=False
                        )
                        logger.info(f"Successfully downloaded {dataset_id}")
                    except Exception as e:
                        logger.error(f"Error downloading {dataset_id}: {e}")
        
        except ImportError:
            logger.error("Kaggle package not installed. Run: pip install kaggle")
        except Exception as e:
            logger.error(f"Error with Kaggle API: {e}")
            logger.info("Make sure kaggle.json is in ~/.kaggle/")
    
    def prepare_clinical_features(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert patient data dictionary to feature DataFrame.
        
        Args:
            patient_data: Dictionary with symptoms, vitals, labs, demographics
        
        Returns:
            DataFrame with processed features
        """
        features = {}
        
        # Extract all feature groups
        for group_name in ['symptoms', 'vitals', 'labs', 'demographics']:
            group_data = patient_data.get(group_name, {})
            
            # Get expected features for this group
            expected_features = self.feature_groups.get(group_name, [])
            
            for feature in expected_features:
                feature_name = f"{group_name}_{feature}"
                value = group_data.get(feature)
                
                # Handle boolean symptoms
                if group_name == 'symptoms' and isinstance(value, bool):
                    features[feature_name] = 1 if value else 0
                else:
                    features[feature_name] = value
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        return df
    
    def prepare_clinical_dataset(
        self, 
        data: pd.DataFrame,
        target_column: str = 'diagnosis',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare clinical dataset for training.
        
        Args:
            data: Raw clinical data DataFrame
            target_column: Name of target column
            test_size: Proportion of data for testing
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and target
        y = data[target_column]
        X = data.drop(columns=[target_column])
        
        # Encode categorical variables first
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_encoded = X.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values after encoding
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_encoded),
            columns=X_encoded.columns,
            index=X_encoded.index
        )
        
        # Scale numerical features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size,
            random_state=self.config.get('training', {}).get('random_state', 42),
            stratify=y
        )
        
        logger.info(f"Dataset prepared: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_image(
        self, 
        image_path: str, 
        is_training: bool = False
    ) -> torch.Tensor:
        """
        Preprocess a single image for CNN input.
        
        Args:
            image_path: Path to image file
            is_training: Whether to apply training augmentations
        
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply appropriate transform
            transform = self.train_transform if is_training else self.test_transform
            image_tensor = transform(image)
            
            return image_tensor
        
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def prepare_image_dataset(
        self,
        image_dir: str,
        labels_file: str,
        test_size: float = 0.2
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Prepare image dataset for training.
        
        Args:
            image_dir: Directory containing images
            labels_file: CSV file with image labels
            test_size: Proportion for testing
        
        Returns:
            train_paths, test_paths, train_labels, test_labels
        """
        # Load labels
        labels_df = pd.read_csv(labels_file)
        
        # Get image paths and labels
        image_paths = [
            os.path.join(image_dir, fname + '.jpg') 
            for fname in labels_df['image_id'].values
        ]
        # Use 'diagnosis' column from HAM10000 metadata
        labels = labels_df['diagnosis'].values
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        self.label_encoders['skin_condition'] = le
        
        # Split dataset
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, encoded_labels,
            test_size=test_size,
            random_state=self.config.get('training', {}).get('random_state', 42),
            stratify=encoded_labels
        )
        
        logger.info(f"Image dataset prepared: {len(train_paths)} train, {len(test_paths)} test")
        
        return train_paths, test_paths, train_labels.tolist(), test_labels.tolist()
    
    def prepare_folder_based_image_dataset(
        self,
        train_dir: str,
        test_dir: str,
        selected_classes: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], List[int], List[int], List[str]]:
        """
        Prepare image dataset from folder structure where each folder is a class.
        
        Args:
            train_dir: Directory containing training images in subfolders
            test_dir: Directory containing test images in subfolders  
            selected_classes: List of specific class folder names to include (None = all)
        
        Returns:
            train_paths, test_paths, train_labels, test_labels, class_names
        """
        train_path = Path(train_dir)
        test_path = Path(test_dir)
        
        # Get all class folders
        all_class_folders = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
        
        # Filter to selected classes if specified
        if selected_classes:
            class_folders = [f for f in all_class_folders if f in selected_classes]
            logger.info(f"Selected {len(class_folders)} classes: {class_folders}")
        else:
            class_folders = all_class_folders
            logger.info(f"Using all {len(class_folders)} classes")
        
        if not class_folders:
            raise ValueError(f"No classes found! Selected: {selected_classes}, Available: {all_class_folders}")
        
        # Create label encoder mapping
        class_to_label = {cls: idx for idx, cls in enumerate(class_folders)}
        
        # Collect training images
        train_paths = []
        train_labels = []
        for class_name in class_folders:
            class_dir = train_path / class_name
            if not class_dir.exists():
                logger.warning(f"Training folder not found: {class_dir}")
                continue
                
            # Get all images in this class folder
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                train_paths.append(str(img_path))
                train_labels.append(class_to_label[class_name])
            
            logger.info(f"  {class_name}: {len(image_files)} training images")
        
        # Collect test images
        test_paths = []
        test_labels = []
        for class_name in class_folders:
            class_dir = test_path / class_name
            if not class_dir.exists():
                logger.warning(f"Test folder not found: {class_dir}")
                continue
                
            # Get all images in this class folder
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                test_paths.append(str(img_path))
                test_labels.append(class_to_label[class_name])
            
            logger.info(f"  {class_name}: {len(image_files)} test images")
        
        logger.info(f"\nDataset prepared: {len(train_paths)} train, {len(test_paths)} test images")
        logger.info(f"Classes: {class_folders}")
        
        return train_paths, test_paths, train_labels, test_labels, class_folders
    
    def prepare_raw_skin_dataset(
        self,
        raw_dir: str = "data/skin_lesions_raw"
    ) -> Tuple[List[str], List[str], List[int], List[int], List[str]]:
        """
        Prepare image dataset from raw Kaggle structure where dataset is organized as:
        raw_dir/<disease_short_name>/train/*.jpg
        raw_dir/<disease_short_name>/val/*.jpg
        raw_dir/<disease_short_name>/test/*.jpg
        
        Args:
            raw_dir: Directory containing raw downloaded datasets
        
        Returns:
            train_paths, val_paths, train_labels, val_labels, class_names
        """
        raw_path = Path(raw_dir)
        
        # Disease mapping from short folder names to full class names
        disease_mapping = {
            "Acne and Rosacea Photos",
            "Eczema Photos",
            "Melanoma Skin Cancer Nevi and Moles",
            "Normal Healthy Skin",
            "Psoriasis pictures Lichen Planus and related diseases"
        }
        
        class_names = list(disease_mapping.values())
        class_to_label = {full_name: idx for idx, full_name in enumerate(class_names)}
        
        train_paths = []
        train_labels = []
        val_paths = []
        val_labels = []
        
        logger.info(f"Loading images from raw dataset: {raw_dir}")
        
        for short_name, full_name in disease_mapping.items():
            disease_dir = raw_path / short_name
            
            if not disease_dir.exists():
                logger.warning(f"Skipping {short_name} - directory not found")
                continue
            
            # Load training images
            train_dir = disease_dir / 'train'
            if train_dir.exists():
                train_count = 0
                for img_path in train_dir.rglob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        train_paths.append(str(img_path))
                        train_labels.append(class_to_label[full_name])
                        train_count += 1
                logger.info(f"  {short_name}/train: {train_count} images")
            
            # Load validation images (check both 'val' and 'valid')
            val_dir = disease_dir / 'val'
            valid_dir = disease_dir / 'valid'
            
            if val_dir.exists():
                val_count = 0
                for img_path in val_dir.rglob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        val_paths.append(str(img_path))
                        val_labels.append(class_to_label[full_name])
                        val_count += 1
                logger.info(f"  {short_name}/val: {val_count} images")
            elif valid_dir.exists():
                val_count = 0
                for img_path in valid_dir.rglob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        val_paths.append(str(img_path))
                        val_labels.append(class_to_label[full_name])
                        val_count += 1
                logger.info(f"  {short_name}/valid: {val_count} images")
            else:
                # If no val folder, use test folder for validation
                test_dir = disease_dir / 'test'
                if test_dir.exists():
                    val_count = 0
                    for img_path in test_dir.rglob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            val_paths.append(str(img_path))
                            val_labels.append(class_to_label[full_name])
                            val_count += 1
                    logger.info(f"  {short_name}/test (used as val): {val_count} images")
        
        logger.info(f"\nRaw dataset prepared: {len(train_paths)} train, {len(val_paths)} val images")
        logger.info(f"Classes: {class_names}")
        
        return train_paths, val_paths, train_labels, val_labels, class_names
    
    def create_synthetic_clinical_data(
        self, 
        n_samples: int = 1000,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create synthetic clinical data for demonstration purposes.
        
        Args:
            n_samples: Number of samples to generate
            save_path: Optional path to save CSV file
        
        Returns:
            DataFrame with synthetic patient data
        """
        np.random.seed(42)
        
        diseases = ['dengue', 'covid19', 'pneumonia', 'none']
        data = []
        
        for _ in range(n_samples):
            # Randomly select disease
            disease = np.random.choice(diseases, p=[0.25, 0.25, 0.25, 0.25])
            
            # Generate symptoms based on disease
            if disease == 'dengue':
                sample = {
                    'fever': np.random.choice([0, 1], p=[0.1, 0.9]),
                    'headache': np.random.choice([0, 1], p=[0.2, 0.8]),
                    'rash': np.random.choice([0, 1], p=[0.3, 0.7]),
                    'nausea': np.random.choice([0, 1], p=[0.4, 0.6]),
                    'cough': np.random.choice([0, 1], p=[0.9, 0.1]),
                    'temperature': np.random.normal(39, 0.5),
                    'platelet_count': np.random.normal(85000, 15000),
                    'wbc_count': np.random.normal(3500, 500),
                    'oxygen_saturation': np.random.normal(97, 1)
                }
            elif disease == 'covid19':
                sample = {
                    'fever': np.random.choice([0, 1], p=[0.2, 0.8]),
                    'cough': np.random.choice([0, 1], p=[0.2, 0.8]),
                    'fatigue': np.random.choice([0, 1], p=[0.3, 0.7]),
                    'loss_of_taste': np.random.choice([0, 1], p=[0.5, 0.5]),
                    'rash': np.random.choice([0, 1], p=[0.9, 0.1]),
                    'temperature': np.random.normal(38.5, 0.5),
                    'oxygen_saturation': np.random.normal(94, 2),
                    'platelet_count': np.random.normal(200000, 50000),
                    'wbc_count': np.random.normal(6000, 1500)
                }
            elif disease == 'pneumonia':
                sample = {
                    'fever': np.random.choice([0, 1], p=[0.2, 0.8]),
                    'cough': np.random.choice([0, 1], p=[0.1, 0.9]),
                    'chest_pain': np.random.choice([0, 1], p=[0.4, 0.6]),
                    'shortness_of_breath': np.random.choice([0, 1], p=[0.3, 0.7]),
                    'rash': np.random.choice([0, 1], p=[0.95, 0.05]),
                    'temperature': np.random.normal(38.8, 0.5),
                    'oxygen_saturation': np.random.normal(92, 2),
                    'platelet_count': np.random.normal(220000, 40000),
                    'wbc_count': np.random.normal(12000, 2000)
                }
            else:  # none
                sample = {
                    'fever': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'cough': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'rash': np.random.choice([0, 1], p=[0.9, 0.1]),
                    'temperature': np.random.normal(37, 0.3),
                    'oxygen_saturation': np.random.normal(98, 0.5),
                    'platelet_count': np.random.normal(250000, 40000),
                    'wbc_count': np.random.normal(7000, 1500)
                }
            
            # Add demographics
            sample['age'] = np.random.randint(18, 80)
            sample['gender'] = np.random.choice(['M', 'F'])
            sample['diagnosis'] = disease
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        # Clip values to realistic ranges
        df['platelet_count'] = df['platelet_count'].clip(20000, 500000)
        df['wbc_count'] = df['wbc_count'].clip(1000, 20000)
        df['oxygen_saturation'] = df['oxygen_saturation'].clip(85, 100)
        df['temperature'] = df['temperature'].clip(35, 42)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Synthetic data saved to {save_path}")
        
        return df


# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Generate synthetic data
    synthetic_data = preprocessor.create_synthetic_clinical_data(
        n_samples=1000,
        save_path="data/synthetic_clinical_data.csv"
    )
    
    print("\n=== Synthetic Data Sample ===")
    print(synthetic_data.head())
    print(f"\nShape: {synthetic_data.shape}")
    print(f"Diseases: {synthetic_data['diagnosis'].value_counts()}")
