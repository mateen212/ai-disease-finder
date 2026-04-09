#!/usr/bin/env python3
"""
Prepare HAM10000 skin lesion data for training
Organizes images and creates labels.csv
"""

import pandas as pd
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def prepare_ham10000_data():
    """Prepare HAM10000 dataset for training"""
    
    # Paths
    skin_dir = Path("data/skin_lesions")
    images_dir = skin_dir / "images"
    metadata_file = skin_dir / "HAM10000_metadata.csv"
    labels_output = skin_dir / "labels.csv"
    
    # Create images directory
    images_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Preparing HAM10000 Skin Lesion Dataset")
    logger.info("=" * 60)
    
    # Read metadata
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return False
    
    logger.info(f"Reading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file)
    logger.info(f"Total entries: {len(metadata)}")
    
    # Get unique images (metadata has duplicates)
    metadata_unique = metadata.drop_duplicates(subset='image_id')
    logger.info(f"Unique images: {len(metadata_unique)}")
    
    # Check diagnosis distribution
    logger.info("\nDiagnosis distribution:")
    diagnosis_map = {
        'mel': 'melanoma',
        'nv': 'nevus',
        'bcc': 'basal_cell_carcinoma',
        'akiec': 'actinic_keratosis',
        'bkl': 'benign_keratosis',
        'df': 'dermatofibroma',
        'vasc': 'vascular_lesion'
    }
    
    for dx_code, count in metadata_unique['dx'].value_counts().items():
        dx_name = diagnosis_map.get(dx_code, dx_code)
        logger.info(f"  {dx_name} ({dx_code}): {count}")
    
    # Copy/symlink images to images directory
    logger.info(f"\nOrganizing images into {images_dir}")
    
    source_dirs = [
        skin_dir / "HAM10000_images_part_1",
        skin_dir / "HAM10000_images_part_2",
        skin_dir / "ham10000_images_part_1",
        skin_dir / "ham10000_images_part_2"
    ]
    
    copied = 0
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
            
        for img_file in source_dir.glob("*.jpg"):
            dest_file = images_dir / img_file.name
            if not dest_file.exists():
                try:
                    # Use symlink for efficiency (or copy if needed)
                    dest_file.symlink_to(img_file.absolute())
                    copied += 1
                except Exception:
                    # If symlink fails, copy instead
                    shutil.copy2(img_file, dest_file)
                    copied += 1
    
    logger.info(f"Organized {copied} images")
    
    # Create labels.csv
    logger.info(f"\nCreating labels file: {labels_output}")
    
    labels_df = metadata_unique[['image_id', 'dx']].copy()
    labels_df.columns = ['image_id', 'diagnosis']
    
    # Add full diagnosis name
    labels_df['diagnosis_name'] = labels_df['diagnosis'].map(diagnosis_map)
    
    # Save
    labels_df.to_csv(labels_output, index=False)
    logger.info(f"Saved {len(labels_df)} labels")
    
    logger.info("\n" + "=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)
    logger.info(f"\nDataset ready for training:")
    logger.info(f"  Images: {images_dir}")
    logger.info(f"  Labels: {labels_output}")
    logger.info(f"  Total samples: {len(labels_df)}")
    logger.info(f"  Classes: {len(labels_df['diagnosis'].unique())}")
    
    return True

if __name__ == "__main__":
    prepare_ham10000_data()
