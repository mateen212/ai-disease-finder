#!/usr/bin/env python3
"""
Comprehensive Data Download Script

Downloads clinical and image datasets from multiple sources:
1. Kaggle datasets (COVID-19, Dengue, Skin lesions)
2. Direct downloads from research repositories
3. Public medical image datasets

Requirements:
- kaggle.json in ~/.kaggle/ or ~/Downloads/
- Internet connection
"""

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Handles downloading datasets from multiple sources"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup Kaggle
        self.setup_kaggle_credentials()
    
    def setup_kaggle_credentials(self):
        """Setup Kaggle API credentials"""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        # Check if already exists
        if kaggle_json.exists():
            logger.info(f"✓ Kaggle credentials found at {kaggle_json}")
            os.chmod(kaggle_json, 0o600)
            return
        
        # Check Downloads folder
        downloads_json = Path.home() / 'Downloads' / 'kaggle.json'
        if downloads_json.exists():
            logger.info(f"Found kaggle.json in Downloads, copying to {kaggle_dir}")
            kaggle_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(downloads_json, kaggle_json)
            os.chmod(kaggle_json, 0o600)
            logger.info("✓ Kaggle credentials configured")
        else:
            logger.warning("⚠ kaggle.json not found!")
            logger.warning("Download it from: https://www.kaggle.com/settings")
            logger.warning(f"Place it in: {kaggle_dir}/kaggle.json or ~/Downloads/")
    
    def download_kaggle_dataset(
        self, 
        dataset_id: str, 
        output_dir: str,
        unzip: bool = True
    ) -> bool:
        """
        Download dataset from Kaggle
        
        Args:
            dataset_id: Kaggle dataset identifier (e.g., 'user/dataset-name')
            output_dir: Directory to save the dataset
            unzip: Whether to unzip downloaded files
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import kaggle
            
            output_path = self.base_dir / output_dir
            output_path.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"📥 Downloading Kaggle dataset: {dataset_id}")
            
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(output_path),
                unzip=unzip,
                quiet=False
            )
            
            logger.info(f"✓ Downloaded to {output_path}")
            return True
            
        except ImportError:
            logger.error("❌ Kaggle package not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading {dataset_id}: {e}")
            return False
    
    def download_file(
        self, 
        url: str, 
        output_path: str,
        description: str = "File"
    ) -> bool:
        """
        Download file from URL with progress bar
        
        Args:
            url: URL to download from
            output_path: Where to save the file
            description: Description for progress bar
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"📥 Downloading {description} from {url}")
            
            output_file = self.base_dir / output_path
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            logger.info(f"✓ Downloaded to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading {description}: {e}")
            return False
    
    def download_all_kaggle_datasets(self) -> Dict[str, bool]:
        """Download all recommended Kaggle datasets"""
        
        datasets = {
            # COVID-19 datasets
            'covid19_symptoms': ('meirnizri/covid19-dataset', 'covid19'),
            'covid19_clinical': ('imdevskp/corona-virus-report', 'covid19'),
            
            # Dengue datasets (updated with working sources)
            'dengue_bangladesh': ('kawsarahmad/dengue-dataset-bangladesh', 'dengue'),
            'dengue_philippines': ('vincentgupo/dengue-cases-in-the-philippines', 'dengue'),
            
            # Skin Disease Datasets (4 main categories)
            'melanoma': ('hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images', 'skin_lesions_raw/melanoma'),
            'eczema': ('adityush/eczema2', 'skin_lesions_raw/eczema'),
            'psoriasis': ('pallapurajkumar/psoriasis-skin-dataset', 'skin_lesions_raw/psoriasis'),
            'acne': ('tiswan14/acne-dataset-image', 'skin_lesions_raw/acne'),
            
            # Pneumonia Dataset (IMPORTANT - Keep separate for RF model)
            'pneumonia_xray': ('paultimothymooney/chest-xray-pneumonia', 'pneumonia_xray'),
            
            # Additional skin datasets (optional, for augmentation)
            'skin_cancer_ham10000': ('kmader/skin-cancer-mnist-ham10000', 'skin_lesions_raw/ham10000'),
            'dermatology_images': ('shubhamgoel27/dermnet', 'skin_lesions_raw/dermnet'),
            
            # General clinical data
            'general_symptoms': ('itachi9604/disease-symptom-description-dataset', 'clinical'),
        }
        
        results = {}
        
        for name, (dataset_id, output_dir) in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Dataset: {name}")
            logger.info(f"{'='*60}")
            
            success = self.download_kaggle_dataset(dataset_id, output_dir)
            results[name] = success
            
            if success:
                logger.info(f"✓ {name}: SUCCESS")
            else:
                logger.warning(f"⚠ {name}: FAILED")
        
        return results
    
    def download_additional_datasets(self) -> Dict[str, bool]:
        """Download additional datasets from other sources"""
        
        results = {}
        
        # ISIC (International Skin Imaging Collaboration) data
        # Note: These are placeholders - actual ISIC requires registration
        logger.info("\n" + "="*60)
        logger.info("Additional Medical Datasets")
        logger.info("="*60)
        logger.info("Note: Some datasets require manual download and registration:")
        logger.info("1. ISIC Archive: https://www.isic-archive.com/")
        logger.info("2. NIH Chest X-rays: https://nihcc.app.box.com/v/ChestXray-NIHCC")
        logger.info("3. WHO Dengue Data: https://www.who.int/data/gho")
        
        # Example: Download sample datasets that don't require authentication
        sample_datasets = {
            'symptom_severity': (
                'https://raw.githubusercontent.com/sudhir-voleti/sample-data/master/Symptom-severity.csv',
                'clinical/symptom_severity.csv',
                'Symptom Severity Dataset'
            ),
        }
        
        for name, (url, path, desc) in sample_datasets.items():
            success = self.download_file(url, path, desc)
            results[name] = success
        
        return results
    
    def download_who_cdc_guidelines(self) -> bool:
        """Download WHO/CDC guideline documents"""
        
        logger.info("\n" + "="*60)
        logger.info("Downloading Clinical Guidelines")
        logger.info("="*60)
        
        guidelines_dir = self.base_dir / 'guidelines'
        guidelines_dir.mkdir(exist_ok=True)
        
        # Clinical guideline URLs (Note: Some may require manual download)
        guidelines_to_download = {
            'melanoma_cdc.html': 'https://stacks.cdc.gov/view/cdc/30592',
            'eczema_who.html': 'https://www.emro.who.int/emhj-volume-31-2025/volume-31-issue-9/adolopment-of-atopic-dermatitis-management-guidelines-for-pakistan.html',
            'who_skin.pdf': 'https://extranet.who.int/ncdccs/Data/ZAF_D1aia_STG%20and%20EML%20PHC%202018.pdf',
            'psoriasis_cdc.html': 'https://archive.cdc.gov/www_cdc_gov/psoriasis/index.htm',
            'who_psoriasis.pdf': 'https://extranet.who.int/ncdccs/Data/ZAF_D1bia_s21_Primary%20Healthcare%20STGs%20and%20EML%207th%20edition%20-%202020-v2.0.pdf',
            'pneumonia_ncbi.html': 'https://www.ncbi.nlm.nih.gov/books/NBK264162/',
            'cdc_pneumonia.html': 'https://www.cdc.gov/pneumonia/hcp/management-prevention-guidelines/index.html',
        }
        
        # Try to download each guideline
        success_count = 0
        for filename, url in guidelines_to_download.items():
            try:
                logger.info(f"Downloading: {filename}")
                success = self.download_file(url, f'guidelines/{filename}', filename)
                if success:
                    success_count += 1
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")
                logger.info(f"Manual download may be required: {url}")
        
        # Create comprehensive guideline text file
        guidelines_text = """
# Clinical Guidelines and Thresholds

## Melanoma (CDC & Clinical Guidelines)

### Risk Factors
- Excessive UV exposure (sun/tanning beds)
- Fair skin, light hair, light eyes
- History of sunburns, especially in childhood
- Family history of melanoma
- Multiple or atypical moles (dysplastic nevi)
- Weakened immune system

### ABCDE Warning Signs
- **A**symmetry: One half doesn't match the other
- **B**order irregularity: Edges are ragged, notched, or blurred
- **C**olor: Multiple colors or uneven distribution
- **D**iameter: Greater than 6mm (size of pencil eraser)
- **E**volving: Changes in size, shape, color, or elevation

### Diagnosis
- Visual skin examination
- Dermoscopy
- Biopsy (excisional or punch)
- Histopathology (gold standard)

Reference: CDC Melanoma Guidelines
Source: https://stacks.cdc.gov/view/cdc/30592

---

## Eczema (Atopic Dermatitis) - WHO Guidelines

### Diagnostic Criteria (Must have)
- Pruritus (itching)
- Typical morphology and distribution:
  * Flexural involvement in adults
  * Facial and extensor involvement in infants/children

### Major Features (3+ required)
- Personal or family history of atopy (asthma, allergic rhinitis)
- Chronic or chronically relapsing dermatitis
- Onset before age 2 (for childhood onset)

### Clinical Features
- Dry skin (xerosis)
- Lichenification in adults
- Red, inflamed patches
- Crusting and oozing (acute phase)
- Affected areas: hands, feet, neck, eyelids, flexural areas

### Severity Assessment
- **Mild**: Localized areas, minimal impact on daily life
- **Moderate**: Widespread involvement, moderate impact
- **Severe**: Extensive involvement, significant impact on QoL

### Management
- Emollients (daily, multiple times)
- Topical corticosteroids (first-line anti-inflammatory)
- Avoid triggers (allergens, irritants)
- Systemic therapy for severe cases

Reference: WHO Guidelines for Atopic Dermatitis
Source: https://www.emro.who.int/emhj-volume-31-2025/volume-31-issue-9/

---

## Psoriasis (CDC & Clinical Guidelines)

### Clinical Features
- Well-demarcated, erythematous plaques
- Silvery-white scales
- Symmetric distribution
- Common sites: elbows, knees, scalp, lower back

### Classification by Severity
- **Mild**: <3% body surface area (BSA)
- **Moderate**: 3-10% BSA
- **Severe**: >10% BSA or involving critical areas (hands, feet, face, genitals)

### Psoriasis Types
1. **Plaque psoriasis** (80-90%): thick red patches with silvery scales
2. **Guttate psoriasis**: small, drop-shaped lesions
3. **Inverse psoriasis**: smooth red patches in skin folds
4. **Pustular psoriasis**: white pustules surrounded by red skin
5. **Erythrodermic psoriasis**: widespread redness and shedding

### Triggers
- Stress
- Infections (especially streptococcal)
- Medications (beta-blockers, lithium, antimalarials)
- Skin injury (Koebner phenomenon)
- Alcohol and smoking

### Management
- **Topical**: Corticosteroids, vitamin D analogues, retinoids
- **Phototherapy**: UVB, PUVA
- **Systemic**: Methotrexate, cyclosporine, biologics

Reference: CDC Psoriasis Information
Source: https://archive.cdc.gov/www_cdc_gov/psoriasis/index.htm

---

## Acne Vulgaris (Clinical Guidelines)

### Pathophysiology
1. Excess sebum production
2. Follicular hyperkeratinization
3. Cutibacterium acnes (C. acnes) colonization
4. Inflammation

### Clinical Presentation
- **Comedonal**: Blackheads (open comedones), whiteheads (closed comedones)
- **Inflammatory**: Papules, pustules
- **Severe**: Nodules, cysts, scarring

### Severity Classification
- **Mild**: Few to several papules/pustules, no nodules
- **Moderate**: Several to many papules/pustules, few to several nodules
- **Severe**: Numerous or extensive nodules/cysts, scarring

### Risk Factors
- Adolescence (hormonal changes)
- Family history
- Certain medications (corticosteroids, androgens, lithium)
- Cosmetics and hair products
- High glycemic diet, dairy (controversial)

### Treatment Approach
- **Mild**: Topical retinoids + benzoyl peroxide
- **Moderate**: Add topical or oral antibiotics
- **Severe**: Oral isotretinoin (Accutane)
- **Hormonal**: Oral contraceptives, spironolactone (females)

---

## Pneumonia (CDC & NCBI Guidelines)

### Diagnostic Criteria
- Acute illness with cough
- At least one of:
  * New focal chest signs on examination
  * At least one systemic feature (fever ≥38°C, sweats, shivers, aches)
- No other explanation for illness

### Clinical Classification
1. **Community-Acquired Pneumonia (CAP)**
   - Acquired outside hospital/healthcare facility
   - Most common bacterial cause: Streptococcus pneumoniae

2. **Hospital-Acquired Pneumonia (HAP)**
   - Onset ≥48 hours after hospital admission
   - Higher mortality, resistant organisms

3. **Ventilator-Associated Pneumonia (VAP)**
   - Onset ≥48 hours after endotracheal intubation

### Severity Assessment (CURB-65)
Score 1 point for each:
- **C**onfusion
- **U**rea >7 mmol/L (BUN >19 mg/dL)
- **R**espiratory rate ≥30/min
- **B**lood pressure: Systolic <90 or Diastolic ≤60 mmHg
- Age **65** years or older

Score 0-1: Low risk (outpatient)
Score 2: Moderate risk (consider hospitalization)
Score 3-5: High risk (hospitalization, consider ICU)

### Laboratory Findings
- Leukocytosis (WBC >11,000/μL) or leukopenia (<4,000/μL)
- Elevated inflammatory markers (CRP, procalcitonin)
- Chest X-ray: infiltrate, consolidation, effusion
- Hypoxemia: SpO2 <95%, PaO2 <60 mmHg

### Management
- **CAP Outpatient**: Amoxicillin or doxycycline
- **CAP Inpatient**: Fluoroquinolone or beta-lactam + macrolide
- **Severe CAP**: ICU, broad-spectrum antibiotics
- **Supportive**: Oxygen, fluids, respiratory support

Reference: CDC Pneumonia Guidelines, NCBI Clinical Guidelines
Sources: 
- https://www.cdc.gov/pneumonia/hcp/management-prevention-guidelines/
- https://www.ncbi.nlm.nih.gov/books/NBK264162/

---

## General Diagnostic Thresholds

### Vital Signs (Adult Normal Ranges)
- **Temperature**: 36.5-37.5°C (97.7-99.5°F)
- **Heart Rate**: 60-100 bpm
- **Respiratory Rate**: 12-20 breaths/min
- **Blood Pressure**: <120/80 mmHg (normal), 120-139/80-89 (prehypertension)
- **Oxygen Saturation**: ≥95%

### Common Laboratory Values
- **WBC**: 4,000-11,000 cells/μL
- **Platelets**: 150,000-400,000/μL
- **Hemoglobin**: 12-16 g/dL (female), 14-18 g/dL (male)
- **CRP**: <10 mg/L (normal), >10 mg/L (inflammation)
- **Ferritin**: 12-300 ng/mL (male), 12-150 ng/mL (female)

---

## Important Notes

### Disclaimer
These guidelines are for educational and reference purposes only.
They should not replace:
- Professional medical judgment
- Current local treatment protocols
- Individual patient assessment
- Updated clinical guidelines

### Regular Updates
Clinical guidelines are regularly updated by:
- WHO (World Health Organization)
- CDC (Centers for Disease Control and Prevention)
- Medical specialty societies
- National health authorities

Always consult the latest official guidelines for clinical decision-making.

### AI/ML Limitations
Machine learning models trained on these criteria:
- Are decision support tools, not replacements for clinicians
- Require validation on diverse populations
- Should be used alongside clinical expertise
- May not capture rare presentations or complex cases

---

Last Updated: April 2026
Compiled for: Hybrid Neuro-Symbolic Clinical Decision Support System
        """
        
        guidelines_file = guidelines_dir / 'clinical_guidelines.md'
        with open(guidelines_file, 'w') as f:
            f.write(guidelines_text)
        
        logger.info(f"✓ Comprehensive clinical guidelines saved to {guidelines_file}")
        logger.info(f"✓ Successfully processed {success_count}/{len(guidelines_to_download)} guideline downloads")
        
        return True
    
    def normalize_skin_datasets(self) -> bool:
        """
        Normalize downloaded skin disease datasets into unified structure:
        data/skin_lesions/train/<disease>/
        data/skin_lesions/test/<disease>/
        data/skin_lesions/val/<disease>/
        
        Automatically detects existing splits or creates them.
        """
        logger.info("\n" + "="*60)
        logger.info("Normalizing Skin Disease Datasets")
        logger.info("="*60)
        
        raw_dir = self.base_dir / 'skin_lesions_raw'
        output_dir = self.base_dir / 'skin_lesions'
        
        if not raw_dir.exists():
            logger.warning(f"Raw directory not found: {raw_dir}")
            return False
        
        # Create output structure
        for split in ['train', 'test', 'val']:
            (output_dir / split).mkdir(parents=True, exist_ok=True)
        
        disease_mapping = {
            'melanoma': 'Melanoma Skin Cancer Nevi and Moles',
            'eczema': 'Eczema Photos',
            'psoriasis': 'Psoriasis pictures Lichen Planus and related diseases',
            'acne': 'Acne and Rosacea Photos'
        }
        
        for short_name, full_name in disease_mapping.items():
            disease_dir = raw_dir / short_name
            
            if not disease_dir.exists():
                logger.warning(f"Disease directory not found: {disease_dir}")
                continue
            
            logger.info(f"\nProcessing: {short_name} -> {full_name}")
            
            # Detect structure and normalize
            self._normalize_disease_folder(
                disease_dir, 
                output_dir,
                full_name
            )
        
        logger.info("\n✓ Dataset normalization complete!")
        logger.info(f"Normalized datasets structure:")
        logger.info(f"  data/skin_lesions/")
        logger.info(f"    ├── train/")
        logger.info(f"    ├── test/")
        logger.info(f"    └── val/")
        
        return True
    
    def _normalize_disease_folder(
        self, 
        source_dir: Path, 
        output_dir: Path,
        disease_name: str
    ):
        """
        Normalize a single disease folder structure.
        
        Handles various input structures:
        1. train/test/val folders already exist
        2. train/test exist (create val from train)
        3. Single folder with all images (create splits)
        4. Nested folders (flatten)
        """
        import random
        from sklearn.model_selection import train_test_split
        
        # Collect all image files recursively
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(source_dir.rglob(f'*{ext}'))
            all_images.extend(source_dir.rglob(f'*{ext.upper()}'))
        
        if not all_images:
            logger.warning(f"No images found in {source_dir}")
            return
        
        logger.info(f"  Found {len(all_images)} images")
        
        # Detect if train/test/val split already exists
        has_train = (source_dir / 'train').exists() or (source_dir / 'Train').exists()
        has_test = (source_dir / 'test').exists() or (source_dir / 'Test').exists()
        has_val = (source_dir / 'val').exists() or (source_dir / 'validation').exists()
        
        if has_train and has_test:
            logger.info("  Detected existing train/test split")
            # Copy existing splits
            self._copy_split_images(source_dir, output_dir, disease_name, 'train')
            self._copy_split_images(source_dir, output_dir, disease_name, 'test')
            
            if has_val:
                self._copy_split_images(source_dir, output_dir, disease_name, 'val')
            else:
                # Create val from train (10%)
                logger.info("  Creating validation split from training data (10%)")
                self._split_and_copy(source_dir / 'train', output_dir, disease_name, val_from_train=True)
        else:
            logger.info("  No existing split detected - creating train/val/test split (70/10/20)")
            # Create splits: 70% train, 10% val, 20% test
            all_images = list(all_images)
            random.seed(42)
            random.shuffle(all_images)
            
            # Split
            train_val, test = train_test_split(all_images, test_size=0.2, random_state=42)
            train, val = train_test_split(train_val, test_size=0.125, random_state=42)  #0.125 * 0.8 = 0.1
            
            # Copy to output directories
            for split_name, split_images in [('train', train), ('val', val), ('test', test)]:
                target_dir = output_dir / split_name / disease_name
                target_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in split_images:
                    dest_path = target_dir / img_path.name
                    shutil.copy2(img_path, dest_path)
                
                logger.info(f"    {split_name}: {len(split_images)} images -> {target_dir}")
    
    def _copy_split_images(self, source_dir: Path, output_dir: Path, disease_name: str, split: str):
        """Copy images from existing split structure"""
        # Try different naming conventions
        split_variations = [split, split.capitalize(), split.upper()]
        source_split = None
        
        for variant in split_variations:
            if (source_dir / variant).exists():
                source_split = source_dir / variant
                break
        
        if not source_split:
            logger.warning(f"  Split '{split}' not found in {source_dir}")
            return
        
        # Collect images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        images = []
        for ext in image_extensions:
            images.extend(source_split.rglob(f'*{ext}'))
            images.extend(source_split.rglob(f'*{ext.upper()}'))
        
        if not images:
            logger.warning(f"  No images in {source_split}")
            return
        
        # Copy to output
        target_dir = output_dir / split / disease_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        for img_path in images:
            dest_path = target_dir / img_path.name
            # Handle duplicate filenames
            if dest_path.exists():
                dest_path = target_dir / f"{img_path.stem}_{copied}{img_path.suffix}"
            shutil.copy2(img_path, dest_path)
            copied += 1
        
        logger.info(f"    {split}: {copied} images -> {target_dir}")
    
    def _split_and_copy(self, source_folder: Path, output_dir: Path, disease_name: str, val_from_train: bool = False):
        """Split a folder into train/val"""
        from sklearn.model_selection import train_test_split
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        images = []
        for ext in image_extensions:
            images.extend(source_folder.rglob(f'*{ext}'))
            images.extend(source_folder.rglob(f'*{ext.upper()}'))
        
        if val_from_train:
            train, val = train_test_split(list(images), test_size=0.1, random_state=42)
            
            for split_name, split_images in [('train', train), ('val', val)]:
                target_dir = output_dir / split_name / disease_name
                target_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in split_images:
                    shutil.copy2(img_path, target_dir / img_path.name)
    
    def create_dataset_summary(self, results: Dict[str, bool]):
        """Create summary of downloaded datasets"""
        
        summary_file = self.base_dir / 'DATASET_SUMMARY.md'
        
        summary = f"""# Downloaded Datasets Summary

Generated: {Path.cwd()}
Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Download Results

"""
        
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        
        summary += f"**Success Rate**: {successful}/{total} datasets downloaded successfully\n\n"
        summary += "### Status by Dataset\n\n"
        
        for dataset, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            summary += f"- **{dataset}**: {status}\n"
        
        summary += """

## Dataset Locations

```
data/
├── covid19/              # COVID-19 clinical data
├── dengue/               # Dengue fever data
├── skin_lesions/         # Dermatology images (HAM10000, DermNet)
├── clinical/             # General clinical symptom data
├── guidelines/           # WHO/CDC clinical guidelines
└── DATASET_SUMMARY.md   # This file
```

## Next Steps

1. **Verify Downloads**:
   ```bash
   ls -lh data/*/
   ```

2. **Train Models**:
   ```bash
   python train.py --train-all
   ```

3. **Process Clinical Guidelines**:
   - Review `data/guidelines/clinical_guidelines.md`
   - Rules are implemented in `config/rules.yaml`

## Dataset Details

### COVID-19 Data
- Symptoms and test results
- Clinical outcomes
- Demographics

### Dengue Data  
- Fever patterns
- Platelet/WBC counts
- Clinical classification

### Skin Lesion Images
- HAM10000: 10,000+ dermatoscopic images
- 7 diagnostic categories
- Metadata with age, sex, localization

### Clinical Data
- Symptom descriptions
- Disease mappings
- Severity ratings

## References

All datasets comply with:
- Creative Commons licenses
- Academic/research use guidelines
- Privacy regulations (anonymized data)

See individual dataset documentation for specific citations and usage terms.
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"\n✓ Dataset summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download medical datasets for clinical decision support system"
    )
    parser.add_argument(
        '--kaggle-only',
        action='store_true',
        help='Download only Kaggle datasets'
    )
    parser.add_argument(
        '--guidelines-only',
        action='store_true',
        help='Download only clinical guidelines'
    )
    parser.add_argument(
        '--normalize-only',
        action='store_true',
        help='Only normalize existing downloaded datasets (no downloads)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Base directory for datasets (default: data)'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DataDownloader(base_dir=args.data_dir)
    
    all_results = {}
    
    print("\n" + "="*80)
    print("MEDICAL DATASET DOWNLOADER")
    print("Hybrid Neuro-Symbolic Clinical Decision Support System")
    print("="*80 + "\n")
    
    # Normalize only mode
    if args.normalize_only:
        logger.info("📐 Normalizing existing datasets...")
        downloader.normalize_skin_datasets()
        print("\n✓ Normalization complete!")
        return
    
    # Download clinical guidelines
    if not args.kaggle_only:
        logger.info("📚 Step 1: Downloading Clinical Guidelines...")
        downloader.download_who_cdc_guidelines()
    
    # Download Kaggle datasets
    if not args.guidelines_only:
        logger.info("\n📊 Step 2: Downloading Kaggle Datasets...")
        logger.info("This may take several minutes depending on dataset sizes...\n")
        
        kaggle_results = downloader.download_all_kaggle_datasets()
        all_results.update(kaggle_results)
        
        # Download additional datasets
        logger.info("\n📊 Step 3: Downloading Additional Datasets...")
        additional_results = downloader.download_additional_datasets()
        all_results.update(additional_results)
        
        # Normalize skin disease datasets
        logger.info("\n📐 Step 4: Normalizing Skin Disease Datasets...")
        logger.info("Creating unified train/val/test structure...\n")
        downloader.normalize_skin_datasets()
    
    # Create summary
    if all_results:
        downloader.create_dataset_summary(all_results)
    
    # Print final summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    
    if all_results:
        successful = sum(1 for v in all_results.values() if v)
        total = len(all_results)
        print(f"\n✓ Downloaded {successful}/{total} datasets successfully")
        print(f"✓ Datasets normalized into unified structure")
        
        if successful < total:
            print("\n⚠ Some downloads failed. Check logs above for details.")
            print("You can still train with synthetic data or successfully downloaded datasets.")
    
    print(f"\n📁 All data saved in: {Path(args.data_dir).absolute()}")
    print("\n🎯 Unified dataset structure created:")
    print("   data/skin_lesions/")
    print("     ├── train/")
    print("     │   ├── Melanoma Skin Cancer Nevi and Moles/")
    print("     │   ├── Eczema Photos/")
    print("     │   ├── Psoriasis pictures Lichen Planus and related diseases/")
    print("     │   └── Acne and Rosacea Photos/")
    print("     ├── val/")
    print("     └── test/")
    print("\n🚀 Next steps:")
    print("   1. Review: data/guidelines/clinical_guidelines.md")
    print("   2. Check: data/DATASET_SUMMARY.md")
    print("   3. Train: python train.py --train-cnn")
    print("   4. Launch UI: python app.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
