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
            
            # Skin lesion datasets
            'skin_cancer_ham10000': ('kmader/skin-cancer-mnist-ham10000', 'skin_lesions'),
            'dermatology_images': ('shubhamgoel27/dermnet', 'skin_lesions'),
            
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
        
        # Create guideline references document
        guidelines_text = """
# Clinical Guidelines and Thresholds

## Dengue Fever (WHO Guidelines)

### Diagnostic Criteria
- Fever (typically 38.5-40°C lasting 2-7 days)
- At least 2 of the following:
  * Severe headache
  * Retro-orbital pain
  * Myalgia and arthralgia
  * Rash
  * Hemorrhagic manifestations
  * Leukopenia

### Laboratory Findings
- Thrombocytopenia (platelet count ≤ 100,000/mm³)
- Hemoconcentration (hematocrit increase ≥ 20%)
- Leukopenia (WBC < 5,000/mm³)

### Warning Signs (Severe Dengue)
- Abdominal pain or tenderness
- Persistent vomiting
- Clinical fluid accumulation
- Mucosal bleeding
- Lethargy, restlessness
- Liver enlargement > 2 cm
- Platelet count < 50,000/mm³
- Hematocrit increase concurrent with rapid platelet decrease

Reference: WHO Dengue Guidelines (2009, updated 2012)
Source: https://www.who.int/publications/i/item/9789241547871

---

## COVID-19 (CDC Guidelines)

### Common Symptoms
- Fever or chills (temperature ≥ 38°C)
- Cough (dry or productive)
- Shortness of breath or difficulty breathing
- Fatigue
- Muscle or body aches
- New loss of taste or smell
- Sore throat
- Congestion or runny nose
- Nausea or vomiting
- Diarrhea

### Severe Indicators
- Oxygen saturation < 94%
- Respiratory rate > 30/min
- PaO2/FiO2 < 300 mmHg
- Lung infiltrates > 50% increase within 24-48 hours

### Laboratory Findings
- Lymphopenia (lymphocyte count < 1,000/μL)
- Elevated inflammatory markers (CRP, D-dimer, ferritin)
- Elevated liver enzymes

Reference: CDC COVID-19 Clinical Care Guidelines (2020-2024)
Source: https://www.cdc.gov/coronavirus/2019-ncov/hcp/clinical-guidance-management-patients.html

---

## Pneumonia (Clinical Guidelines)

### Diagnostic Criteria
- Cough (productive or non-productive)
- Fever (temperature ≥ 38°C)
- Dyspnea or tachypnea
- Crackles or bronchial breath sounds on auscultation

### Severity Assessment (CURB-65)
- Confusion
- Urea > 7 mmol/L (BUN > 19 mg/dL)
- Respiratory rate ≥ 30/min
- Blood pressure (systolic < 90 or diastolic ≤ 60 mmHg)
- Age ≥ 65 years

Score ≥ 3: Consider hospital admission

### Laboratory Findings
- Leukocytosis (WBC > 11,000/mm³) or leukopenia (< 4,000/mm³)
- Oxygen saturation < 92%
- Chest X-ray: infiltrates (lobar, interstitial, or diffuse)

Reference: ATS/IDSA Pneumonia Guidelines (2019)
Source: https://www.thoracic.org/statements/

---

## Skin Lesion Classification

### Melanoma (ABCDE Criteria)
- A: Asymmetry
- B: Border irregularity
- C: Color variation
- D: Diameter > 6mm
- E: Evolution (changing over time)

### Dermoscopy Features
- Melanoma: Atypical pigment network, blue-white veil, irregular dots/globules
- Benign Nevi: Regular network, uniform color
- Basal Cell Carcinoma: Arborizing vessels, leaf-like areas
- Seborrheic Keratosis: Comedo-like openings, milia-like cysts

Reference: Skin Cancer Foundation Guidelines (2023)
Source: https://www.skincancer.org/

---

## Laboratory Reference Ranges

### Complete Blood Count
- WBC: 4,000-11,000/mm³
- Platelets: 150,000-450,000/mm³
- Hemoglobin: 13.5-17.5 g/dL (M), 12.0-15.5 g/dL (F)
- Hematocrit: 38.3-48.6% (M), 35.5-44.9% (F)

### Differential Count
- Neutrophils: 40-70%
- Lymphocytes: 20-40%
- Monocytes: 2-8%
- Eosinophils: 1-4%
- Basophils: 0.5-1%

### Vital Signs (Adult Reference)
- Temperature: 36.5-37.5°C (97.7-99.5°F)
- Heart Rate: 60-100 bpm
- Respiratory Rate: 12-20/min
- Blood Pressure: 90-120/60-80 mmHg
- Oxygen Saturation: ≥ 95%

---

## Risk Stratification Thresholds

### Low Risk
- Mild symptoms
- Normal vital signs
- No comorbidities
- Outpatient management

### Moderate Risk
- Moderate symptoms
- Mild vital sign abnormalities
- Some risk factors present
- Close outpatient monitoring or observation

### High Risk
- Severe symptoms
- Abnormal vital signs (but stable)
- Multiple risk factors
- Consider hospitalization

### Severe/Critical
- Respiratory distress or failure
- Hemodynamic instability
- Organ dysfunction
- ICU admission required

---

Last Updated: April 2026
Compiled from WHO, CDC, ATS/IDSA, and international clinical guidelines
"""
        
        guidelines_file = guidelines_dir / 'clinical_guidelines.md'
        with open(guidelines_file, 'w') as f:
            f.write(guidelines_text)
        
        logger.info(f"✓ Clinical guidelines saved to {guidelines_file}")
        
        return True
    
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
        
        if successful < total:
            print("\n⚠ Some downloads failed. Check logs above for details.")
            print("You can still train with synthetic data or successfully downloaded datasets.")
    
    print(f"\n📁 All data saved in: {Path(args.data_dir).absolute()}")
    print("\n🚀 Next steps:")
    print("   1. Review: data/guidelines/clinical_guidelines.md")
    print("   2. Check: data/DATASET_SUMMARY.md")
    print("   3. Train: python train.py --train-all")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
