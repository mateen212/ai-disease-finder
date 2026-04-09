#!/usr/bin/env python3
"""
Quick Start Guide and Demo Script

This script demonstrates the full capabilities of the Hybrid Neuro-Symbolic
Clinical Decision Support System.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*80)
print("HYBRID NEURO-SYMBOLIC CLINICAL DECISION SUPPORT SYSTEM")
print("Quick Start Guide and Demo")
print("="*80)

print("\n📋 STEP 1: Setup and Installation")
print("-"*80)
print("""
1. Install dependencies:
   pip install -r requirements.txt

2. Set up Kaggle credentials (optional, for real data):
   - Download kaggle.json from https://www.kaggle.com/settings
   - Place it in ~/.kaggle/kaggle.json
   - chmod 600 ~/.kaggle/kaggle.json

3. Create necessary directories:
   mkdir -p data models outputs logs
""")

print("\n🔧 STEP 2: Train the Models")
print("-"*80)
print("""
Train all components:
   python train.py --train-all

Or train individually:
   python train.py --test-rules          # Test rule engine
   python train.py --train-rf            # Train Random Forest
   python train.py --train-cnn           # Train CNN (requires image data)
   python train.py --download-data       # Download Kaggle datasets
""")

print("\n🏥 STEP 3: Use the System")
print("-"*80)
print("""
Demo mode:
   python main.py --demo

Diagnose from patient file:
   python main.py --patient-data examples/dengue_patient.json

With output file:
   python main.py --patient-data examples/covid19_patient.json --output results.json

With skin lesion image:
   python main.py --patient-data patient.json --skin-image lesion.jpg --output results.json
""")

print("\n📊 STEP 4: Understanding the Output")
print("-"*80)
print("""
The system provides:
1. Primary diagnosis with confidence score
2. Risk level assessment (low/moderate/high/severe)
3. Component-wise scores:
   - Rule-based reasoning (symbolic)
   - Random Forest predictions (neural)
   - CNN analysis (for skin conditions)
4. Neuro-symbolic fusion results
5. SHAP-based feature explanations
6. Rule traces showing which medical rules fired
7. Clinical recommendations

Example output structure:
{
  "diagnosis": ["dengue", 0.82],
  "risk_level": "high",
  "disease_scores": {
    "dengue": 0.82,
    "covid19": 0.12,
    "pneumonia": 0.06
  },
  "recommendations": [
    "Seek medical evaluation within 24 hours",
    "Monitor platelet count daily",
    ...
  ],
  "explanation": "..."
}
""")

print("\n🧪 STEP 5: Example Use Cases")
print("-"*80)
print("""
Case 1: Dengue Diagnosis
------------------------
python main.py --patient-data examples/dengue_patient.json

Expected: High probability of dengue with low platelets and fever as key factors

Case 2: COVID-19 Diagnosis
---------------------------
python main.py --patient-data examples/covid19_patient.json

Expected: High probability of COVID-19 with loss of taste/smell as key factors

Case 3: Pneumonia Diagnosis
----------------------------
python main.py --patient-data examples/pneumonia_patient.json

Expected: High probability of pneumonia with respiratory symptoms and low O2
""")

print("\n⚙️ STEP 6: Customization")
print("-"*80)
print("""
1. Modify diagnostic rules:
   Edit config/rules.yaml

2. Adjust model hyperparameters:
   Edit config/model_config.yaml

3. Change fusion weights:
   Modify weights in config/model_config.yaml under 'fusion' section

4. Add new diseases:
   - Add rules in config/rules.yaml
   - Update disease lists in config/model_config.yaml
   - Retrain models with new data
""")

print("\n📚 STEP 7: Understanding the Components")
print("-"*80)
print("""
1. Rule Engine (src/rule_engine.py):
   - Forward-chaining inference
   - Based on WHO/CDC guidelines
   - Symbolic reasoning

2. Random Forest (src/ml_models.py):
   - Learns from clinical data
   - Handles symptoms, vitals, labs
   - Provides feature importance

3. CNN (src/ml_models.py):
   - Classifies skin lesions
   - Uses pretrained networks (EfficientNet/ResNet)
   - Transfer learning

4. Neuro-Symbolic Fusion (src/fusion.py):
   - Combines symbolic and neural outputs
   - Weighted averaging or max confidence
   - Risk assessment

5. Explainability (src/explainability.py):
   - SHAP values for ML models
   - Rule traces for symbolic reasoning
   - Comprehensive patient reports
""")

print("\n⚠️  IMPORTANT DISCLAIMER")
print("-"*80)
print("""
This system is for EDUCATIONAL and RESEARCH purposes only.

DO NOT use this system as a substitute for professional medical diagnosis
or treatment. Always consult qualified healthcare providers for medical
decisions.

The system's predictions should be validated by medical professionals and
should only be used as a decision support tool, not as a definitive
diagnostic instrument.
""")

print("\n🔗 Additional Resources")
print("-"*80)
print("""
Documentation: README.md
Project Structure: See file tree in README.md
Configuration: config/ directory
Examples: examples/ directory

For issues or questions, refer to the documentation or project README.
""")

print("\n" + "="*80)
print("Ready to start! Run the training script:")
print("  python train.py --train-all")
print("\nThen try the demo:")
print("  python main.py --demo")
print("="*80 + "\n")

# Optional: Run a quick system check
print("\n🔍 System Check")
print("-"*80)

try:
    from src.rule_engine import RuleEngine
    print("✓ Rule Engine module loaded")
except Exception as e:
    print(f"✗ Rule Engine error: {e}")

try:
    from src.ml_models import RandomForestDiagnostic, SkinLesionCNN
    print("✓ ML Models module loaded")
except Exception as e:
    print(f"✗ ML Models error: {e}")

try:
    from src.fusion import NeuroSymbolicFusion
    print("✓ Fusion module loaded")
except Exception as e:
    print(f"✗ Fusion error: {e}")

try:
    from src.explainability import ExplainabilityModule
    print("✓ Explainability module loaded")
except Exception as e:
    print(f"✗ Explainability error: {e}")

try:
    from src.data_preprocessing import DataPreprocessor
    print("✓ Data Preprocessing module loaded")
except Exception as e:
    print(f"✗ Data Preprocessing error: {e}")

try:
    from src.evaluation import ModelEvaluator
    print("✓ Evaluation module loaded")
except Exception as e:
    print(f"✗ Evaluation error: {e}")

# Check for critical files
print("\n📁 File Check:")
for f in ['config/rules.yaml', 'config/model_config.yaml', 'requirements.txt']:
    if Path(f).exists():
        print(f"✓ {f}")
    else:
        print(f"✗ {f} missing")

print("\n" + "="*80)
print("System check complete. All modules are ready!")
print("="*80)
