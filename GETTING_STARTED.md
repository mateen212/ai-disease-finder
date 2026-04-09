# Getting Started - Quick Reference

## Installation (5 minutes)

```bash
# 1. Clone/Navigate to project
cd /path/to/vspython

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python quickstart.py
```

## Quick Start (10 minutes)

### Option 1: Run Demo (Fastest)

```bash
# Train with synthetic data and run demo
python train.py --train-all
python main.py --demo
```

### Option 2: Diagnose Specific Cases

```bash
# Train models
python train.py --train-all

# Diagnose dengue case
python main.py --patient-data examples/dengue_patient.json

# Diagnose COVID-19 case
python main.py --patient-data examples/covid19_patient.json

# Diagnose pneumonia case
python main.py --patient-data examples/pneumonia_patient.json
```

## System Capabilities

✅ **Multi-Disease Diagnosis**
- Dengue fever
- COVID-19
- Pneumonia
- Skin disorders (melanoma, eczema, psoriasis, acne)

✅ **Hybrid AI Approach**
- Rule-based reasoning (symbolic AI)
- Random Forest classifier (machine learning)
- CNN for image classification (deep learning)
- Neuro-symbolic fusion

✅ **Full Explainability**
- SHAP feature importance
- Rule trace explanations
- Component-wise contributions
- Clinical recommendations

## Project Structure

```
vspython/
├── src/                          # Core source code
│   ├── rule_engine.py           # Forward-chaining inference
│   ├── ml_models.py             # Random Forest + CNN
│   ├── fusion.py                # Neuro-symbolic fusion
│   ├── explainability.py        # SHAP + explanations
│   ├── data_preprocessing.py    # Data handling
│   └── evaluation.py            # Model evaluation
├── config/                       # Configuration files
│   ├── rules.yaml               # Medical diagnostic rules
│   └── model_config.yaml        # Model hyperparameters
├── examples/                     # Sample patient data
│   ├── dengue_patient.json
│   ├── covid19_patient.json
│   └── pneumonia_patient.json
├── main.py                       # Main CLI application
├── train.py                      # Training script
├── quickstart.py                 # Setup guide and demo
├── requirements.txt              # Python dependencies
├── README.md                     # Full documentation
└── ARCHITECTURE.md              # Technical details
```

## Common Commands

```bash
# Training
python train.py --train-all              # Train everything
python train.py --test-rules             # Test rule engine only
python train.py --train-rf               # Train Random Forest only
python train.py --train-cnn              # Train CNN only (needs images)

# Diagnosis
python main.py --demo                    # Run demo
python main.py --patient-data FILE       # Diagnose from file
python main.py --patient-data FILE --output results.json  # Save results

# Help
python main.py --help                    # Show all options
python train.py --help                   # Show training options
python quickstart.py                     # Run setup guide
```

## Creating Custom Patient Files

Create a JSON file with this structure:

```json
{
  "symptoms": {
    "fever": true,
    "headache": true,
    "cough": false,
    "rash": true,
    ...
  },
  "vitals": {
    "temperature": 39.5,
    "oxygen_saturation": 96,
    ...
  },
  "labs": {
    "platelet_count": 95000,
    "wbc_count": 3200,
    ...
  },
  "demographics": {
    "age": 35,
    "gender": "M",
    ...
  }
}
```

See `examples/` directory for complete examples.

## Understanding Output

The system provides:

1. **Primary Diagnosis**
   ```
   Primary Diagnosis: DENGUE
   Confidence: 82%
   Risk Level: HIGH
   ```

2. **All Disease Scores**
   ```
   dengue      ████████████████████████████████░░░░░░░░ 82%
   covid19     ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 12%
   pneumonia   ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  6%
   ```

3. **Component Contributions**
   ```
   Rule-Based: dengue=70%
   Random Forest: dengue=75%
   Fusion: dengue=82%
   ```

4. **Explanations**
   ```
   Key factors:
   1. Platelet Count (95,000): ↓ decreases (lab finding)
   2. Fever (39.5°C): ↑ increases (vital sign)
   3. Rash (present): ↑ increases (symptom)
   
   Rules fired:
   ✓ Dengue_Classic: fever + headache + rash
   ✓ Dengue_Lab_Confirmed: low platelets + low WBC
   ```

5. **Recommendations**
   ```
   ⚠️ Seek medical evaluation within 24 hours
   • Monitor platelet count daily
   • Ensure adequate hydration
   • Watch for warning signs
   ```

## Customization

### Adjust Fusion Weights

Edit `config/model_config.yaml`:

```yaml
fusion:
  weights:
    rule_based: 0.3      # Change these
    random_forest: 0.5
    cnn: 0.2
```

### Add New Rules

Edit `config/rules.yaml`:

```yaml
dengue_rules:
  - name: "My_Custom_Rule"
    description: "Custom dengue criteria"
    conditions:
      all:
        - symptom: "fever"
          operator: "=="
          value: true
    conclusion:
      disease: "dengue"
      probability_boost: 0.3
```

### Modify Model Parameters

Edit `config/model_config.yaml`:

```yaml
random_forest:
  n_estimators: 300      # More trees
  max_depth: 25          # Deeper trees
```

## Troubleshooting

**Problem**: `SHAP not available`
```bash
pip install shap
```

**Problem**: `Kaggle API error`
```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Problem**: `Model not found`
```bash
# Train the models first
python train.py --train-all
```

**Problem**: `PyTorch/CUDA errors`
```bash
# CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Important Notes

⚠️ **MEDICAL DISCLAIMER**: This system is for educational/research purposes ONLY. Do NOT use for actual medical diagnosis. Always consult healthcare professionals.

🔒 **Privacy**: Patient data should be anonymized and handled according to regulations (HIPAA, GDPR, etc.)

📊 **Accuracy**: System accuracy depends on training data quality. Validate thoroughly before any use.

## Next Steps

1. ✅ Run the demo: `python main.py --demo`
2. ✅ Try example cases in `examples/`
3. ✅ Read `ARCHITECTURE.md` for technical details
4. ✅ Customize rules and parameters
5. ✅ Train on your own data (if available)

## Getting Help

- **Documentation**: See README.md and ARCHITECTURE.md
- **Examples**: Check `examples/` directory
- **Configuration**: Review `config/` files
- **Code**: All source in `src/` with docstrings

## Performance Tips

💡 **Fast Testing**: Use `--demo` mode with synthetic data

💡 **GPU Training**: For CNN, use GPU if available:
```python
# Will automatically use CUDA if available
cnn_model = SkinLesionCNN()
print(f"Using: {cnn_model.device}")
```

💡 **Parallel Processing**: Random Forest uses all CPU cores by default (`n_jobs=-1`)

💡 **Memory**: For large datasets, process in batches

## Citation

If you use this system in research, please cite:

```
Hybrid Neuro-Symbolic Clinical Decision Support System (2026)
Based on WHO/CDC guidelines and contemporary AI methods
```

---

**Ready to start?**

```bash
python train.py --train-all && python main.py --demo
```

Enjoy exploring the hybrid neuro-symbolic approach to clinical decision support! 🏥🤖
