# Hybrid Neuro-Symbolic Clinical Decision Support System

A comprehensive multi-disease diagnosis system that combines rule-based reasoning with machine learning for diagnosing dengue, COVID-19, pneumonia, and skin disorders (melanoma, eczema, psoriasis, acne).

## Features

- **Rule-Based Inference Engine**: Forward-chaining logic based on WHO/CDC guidelines
- **Random Forest Classifier**: For tabular clinical data (symptoms, vitals, labs)
- **CNN Image Classifier**: For skin lesion classification
- **Neuro-Symbolic Fusion**: Combines symbolic rules with neural predictions
- **SHAP Explainability**: Provides feature-level explanations for all predictions
- **Comprehensive Reports**: Disease probabilities, risk levels, and recommendations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Kaggle API credentials:
   - Download your `kaggle.json` from https://www.kaggle.com/settings
   - Place it in `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── rule_engine.py          # Forward-chaining inference engine
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── ml_models.py            # Random Forest and CNN models
│   ├── fusion.py               # Neuro-symbolic fusion logic
│   ├── explainability.py       # SHAP and rule explanations
│   └── evaluation.py           # Model evaluation utilities
├── config/
│   ├── rules.yaml              # Medical diagnostic rules
│   └── model_config.yaml       # Model hyperparameters
├── data/                       # Downloaded datasets (gitignored)
├── models/                     # Trained models (gitignored)
├── main.py                     # Main CLI application
├── train.py                    # Training script
└── requirements.txt
```

## Usage

### Training the System

```bash
python train.py --download-data --train-all
```

### Making Predictions

```bash
python main.py --patient-data patient.json --skin-image lesion.jpg
```

### Example Patient Input

```json
{
  "symptoms": {
    "fever": true,
    "headache": true,
    "cough": false,
    "rash": true,
    "nausea": true
  },
  "vitals": {
    "temperature": 39.5,
    "oxygen_saturation": 96,
    "heart_rate": 88
  },
  "labs": {
    "platelet_count": 95000,
    "wbc_count": 3200,
    "lymphocyte_percentage": 25
  },
  "demographics": {
    "age": 35,
    "gender": "M",
    "travel_history": true
  }
}
```

## Medical Knowledge Base

The system implements diagnostic criteria from:
- WHO Dengue Guidelines
- CDC COVID-19 and Dengue Clinical Features
- Standard pneumonia diagnostic criteria
- Dermatology classification standards

## Disclaimer

**This system is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare providers for medical decisions.**

## License

MIT License - See LICENSE file for details
