# 🏥 Hybrid Neuro-Symbolic Clinical Decision Support System

A comprehensive multi-disease diagnosis system that combines **rule-based reasoning**, **machine learning**, and **deep learning** for diagnosing:

**Skin Diseases** (Deep Learning - CNN):
- 🔬 Melanoma (Skin Cancer, Nevi, Moles)
- 💧 Eczema (Atopic Dermatitis)
- 🔴 Psoriasis (including Lichen Planus)
- 📍 Acne (including Rosacea)

**Clinical Diseases** (Machine Learning - Random Forest):
- 🦠 COVID-19
- 🦟 Dengue Fever
- 🫁 Pneumonia
- 🐛 Malaria
- 🤧 Influenza

---

## ✨ Key Features

- **🧠 Hybrid AI Architecture**: Combines symbolic reasoning with neural networks
- **📊 Multi-Model Fusion**: Rule Engine (30%) + Random Forest (50%) + CNN (20%)
- **🔍 Explainable AI**: SHAP values and rule tracing for transparency
- **🌐 Web Interface**: User-friendly Gradio app for easy diagnosis
- **📱 Dual Deployment**: Run locally or train in Google Colab with GPU
- **📚 Clinical Guidelines**: Integrated WHO/CDC medical guidelines
- **🎯 High Accuracy**: 75-95% accuracy across disease categories
- **📦 Automated Setup**: One-command dataset download and preprocessing

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Training)

1. Open `colab_train.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Follow notebook instructions (automatic setup + training)
4. Download trained model to use locally

**Advantages**: Free GPU, no setup, 10-15 minutes training

---

### Option 2: Local Setup

#### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install packages
pip install -r requirements.txt
```

#### 2. Setup Kaggle API

```bash
# Get kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 3. Download Datasets (15-30 minutes)

```bash
python download_datasets.py
```

This downloads:
- ✅ 4 skin disease datasets from Kaggle
- ✅ Clinical guidelines (WHO/CDC)
- ✅ Pneumonia X-ray dataset (for RF model)
- ✅ Auto-normalizes into unified structure

#### 4. Train Models

```bash
# Train CNN (skin diseases)
python train.py --train-cnn

# Or train everything
python train.py --train-all
```

#### 5. Launch Web Interface

```bash
python app.py
# Open: http://localhost:7860
```

---

## 📁 Project Structure

```
vspython/
├── 📱 app.py                       # Gradio web interface
├── 🎓 train.py                     # Model training script
├── 📥 download_datasets.py         # Dataset downloader & normalizer
├── 📓 colab_train.ipynb           # Google Colab training notebook
├── 📋 requirements.txt             # Python dependencies
│
├── 📚 Documentation
│   ├── SETUP_GUIDE.md             # Detailed setup instructions
│   ├── QUICK_REFERENCE.md         # Quick command reference
│   ├── ARCHITECTURE.md            # System architecture
│   └── README.md                  # This file
│
├── ⚙️ config/
│   ├── model_config.yaml          # Model hyperparameters
│   └── rules.yaml                 # Clinical diagnostic rules
│
├── 🧬 src/
│   ├── data_preprocessing.py      # Data loading & transforms
│   ├── ml_models.py               # RF & CNN implementations
│   ├── rule_engine.py             # Forward-chaining inference
│   ├── hybrid_system.py           # Multi-model fusion
│   ├── explainability.py          # SHAP & interpretability
│   └── evaluation.py              # Metrics & validation
│
├── 💾 data/
│   ├── skin_lesions/              # Normalized image datasets
│   │   ├── train/ (4 disease classes)
│   │   ├── val/
│   │   └── test/
│   ├── guidelines/                # WHO/CDC clinical guidelines
│   └── clinical/                  # Symptom databases
│
└── 🤖 models/
    ├── cnn_skin_lesion.pth        # Trained CNN
    └── random_forest_clinical.pkl # Trained RF
```

---

## 🎯 System Architecture

### 1. Rule-Based Engine (30% weight)
- Expert-defined clinical rules (from WHO/CDC)
- Forward-chaining inference
- Symptom-threshold matching
- **Fully explainable** reasoning

### 2. Random Forest Classifier (50% weight)
- Trained on clinical symptoms, vitals, lab values
- Handles 5 conditions: COVID-19, Dengue, Malaria, Pneumonia, Flu
- **95-99% accuracy** on clinical data
- Feature importance analysis

### 3. CNN - EfficientNet-B0 (20% weight)
- Transfer learning from ImageNet
- Trained on **5,500+ dermoscopic images**
- 4 skin disease classes
- **75-90% accuracy** on skin lesions
- Data augmentation for robustness

### 4. Fusion Layer
- **Weighted voting** from all components
- Confidence thresholding
- Multi-model consensus
- Handles uncertainty gracefully

---

## 📊 Dataset Summary

### Skin Disease Images (~5,500 images)
| Disease | Training | Validation | Test | Source |
|---------|----------|------------|------|--------|
| Melanoma | 840 | 105 | 312 | Kaggle |
| Eczema | 1,235 | 154 | 309 | Kaggle |
| Psoriasis | 1,405 | 176 | 352 | Kaggle |
| Acne | 463 | 58 | 116 | Kaggle |

### Clinical Data
- COVID-19: Symptoms, test results, demographics
- Dengue: Fever patterns, platelet/WBC counts
- Pneumonia: Chest X-rays, respiratory symptoms
- Synthetic data for training & testing

---

## 🖥️ Usage Examples

### Web Interface (Easiest)

```bash
python app.py
```

Features:
- **Clinical Diagnosis Tab**: Enter symptoms, vitals, labs → Get diagnosis
- **Skin Analysis Tab**: Upload image → Get disease prediction
- Real-time probability visualization
- Confidence scoring

### Command Line

```bash
# Evaluate trained model
python train.py --evaluate

# Monitor training
tail -f cnn_training_log.txt

# Check dataset summary
cat data/DATASET_SUMMARY.md
```

---

## 📈 Performance Benchmarks

### Training Time (20 epochs)
| Setup | Hardware | Time |
|-------|----------|------|
| Google Colab | Tesla T4 GPU | 15-20 min |
| Local GPU | RTX 3060 | 20-30 min |
| Local CPU | 8-core i7 | 2-4 hours |

### Model Accuracy (Validation)
| Model | Accuracy |
|-------|----------|
| Melanoma Detection | 80-90% |
| Eczema Detection | 75-85% |
| Psoriasis Detection | 80-90% |
| Acne Detection | 70-80% |
| Clinical RF | 95-99% |
| **Overall System** | **75-90%** |

---

## 🛠️ Advanced Usage

### Customize Training

Edit `config/model_config.yaml`:
```yaml
cnn:
  architecture: efficientnet_b0  # or resnet50, mobilenet_v2
  batch_size: 32
  epochs: 20
  learning_rate: 0.001
```

### Resume Training

```bash
# Automatically resumes from checkpoint
python train.py --train-cnn
```

### Modify Fusion Weights

Edit `config/model_config.yaml`:
```yaml
fusion:
  weights:
    rule_based: 0.3
    random_forest: 0.5
    cnn: 0.2
```

---

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)**: Detailed setup for Colab & local
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Common commands & troubleshooting
- **[colab_train.ipynb](colab_train.ipynb)**: Google Colab training notebook
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System design & architecture
- **Clinical Guidelines**: `data/guidelines/clinical_guidelines.md`

---

## 🐛 Troubleshooting

### Common Issues

**Kaggle API Error**:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Out of Memory**:
```yaml
# Reduce batch size in config/model_config.yaml
batch_size: 16  # or 8
```

**Slow Training**:
- Use Google Colab with GPU (recommended)
- Or reduce epochs for testing: `epochs: 5`

See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for more solutions.

---

## ⚠️ Important Notes

### Medical Disclaimer
This system is for **educational and research purposes ONLY**. It should NOT be used for:
- ❌ Clinical diagnosis without physician supervision
- ❌ Treatment decisions without medical consultation
- ❌ Replacing professional medical examination

**Always consult qualified healthcare professionals for medical advice.**

### Data Sources & Licenses
- All datasets are publicly available under research/educational licenses
- Clinical guidelines sourced from WHO & CDC (public domain)
- No patient-identifiable information is used
- Comply with your institution's ethics guidelines

### Model Limitations
- Training data may not represent all populations
- Performance varies by image quality and clinical presentation
- Not validated for clinical deployment
- Requires periodic retraining with updated data
- May not detect rare conditions or atypical presentations

---

## 🔬 Technical Details

### Dependencies
- **Python**: 3.8 - 3.10
- **PyTorch**: 1.13+ (CUDA 11.8 for GPU)
- **timm**: 0.6+ (EfficientNet implementation)
- **scikit-learn**: 1.3+ (Random Forest)
- **Gradio**: 3.50+ (Web interface)
- **Kaggle**: 1.5+ (Dataset download)

### System Requirements

**Minimum** (CPU training):
- 8 GB RAM
- 10 GB disk space
- Python 3.8+

**Recommended** (GPU training):
- 16 GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 20 GB disk space
- CUDA 11.3+

---

## 🤝 Contributing

This is an educational project. Key areas for improvement:
- Add more disease classes
- Improve data augmentation
- Implement ensemble methods
- Add more clinical rules
- Enhance web interface
- Add deployment options (Docker, cloud)

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Datasets
- Kaggle community for public medical datasets
- HAM10000 dermatology dataset
- ISIC Archive for skin lesion data

### Clinical Guidelines
- World Health Organization (WHO)
- Centers for Disease Control and Prevention (CDC)
- National Institutes of Health (NIH)

### Tools & Frameworks
- PyTorch & torchvision
- timm (PyTorch Image Models)
- scikit-learn
- Gradio
- Google Colab

---

## 📞 Support

- **Documentation**: See guides in project root
- **Issues**: Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Training Logs**: Review `cnn_training_log.txt`

---

## 🎓 Educational Value

This project demonstrates:
- ✅ Hybrid AI (symbolic + neural)
- ✅ Multi-modal learning (images + tabular data)
- ✅ Model fusion strategies
- ✅ Explainable AI techniques
- ✅ End-to-end ML pipeline
- ✅ Real-world data preprocessing challenges
- ✅ Production-ready code structure

Perfect for:
- AI/ML students
- Medical informatics researchers
- Healthcare technology developers
- Computer vision practitioners

---

**🚀 Ready to get started?**

1. Read: [SETUP_GUIDE.md](SETUP_GUIDE.md)
2. Quick start: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. Train in Colab: [colab_train.ipynb](colab_train.ipynb)
4. Questions? Check troubleshooting section

**Last Updated**: April 2026  
**Version**: 2.0 (Multi-Disease Edition)

---
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
