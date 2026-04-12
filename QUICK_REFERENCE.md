# ⚡ Quick Reference Guide

## 🚀 Quick Start Commands

### First Time Setup (5 minutes)
```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Kaggle (place kaggle.json in ~/.kaggle/)
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Download & Prepare Data (15-30 minutes)
```bash
# Download all datasets and prepare structure
python download_datasets.py
```

### Train Models
```bash
# Train CNN only (skin diseases)
python train.py --train-cnn

# Train Random Forest only (clinical symptoms)
python train.py --train-rf

# Train everything
python train.py --train-all
```

### Launch Web Interface
```bash
python app.py
# Open: http://localhost:7860
```

---

## 📁 Project Structure

```
vspython/
├── app.py                      # Gradio web interface
├── train.py                    # Training script
├── download_datasets.py        # Dataset downloader
├── requirements.txt            # Python dependencies
├── SETUP_GUIDE.md             # Detailed setup instructions
├── colab_train.ipynb          # Google Colab notebook
│
├── config/
│   ├── model_config.yaml      # Model hyperparameters
│   └── rules.yaml             # Clinical rules
│
├── src/
│   ├── data_preprocessing.py  # Data loading & transforms
│   ├── ml_models.py           # RF & CNN models
│   ├── rule_engine.py         # Rule-based reasoning
│   ├── hybrid_system.py       # Model fusion
│   └── evaluation.py          # Metrics & evaluation
│
├── data/
│   ├── skin_lesions/          # Normalized image datasets
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── skin_lesions_raw/      # Downloaded raw data
│   ├── guidelines/            # Clinical guidelines (WHO/CDC)
│   └── clinical/              # Clinical symptom data
│
└── models/
    ├── cnn_skin_lesion.pth          # Trained CNN
    ├── cnn_skin_lesion_checkpoint.pth  # Training checkpoint
    └── random_forest_clinical.pkl   # Trained RF
```

---

## 🎯 Model Overview

### CNN (Skin Diseases)
- **Architecture**: EfficientNet-B0 (pretrained on ImageNet)
- **Input**: 224×224 RGB images
- **Output**: 4 classes
  - Melanoma
  - Eczema
  - Psoriasis
  - Acne
- **Accuracy**: 75-90% (depending on training)

### Random Forest (Clinical)
- **Features**: Symptoms, vitals, lab values
- **Output**: 5 diseases
  - COVID-19
  - Dengue
  - Pneumonia
  - Malaria
  - Influenza
- **Accuracy**: 95-99%

### Rule Engine
- Expert-defined clinical rules
- Forward chaining inference
- Explainable reasoning

### Fusion Strategy
- Weighted voting: RF(50%), Rules(30%), CNN(20%)
- Confidence thresholding
- Multi-model consensus

---

## 💡 Common Tasks

### Monitor Training Progress
```bash
# Watch training logs in real-time
tail -f cnn_training_log.txt
```

### Check GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Resume Training
```bash
# Training automatically resumes from checkpoint if available
python train.py --train-cnn

# To start fresh, backup old checkpoints:
mkdir -p models/backup
mv models/cnn_skin_lesion*.pth models/backup/
```

### Evaluate Model
```bash
# Evaluate trained models
python train.py --evaluate
```

### Verify Dataset Structure
```bash
# Check data was downloaded correctly
ls -R data/skin_lesions/train/

# Count images per class
find data/skin_lesions/train -type f -name "*.jpg" | wc -l

# View dataset summary
cat data/DATASET_SUMMARY.md
```

### Clean Up & Reset
```bash
# Remove downloaded raw data (keep normalized)
rm -rf data/skin_lesions_raw/

# Remove training logs
rm cnn_training_log*.txt

# Reset models (backup first!)
mkdir -p models/backup_$(date +%Y%m%d)
mv models/*.pth models/backup_$(date +%Y%m%d)/
```

---

## 🐛 Quick Troubleshooting

### Problem: "No module named 'torch'"
```bash
pip install torch torchvision timm
```

### Problem: "Kaggle API credentials not found"
```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Problem: "CUDA out of memory"
```bash
# Edit config/model_config.yaml
# Change: batch_size: 32  →  batch_size: 16
```

### Problem: "Training too slow on CPU"
```bash
# Use Google Colab with GPU
# Upload: colab_train.ipynb to Google Colab
```

### Problem: "Dataset not found"
```bash
# Re-run normalization
python download_datasets.py --normalize-only
```

---

## 📊 Training Parameters

### Current Settings (config/model_config.yaml)

```yaml
cnn:
  architecture: efficientnet_b0
  num_classes: 4
  batch_size: 32
  learning_rate: 0.001
  epochs: 20
  input_size: 224
```

### Recommended Adjustments

**For Faster Training (Testing)**:
```yaml
epochs: 5
batch_size: 32
```

**For Better Accuracy (Production)**:
```yaml
epochs: 50
batch_size: 16  # if GPU memory limited
learning_rate: 0.0005  # lower for fine-tuning
```

**For Low-Memory Systems**:
```yaml
batch_size: 8
num_workers: 1
```

---

## 🔗 Important URLs

### Dataset Sources
- Melanoma: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
- Eczema: https://www.kaggle.com/datasets/adityush/eczema2
- Psoriasis: https://www.kaggle.com/datasets/pallapurajkumar/psoriasis-skin-dataset
- Acne: https://www.kaggle.com/datasets/tiswan14/acne-dataset-image
- Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Clinical Guidelines
- WHO: https://www.who.int/
- CDC: https://www.cdc.gov/
- Guidelines file: `data/guidelines/clinical_guidelines.md`

### Tools
- Kaggle API: https://www.kaggle.com/settings
- Google Colab: https://colab.research.google.com/

---

## 📈 Expected Performance

### Training Time (20 epochs)
- **Google Colab (T4 GPU)**: 15-20 minutes
- **Local GPU (RTX 3060)**: 20-30 minutes
- **Local CPU (8-core)**: 2-4 hours

### Accuracy Targets
- **Melanoma**: 80-90%
- **Eczema**: 75-85%
- **Psoriasis**: 80-90%
- **Acne**: 70-80%
- **Overall**: 75-85%

### Dataset Sizes (after normalization)
- **Training**: ~3,900 images
- **Validation**: ~550 images
- **Test**: ~1,100 images
- **Total**: ~5,500 images across 4 classes

---

## 🎓 Learning Resources

### Understanding the Code

**Data Preprocessing**:
```python
# See: src/data_preprocessing.py
# Key functions:
# - prepare_folder_based_image_dataset()  # Load images from folders
# - train_transform, test_transform       # Image augmentation
```

**Model Training**:
```python
# See: src/ml_models.py
# Key classes:
# - SkinLesionCNN         # CNN wrapper
# - RandomForestDiagnostic  # RF wrapper
# - SkinLesionDataset     # PyTorch dataset
```

**Hybrid Fusion**:
```python
# See: src/hybrid_system.py
# Combines predictions from:
# - Rule engine (30% weight)
# - Random Forest (50% weight)
# - CNN (20% weight)
```

### Modifying the System

**Add New Disease Class**:
1. Download dataset
2. Place in `data/skin_lesions_raw/new_disease/`
3. Update `download_datasets.py` disease_mapping
4. Run normalization
5. Update `config/model_config.yaml` num_classes
6. Retrain

**Change Model Architecture**:
```yaml
# In config/model_config.yaml
cnn:
  architecture: resnet50  # or mobilenet_v2, efficientnet_b1
```

**Adjust Fusion Weights**:
```yaml
# In config/model_config.yaml
fusion:
  weights:
    rule_based: 0.2
    random_forest: 0.6
    cnn: 0.2
```

---

## ✅ Checklist

### Initial Setup
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Kaggle API configured (`~/.kaggle/kaggle.json`)

### Data Preparation
- [ ] Datasets downloaded (`python download_datasets.py`)
- [ ] Data structure verified (`ls data/skin_lesions/train/`)
- [ ] Dataset summary exists (`cat data/DATASET_SUMMARY.md`)

### Training
- [ ] CNN training completed (`python train.py --train-cnn`)
- [ ] Model saved (`ls models/cnn_skin_lesion.pth`)
- [ ] Training logs available (`cat cnn_training_log.txt`)

### Deployment
- [ ] Web interface launches (`python app.py`)
- [ ] Can upload images and get predictions
- [ ] Clinical diagnosis tab works

---

## 🚨 IMPORTANT: Pneumonia Dataset

The pneumonia chest X-ray dataset is used **ONLY** for the Random Forest model pipeline. 

**DO NOT**:
- ❌ Modify the pneumonia RF training code
- ❌ Remove pneumonia from clinical diseases
- ❌ Mix pneumonia images with skin disease training

**The system maintains TWO separate pipelines**:
1. **Skin diseases** → CNN (4 classes)
2. **Clinical symptoms** → Random Forest (5 diseases including pneumonia)

---

## 📞 Getting Help

1. **Read documentation**: SETUP_GUIDE.md, README.md
2. **Check logs**: Training logs show detailed errors
3. **Verify environment**: `pip list`, `python --version`
4. **Test components**: Run each part separately

---

**Last Updated**: April 2026
**Version**: 1.0

For detailed setup instructions, see: [SETUP_GUIDE.md](SETUP_GUIDE.md)
For Google Colab training, see: [colab_train.ipynb](colab_train.ipynb)
