# 🚀 Complete Setup & Training Guide

This guide provides step-by-step instructions for setting up and training the Hybrid Neuro-Symbolic Clinical Decision Support System on **both Google Colab and local machines**.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Google Colab Setup (Recommended for GPU Training)](#google-colab-setup)
3. [Local System Setup](#local-system-setup)
4. [Dataset Download & Preparation](#dataset-download--preparation)
5. [Model Training](#model-training)
6. [Running the Web Interface](#running-the-web-interface)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 System Overview

### Components:
- **Rule-Based Engine**: Expert clinical rules (30% weight)
- **Random Forest**: Clinical symptom classifier (50% weight)
- **CNN (EfficientNet-B0)**: Skin disease image classifier (20% weight)

### Diseases Diagnosed:
**Skin Diseases (CNN)**:
- Melanoma
- Eczema
- Psoriasis
- Acne

**Clinical Diseases (RF)**:
- COVID-19
- Dengue
- Pneumonia
- Malaria
- Influenza

---

## 🌐 Google Colab Setup

**Best for:** GPU-accelerated training, no local setup required

### Steps:

1. **Open the Colab Notebook**
   ```
   Navigate to: colab_train.ipynb
   Upload to Google Colab
   ```

2. **Enable GPU**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU → Save
   ```

3. **Follow Notebook Steps**
   - The notebook handles everything automatically
   - Downloads datasets from Kaggle
   - Normalizes data structure
   - Trains the model
   - Saves to Google Drive

4. **Download Trained Model**
   ```
   After training, download: cnn_skin_lesion.pth
   Place in: models/cnn_skin_lesion.pth (in your local project)
   ```

### Advantages:
- ✅ Free GPU (Tesla T4 or better)
- ✅ No installation required
- ✅ ~10-15 minutes training for 10 epochs
- ✅ Automatic environment setup

### Requirements:
- Google account
- Kaggle API key (kaggle.json)

---

## 💻 Local System Setup

**Best for:** Development, inference, full project control

### System Requirements:

#### Minimum (CPU Training):
- **OS**: Linux, macOS, or Windows 10+
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.8 - 3.10

#### Recommended (GPU Training):
- **RAM**: 16 GB+
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **CUDA**: 11.3 or higher
- **Storage**: 20 GB free space

---

### 1. Clone/Setup Project

```bash
# If you have a Git repository:
git clone <your-repo-url>
cd vspython

# Or create project directory:
mkdir clinical_diagnosis_system
cd clinical_diagnosis_system
```

---

### 2. Create Virtual Environment

#### Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**If you encounter issues**, install core packages individually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install timm scikit-learn pandas numpy pillow opencv-python
pip install gradio pyyaml requests tqdm kaggle
```

**For GPU support** (NVIDIA CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 4. Setup Kaggle API

#### Get Kaggle API Key:
1. Go to: https://www.kaggle.com/settings
2. Scroll to **API** section
3. Click **Create New API Token**
4. Download `kaggle.json`

#### Install Kaggle Credentials:

**Linux/macOS:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

---

## 📊 Dataset Download & Preparation

### Option 1: Automatic Download (Recommended)

```bash
# Download all datasets and normalize structure
python download_datasets.py
```

**This will:**
- Download 4 skin disease datasets from Kaggle
- Download clinical guidelines (WHO/CDC)
- Normalize all data into unified structure
- Create train/val/test splits

**Expected time:** 15-30 minutes (depending on internet speed)

---

### Option 2: Download Specific Components

```bash
# Only download Kaggle datasets
python download_datasets.py --kaggle-only

# Only download clinical guidelines
python download_datasets.py --guidelines-only

# Only normalize existing data (if already downloaded)
python download_datasets.py --normalize-only
```

---

### Option 3: Manual Download

If automated download fails, download manually from Kaggle:

1. **Melanoma**: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
2. **Eczema**: https://www.kaggle.com/datasets/adityush/eczema2
3. **Psoriasis**: https://www.kaggle.com/datasets/pallapurajkumar/psoriasis-skin-dataset
4. **Acne**: https://www.kaggle.com/datasets/tiswan14/acne-dataset-image

Extract to: `data/skin_lesions_raw/<disease_name>/`

Then run:
```bash
python download_datasets.py --normalize-only
```

---

### Verify Dataset Structure

After normalization, you should have:
```
data/
├── skin_lesions/
│   ├── train/
│   │   ├── Melanoma Skin Cancer Nevi and Moles/
│   │   ├── Eczema Photos/
│   │   ├── Psoriasis pictures Lichen Planus and related diseases/
│   │   └── Acne and Rosacea Photos/
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── (same structure)
└── guidelines/
    └── clinical_guidelines.md
```

Check dataset summary:
```bash
cat data/DATASET_SUMMARY.md
```

---

## 🎓 Model Training

### Train CNN (Skin Disease Classifier)

```bash
# Train from scratch
python train.py --train-cnn
```

**Training time:**
- **CPU**: 2-4 hours (20 epochs)
- **GPU (NVIDIA)**: 15-30 minutes (20 epochs)
- **Google Colab GPU**: 10-15 minutes (10 epochs)

**Monitor training:**
```bash
# In another terminal, watch progress
tail -f cnn_training_log.txt
```

---

### Train Random Forest (Clinical Classifier)

```bash
# Train Random Forest on clinical data
python train.py --train-rf
```

**Note:** Random Forest training is fast (~1-2 minutes)

---

### Train All Models

```bash
# Train both CNN and Random Forest
python train.py --train-all
```

---

### Resume Training

If training is interrupted, it will automatically resume from the last checkpoint:
```bash
# Will resume from models/cnn_skin_lesion_checkpoint.pth if it exists
python train.py --train-cnn
```

To start fresh (ignore checkpoints):
```bash
# Move old checkpoints
mv models/cnn_skin_lesion*.pth models/backup/

# Start new training
python train.py --train-cnn
```

---

## 🖥️ Running the Web Interface

### Start Gradio Web App

```bash
python app.py
```

**Access the interface:**
- Local: http://localhost:7860
- Network: http://<your-ip>:7860

### Features:
1. **Clinical Diagnosis Tab**:
   - Enter patient symptoms, vitals, lab values
   - Get multi-model diagnosis with confidence scores

2. **Skin Lesion Analysis Tab**:
   - Upload dermoscopic images
   - Get CNN predictions for 4 skin diseases

---

## 🐛 Troubleshooting

### Issue: Kaggle API Error

**Problem:**
```
OSError: Could not find kaggle.json
```

**Solution:**
```bash
# Verify kaggle.json location
ls -la ~/.kaggle/kaggle.json  # Linux/Mac
dir %USERPROFILE%\.kaggle\kaggle.json  # Windows

# Set correct permissions
chmod 600 ~/.kaggle/kaggle.json
```

---

### Issue: Out of Memory (OOM)

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size** (in `config/model_config.yaml`):
   ```yaml
   cnn:
     batch_size: 16  # or 8
   ```

2. **Use CPU training**:
   ```python
   # Model automatically uses CPU if CUDA unavailable
   ```

3. **Use Google Colab** with GPU

---

### Issue: Dataset Not Found

**Problem:**
```
FileNotFoundError: data/skin_lesions/train not found
```

**Solution:**
```bash
# Re-run normalization
python download_datasets.py --normalize-only

# Or check raw data exists
ls -R data/skin_lesions_raw/
```

---

### Issue: Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'timm'
```

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or install specific package
pip install timm
```

---

### Issue: Slow Training on CPU

**Problem:** Training takes too long on CPU

**Solutions:**

1. **Use Google Colab** (recommended)
2. **Reduce epochs** temporarily:
   ```yaml
   # config/model_config.yaml
   cnn:
     epochs: 5  # Instead of 20
   ```
3. **Use smaller dataset** for testing:
   ```bash
   # Manually reduce images in train folders for quick test
   ```

---

### Issue: Low Accuracy

**Problem:** Model accuracy is below 70%

**Solutions:**

1. **Train longer**:
   ```yaml
   cnn:
     epochs: 50  # Increase from 20
   ```

2. **Check data quality**:
   ```bash
   # Verify images loaded correctly
   python -c "from PIL import Image; Image.open('data/skin_lesions/train/Melanoma Skin Cancer Nevi and Moles/image_0.jpg').show()"
   ```

3. **Verify class balance**:
   ```bash
   # Count images per class
   find data/skin_lesions/train -type f | wc -l
   ls data/skin_lesions/train/*/  | wc -l
   ```

---

## 📈 Performance Benchmarks

### Expected Training Times:

| Setup | GPU | Epochs | Time | Final Accuracy |
|-------|-----|--------|------|----------------|
| Google Colab | Tesla T4 | 10 | 10-15 min | 75-85% |
| Local GPU | RTX 3060 | 20 | 20-30 min | 80-90% |
| Local CPU | i7-9700K | 20 | 2-3 hours | 75-85% |
| Colab Pro | V100 | 50 | 30-40 min | 85-95% |

### Expected Accuracy (Validation):
- **Melanoma**: 80-90%
- **Eczema**: 75-85%
- **Psoriasis**: 80-90%
- **Acne**: 70-80%

---

## 🎯 Next Steps After Training

1. **Evaluate Models**:
   ```bash
   python train.py --evaluate
   ```

2. **Run Web Interface**:
   ```bash
   python app.py
   ```

3. **Test with Examples**:
   ```bash
   # Test images in examples/ folder
   ls examples/
   ```

4. **Deploy** (optional):
   - Use Gradio's `.launch(share=True)` for public link
   - Deploy to Hugging Face Spaces
   - Containerize with Docker

---

## 📚 Additional Resources

- **Clinical Guidelines**: `data/guidelines/clinical_guidelines.md`
- **Dataset Summary**: `data/DATASET_SUMMARY.md`
- **Training Logs**: `cnn_training_log.txt`
- **Architecture**: `ARCHITECTURE.md`
- **Quick Start**: `QUICK_START.txt`

---

## 🔒 Important Notes

### Medical Disclaimer:
This system is for **educational and research purposes only**. It should NOT be used for:
- Clinical decision-making without physician oversight
- Diagnosis without proper medical examination
- Treatment recommendations without medical consultation

Always consult qualified healthcare professionals for medical advice.

### Data Privacy:
- All datasets are publicly available
- No patient-identifiable information
- Compliant with research use guidelines

### Model Limitations:
- Training data may not represent all populations
- Performance varies by image quality
- Not validated for clinical use
- Requires periodic retraining with new data

---

## 🤝 Support

### Getting Help:

1. **Check documentation**: README.md, ARCHITECTURE.md
2. **Review logs**: Training logs contain detailed error messages
3. **Verify environment**: Run `pip list` to check installed packages
4. **Test components individually**: Use `train.py --train-cnn` vs `--train-all`

### Common Commands Reference:

```bash
# Full workflow
python download_datasets.py          # Download data
python train.py --train-cnn          # Train CNN
python app.py                        # Launch web UI

# Utilities
python train.py --evaluate           # Evaluate models
python download_datasets.py --normalize-only  # Re-normalize data

# Monitoring
tail -f cnn_training_log.txt         # Watch training
ls -lh models/                       # Check model files
du -sh data/                         # Check data size
```

---

## ✅ Success Checklist

Before considering setup complete:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list | grep torch`)
- [ ] Kaggle credentials configured (`~/.kaggle/kaggle.json`)
- [ ] Datasets downloaded (check `data/DATASET_SUMMARY.md`)
- [ ] Data normalized (verify `data/skin_lesions/train/` structure)
- [ ] CNN model trained (check `models/cnn_skin_lesion.pth` exists)
- [ ] Web interface launches successfully (`python app.py`)
- [ ] Can make predictions on test images

---

**Happy Training! 🎓**

For the latest updates and improvements, check the project repository.
