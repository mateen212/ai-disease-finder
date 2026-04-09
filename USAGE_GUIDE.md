# 🚀 Usage Guide - Hybrid Clinical Decision Support System

## Quick Start

### 1️⃣ Launch the Web Interface (GUI)

The easiest way to use the system is through the web interface:

```bash
python3 app.py
```

Or with virtual environment:
```bash
/home/dev/projects/python/vspython/.venv/bin/python app.py
```

**Access the interface:**
- Open your browser to: `http://localhost:7860`
- The GUI will automatically load all trained models
- You can use it from any device on your network

---

## 📊 Using the Web Interface

### Clinical Diagnosis Tab

1. **Enter Patient Information:**
   - ✅ Check relevant symptoms (fever, cough, etc.)
   - 🌡️ Adjust vital signs (temperature, heart rate, oxygen saturation)
   - 🧪 Input lab values (WBC, platelets, CRP, etc.)
   - 👤 Set demographics (age, sex)

2. **Click "Diagnose"**
   - System analyzes using all three models:
     - Rule Engine (expert rules)
     - Random Forest (ML classifier)
     - CNN (image analysis if applicable)
   
3. **View Results:**
   - Primary diagnosis with confidence score
   - Disease probabilities (bar chart)
   - Explanation of reasoning
   - Confidence level (color-coded)

### Skin Lesion Analysis Tab

1. **Upload Image:**
   - Click the image upload area
   - Select a dermoscopic skin lesion image
   - Supported formats: JPG, PNG, JPEG

2. **Click "Analyze Image"**
   - CNN processes the image
   - Identifies lesion type

3. **View Results:**
   - Top diagnosis with confidence
   - Probabilities for all 7 lesion types:
     - Melanoma
     - Melanocytic Nevus
     - Basal Cell Carcinoma
     - Actinic Keratosis
     - Benign Keratosis
     - Dermatofibroma
     - Vascular Lesion

---

## 🔄 Resume Training (Checkpoint Support)

### The system now automatically saves and resumes training!

**How it works:**
- Every epoch, the system saves a checkpoint: `models/cnn_skin_lesion_checkpoint.pth`
- This checkpoint contains:
  - Model weights
  - Optimizer state
  - Current epoch number
  - Best validation accuracy so far

**To resume training after stopping:**

```bash
python3 train.py --train-cnn 2>&1 | tee cnn_training_log.txt
```

**What happens:**
1. ✅ Script checks for existing checkpoint
2. ✅ If found, loads model + optimizer state
3. ✅ Continues from the next epoch
4. ✅ Preserves best validation accuracy
5. ✅ Appends to training log

**Example output when resuming:**
```
Found existing checkpoint: models/cnn_skin_lesion_checkpoint.pth
✓ Resuming from epoch 3, best_val_acc=80.28%
Training CNN model...
Epoch 3/20: 100%|██████████| 251/251 [35:12<00:00]
```

**You can stop and resume as many times as you want!**
- Press `Ctrl+C` to stop
- Run the same command to resume
- Works even after closing the terminal or rebooting

---

## 📁 Important Files

### Models (saved during training)
- `models/cnn_skin_lesion.pth` - Best model (highest validation accuracy)
- `models/cnn_skin_lesion_checkpoint.pth` - Latest checkpoint for resuming
- `models/random_forest_clinical.pkl` - Random Forest classifier
- `models/rule_engine.yaml` - Expert rules

### Data
- `data/skin_lesions/images/` - Training images
- `data/skin_lesions/labels.csv` - Image labels
- `data/clinical_training_data.csv` - Clinical training data

### Outputs
- `cnn_training_log.txt` - Training progress log
- `outputs/` - Evaluation results, plots, metrics

---

## 🛠️ Training Commands

### Train Everything (from scratch or resume)
```bash
python3 train.py --train-all
```

### Train Only CNN (with resume support)
```bash
python3 train.py --train-cnn 2>&1 | tee cnn_training_log.txt
```

### Train Only Random Forest
```bash
python3 train.py --train-rf
```

### Test Rule Engine
```bash
python3 train.py --test-rules
```

---

## 📊 Monitor Training Progress

### View live training log:
```bash
tail -f cnn_training_log.txt
```

### Check current epoch:
```bash
grep "Epoch.*Val Acc" cnn_training_log.txt | tail -5
```

### Check if training is running:
```bash
ps aux | grep "train.py"
```

### View checkpoint details:
```bash
ls -lh models/cnn_skin_lesion*.pth
```

---

## 🔍 Command-Line Diagnosis (No GUI)

You can also use the system from command line:

```bash
python3 main.py --help
```

**Examples:**

```bash
# Diagnose from example file
python3 main.py --patient-file examples/covid19_patient.json

# Interactive mode
python3 main.py --interactive

# Diagnose skin lesion
python3 main.py --skin-lesion path/to/image.jpg
```

---

## ⚙️ Advanced Configuration

### Change training settings:
Edit `config/model_config.yaml`

**CNN settings:**
```yaml
cnn:
  architecture: efficientnet_b0
  num_classes: 7
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
```

**Random Forest settings:**
```yaml
random_forest:
  n_estimators: 200
  max_depth: 20
  class_weight: balanced
```

---

## 🐛 Troubleshooting

### GUI doesn't start
```bash
# Install Gradio
pip install gradio --user

# Or with virtual env
/home/dev/projects/python/vspython/.venv/bin/python -m pip install gradio
```

### "Model not found" error
```bash
# Train models first
python3 train.py --train-all
```

### Training doesn't resume
```bash
# Check if checkpoint exists
ls -lh models/cnn_skin_lesion_checkpoint.pth

# If missing, training starts from scratch (this is normal)
```

### Out of memory during training
```bash
# Reduce batch size in config/model_config.yaml
# Change: batch_size: 32 -> batch_size: 16
```

---

## 📸 Example Screenshots

### Clinical Diagnosis Interface
- Symptom checkboxes
- Vital sign sliders
- Lab value inputs
- Real-time diagnosis results

### Skin Lesion Analysis Interface
- Drag-and-drop image upload
- CNN predictions with probabilities
- Color-coded confidence levels

---

## 🔐 Security Notes

- GUI runs locally by default (`localhost:7860`)
- To allow external access, edit `app.py`:
  - Change `share=False` to `share=True`
  - Creates public URL (be careful with patient data!)

---

## 📚 Additional Resources

- **Full Documentation:** See `KBS_DOCUMENTATION_REPORT.md`
- **CNN Training Details:** See `CNN_TRAINING_EXPLAINED.md`
- **Quick Answers:** See `QUICK_ANSWERS.md`
- **System Summary:** See `KBS_SUMMARY.md`

---

## 💡 Tips

1. **Always use the GUI** - It's the easiest way to interact with the system
2. **Save checkpoint frequently** - Training automatically saves every epoch
3. **Monitor training** - Use `tail -f cnn_training_log.txt` to watch progress
4. **Test with examples** - Use files in `examples/` folder to test
5. **Check confidence** - Only act on high-confidence predictions

---

## ⚠️ Medical Disclaimer

This system is for **educational and research purposes only**.

**DO NOT use for actual medical diagnosis without:**
- Professional medical supervision
- Proper clinical validation
- Regulatory approval
- Patient consent

Always consult qualified healthcare professionals for medical decisions.

---

## 📞 Support

For issues or questions, review:
1. This usage guide
2. Documentation files
3. Training logs
4. Error messages

The system provides detailed error messages to help troubleshoot.

---

**Happy diagnosing! 🏥**
