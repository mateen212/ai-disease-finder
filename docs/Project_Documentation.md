# Hybrid Neuro-Symbolic Clinical Decision Support System

**Final Year Project**: Hybrid Neuro-Symbolic Clinical Decision Support System for Multi-Disease and Skin Disorder Diagnosis with Explainable AI

Date: April 14, 2026

---

## 1. Project Overview

A hybrid system combining forward-chaining clinical rules, a Random Forest classifier on clinical features, and an EfficientNet-B0 CNN on skin images. The system produces explainable, multi-disease diagnoses and human-readable recommendations.

---

## 2. Repository Files & Folders (explanation)

- Root scripts:
  - `main.py`: Entry point that initializes components, loads models, runs diagnosis pipeline (rules → RF → optional CNN → fusion → explainability) and provides CLI/demo.
  - `app.py`: Lightweight application wrapper / alternate entry.
  - `train.py`: Training driver for Random Forest and CNN, saves models into `models/`.
  - `quickstart.py`, `QUICK_START.txt`: Quick usage examples for demos.
  - `colab_train.ipynb`: Colab notebook for GPU training and reproducible experiments.
  - `install.sh`, `launcher.sh`: Environment/setup convenience scripts.
  - `requirements.txt`: Python package list.
  - `README.md`, `KBS_FINAL_DOCUMENTATION.md`: Project overview and detailed documentation.

- `config/`:
  - `model_config.yaml`: Fusion weights, disease lists, thresholds.
  - `rules.yaml`: Encoded forward-chaining production rules used by the `RuleEngine`.

- `data/`:
  - `clinical/`: CSVs used by the Random Forest and rules.
  - `skin_lesions_raw/`: Original dataset images (provenance, keep for retraining).
  - `skin_lesions/`: Preprocessed / train/val/test folders for CNN training.
  - `guidelines/`: CDC/WHO guideline files used to design rules.

- `models/`: Saved trained artifacts for inference (Random Forest `.pkl`, CNN `.pth`). Keep for running the system; archive if needed.

- `outputs/`: Generated reports and analysis artifacts (feature importances, CSVs).

- `scripts/`: Utility scripts (data download, evaluation). Inspect before deletion.

- `src/` (core code) — keep:
  - `data_preprocessing.py`: Prepares clinical features and image preprocessing.
  - `rule_engine.py`: Forward-chaining rule engine and fact management.
  - `ml_models.py`: Wrappers for Random Forest and CNN model load/predict.
  - `fusion.py`: Neuro-symbolic fusion strategies and weight management.
  - `explainability.py`: SHAP integration and patient report generation.
  - `evaluation.py`: Metrics and test harnesses.
  - `hybrid_system.py`: Programmatic API for combined usage.

- Documentation and reports: `KBS_FINAL_DOCUMENTATION.md`, `PROJECT_VERIFICATION_REPORT.md` — keep for thesis and viva.

---

## 3. Viva / Expo Q&A (short answers, 3–5 lines each)

Q: What is your project in one sentence?

A: It is a hybrid clinical decision support system that combines expert rules, a Random Forest for clinical data, and an EfficientNet-B0 CNN for skin images to give explainable diagnoses and recommendations.

Q: Why use a hybrid (neuro-symbolic) approach?

A: Rules provide clear, interpretable clinical knowledge and safety checks, while ML/CNN learn patterns from data. Combining them improves accuracy and clinician trust.

Q: Which diseases does the system handle?

A: Clinical diseases: dengue, COVID-19, pneumonia. Skin disorders: melanoma, eczema, psoriasis, acne. The system supports multi-disease outputs and combined inputs.

Q: Why EfficientNet-B0 for the CNN?

A: EfficientNet-B0 balances accuracy and model size; with transfer learning it trains well on limited GPU resources like Colab and works effectively on dermatoscopic images.

Q: How does input flow through the system?

A: Inputs are preprocessed (clinical features and optional image), then evaluated by the rule engine, Random Forest, and CNN. The `NeuroSymbolicFusion` combines results and the explainability module builds the report.

Q: What is the default fusion strategy?

A: Weighted averaging with configurable weights (default: rules 30%, RF 50%, CNN 20%), with alternatives like max or stacking if needed.

Q: How is the system explainable?

A: It records rule traces from `rule_engine.py` and uses SHAP to explain Random Forest predictions; both are combined into a readable clinician report.

Q: How did you evaluate model performance?

A: Standard train/validation/test splits were used; Random Forest and CNN metrics were recorded, and SHAP/feature importance checked to ensure sensible decisions.

Q: How would a clinician use the tool?

A: Run `python main.py --demo` or provide a JSON patient file and optional skin image; the system returns a diagnosis, risk level, component scores, and recommendations.

Q: What are the main challenges you faced?

A: Balancing image vs clinical signals in fusion, handling class imbalance, and presenting concise, clinically relevant explanations.

Q: What improvements do you plan?

A: Dynamic fusion weights based on confidence, larger clinical validation, better UI for clinical workflows, and more diverse dataset evaluation.

---

## 4. Libraries / Packages (what and why — short)

- `numpy` — numeric arrays and math. Used for data handling and numeric ops.
- `pandas` — table data handling. Used to read and transform clinical CSVs.
- `scikit-learn` — ML algorithms. Used for Random Forest classifier and evaluation.
- `scipy` — scientific utilities. Used for stats and auxiliary computations.
- `torch` (PyTorch) — deep learning framework. Used to train and run EfficientNet-B0 CNN.
- `torchvision` — image transforms and pretrained helpers. Used for image preprocessing and utilities.
- `timm` — model zoo including EfficientNet implementations. Used to load EfficientNet-B0 pretrained weights.
- `Pillow` — image I/O. Used to open, resize, and save images.
- `opencv-python` — image processing. Used for augmentations and fast ops.
- `shap` — explainability (SHAP values). Used to produce feature-level explanations for RF.
- `kaggle` — dataset downloader. Used to fetch the unified skin dataset programmatically.
- `matplotlib` / `seaborn` — plotting. Used for training curves and visualization in reports.
- `tqdm` — progress bars. Used to show progress during training and preprocessing.
- `pyyaml` — YAML parsing. Used to load `model_config.yaml` and `rules.yaml`.
- `requests` — HTTP requests. Used for auxiliary downloads.
- `gradio` — demo UI. Used to provide a simple web interface for demos.
- `plotly` (optional) — interactive charts. Used for richer visualizations if needed.

---

## 5. Files I removed (clean-up done earlier)

- `cnn_training_log_old_7classes.txt` (training log)
- `cnn_training_log.txt` (training log)
- `rf_training_log.txt` (training log)
- `training_log.txt` (training log)
- `README.md.backup` (backup file)

These were logs and a backup not required in the source tree.

---

## 6. Next steps / Delivery

- The Markdown file `docs/Project_Documentation.md` has been created in the repository.
- I will now attempt to convert it to `docs/Project_Documentation.pdf` using `pandoc`. If conversion fails, the Markdown file remains available for easy printing or conversion.


---

_End of document._
