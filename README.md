Hybrid Neuro‑Symbolic Clinical Decision Support System
===============================================

Overview
--------

This repository implements a production‑oriented hybrid clinical decision support system (KBS) that combines symbolic rules, machine learning (Random Forest), and deep learning (CNN) for multi‑disease diagnosis and explainability. The codebase includes dataset ingestion, training pipelines, explainability artifacts (SHAP), and a Gradio web UI suitable for demonstration or integration into clinical workflows after validation.

Status & metadata
------------------

- Version: 2.0
- Last updated: 2026-04-23
- Status: Production Ready ✅ (research/prototype; requires regulatory validation before clinical use)

Important: you added a dataset URL in Google Colab. Please replace the placeholder `NEW_DATASET_URL` below with the exact URL copied from your Colab notebook so the README and automation instructions reference the correct resource.

Datasets
--------

- Primary skin images: mateenzahid/skin-diesease (Kaggle)
- Multiple public clinical CSVs are stored in `data/`
- New dataset (added in Colab):

  NEW_DATASET_URL: https://paste-your-colab-dataset-url-here  <-- replace this with your Colab dataset URL

Installation
------------

Linux/macOS (recommended):

```bash
git clone <repo-url>
cd vspython
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you will use the Kaggle downloader, configure your API key:

```bash
mkdir -p ~/.kaggle
# move your kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Download datasets (automated):

```bash
python3 download_datasets.py
```

Quickstart
----------

1. For fast model training use the Colab notebook: `colab_train.ipynb` (TPU recommended).
2. To train locally:

```bash
python3 train.py --train-rf    # RandomForest (clinical)
python3 train.py --train-cnn   # CNN (skin lesions)
```

3. Launch the web demo:

```bash
python3 app.py
# Open http://localhost:7860
```

Architecture & components
-------------------------

- `src/rule_engine.py` — forward chaining rules (explainable logic)
- `src/ml_models.py` — Random Forest and CNN model wrappers
- `src/fusion.py` — neuro‑symbolic fusion logic (weighted ensemble)
- `src/explainability.py` — SHAP pipelines and reporting
- `download_datasets.py` — automated data ingestion
- `colab_train.ipynb` — reproducible Colab training workflow (TPU)

Outputs & models
----------------

- `models/random_forest_clinical.pkl` — clinical model (RF)
- `models/cnn_skin_lesion.pth` — CNN for skin lesion classification
- `outputs/` — inference outputs, logs, and CSV summaries

Explainability
--------------

The system produces multi‑level explanations:

- Rule trace (which rules fired and why)
- SHAP feature contributions for RF predictions
- Per‑model confidence and fusion weights for transparent final scores

Deployment & governance
-----------------------

This repository is production‑oriented but not a certified medical device. Before any clinical deployment:

- Perform IRB/clinical validation studies
- Engage regulatory and clinical partners
- Establish data governance and monitoring

Contributing
------------

Please follow the repository coding standards and open a PR. If you added a dataset in Colab and want me to update this README and `download_datasets.py` with the exact URL, paste the URL here (or grant access) and I will update the docs and automation steps.

Contact & acknowledgements
--------------------------

See `KBS_FINAL_DOCUMENTATION.md` and `System_Documentation_Complete.docx` for full methodology, datasets, and citations. Acknowledgements: Kaggle dataset authors, WHO/CDC guidelines, and open-source ML tooling (PyTorch, scikit-learn, timm, Gradio, SHAP).

License
-------

MIT — see LICENSE
