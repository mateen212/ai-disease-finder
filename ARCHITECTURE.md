# System Architecture and Technical Documentation

## Overview

The Hybrid Neuro-Symbolic Clinical Decision Support System combines rule-based reasoning (symbolic AI) with machine learning (neural AI) to provide explainable multi-disease diagnosis.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Symptoms    │  │  Vitals/Labs │  │  Skin Lesion Image   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                           │
│  • Missing value imputation    • Feature encoding               │
│  • Normalization               • Image augmentation             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REASONING LAYER (PARALLEL)                    │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Symbolic AI    │  │  Neural AI   │  │  Deep Learning   │  │
│  │  ═════════════  │  │  ══════════  │  │  ══════════════  │  │
│  │  Rule Engine    │  │  Random      │  │  CNN for Skin    │  │
│  │  (Forward       │  │  Forest      │  │  Lesions         │  │
│  │   Chaining)     │  │  Classifier  │  │  (EfficientNet)  │  │
│  │                 │  │              │  │                  │  │
│  │  • WHO/CDC      │  │  • 200 trees │  │  • Pretrained    │  │
│  │    guidelines   │  │  • SHAP      │  │  • Transfer      │  │
│  │  • Medical      │  │  • Feature   │  │    learning      │  │
│  │    rules        │  │    importance│  │                  │  │
│  └─────────────────┘  └──────────────┘  └──────────────────┘  │
│          │                    │                    │            │
│          ▼                    ▼                    ▼            │
│    Rule Scores         Disease Probs         Skin Probs        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NEURO-SYMBOLIC FUSION                         │
│  • Weighted averaging of symbolic and neural outputs            │
│  • Risk level assessment                                        │
│  • Confidence scoring                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPLAINABILITY LAYER                          │
│  • SHAP values (feature importance)                             │
│  • Rule traces (which rules fired)                              │
│  • Feature contributions                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                             │
│  • Primary diagnosis + confidence                               │
│  • All disease probabilities                                    │
│  • Risk level (low/moderate/high/severe)                        │
│  • Explanations (SHAP + rules)                                  │
│  • Clinical recommendations                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Rule Engine (Symbolic AI)

**Location**: `src/rule_engine.py`

**Purpose**: Implements medical diagnostic logic based on clinical guidelines.

**Implementation**:
- Forward-chaining inference engine
- Rules defined in `config/rules.yaml`
- Evaluates patient data against diagnostic criteria

**Example Rule**:
```yaml
- name: "Dengue_Rule_Classic"
  conditions:
    all:
      - symptom: "fever" (== true)
      - vital: "temperature" (>= 38.5)
    any_2_of:
      - symptom: "headache"
      - symptom: "rash"
      - symptom: "nausea"
  conclusion:
    disease: "dengue"
    probability_boost: 0.4
```

**Advantages**:
- Transparent reasoning
- Incorporates medical expertise
- Explainable decisions
- No training data required

### 2. Random Forest Classifier (Neural AI)

**Location**: `src/ml_models.py`

**Purpose**: Learn complex patterns from clinical data.

**Implementation**:
- Scikit-learn RandomForestClassifier
- 200 decision trees
- Balanced class weights
- Trains on symptoms, vitals, labs, demographics

**Advantages**:
- Handles non-linear relationships
- Feature importance scores
- Robust to noise
- SHAP explainability

**Training**:
```python
rf_model = RandomForestDiagnostic()
rf_model.train(X_train, y_train, X_val, y_val)
rf_model.save("models/random_forest.pkl")
```

### 3. CNN for Skin Lesions (Deep Learning)

**Location**: `src/ml_models.py`

**Purpose**: Classify skin lesion images.

**Implementation**:
- EfficientNet-B0 architecture (via timm)
- Transfer learning from ImageNet
- Fine-tuned on dermatology datasets
- Input size: 224×224

**Advantages**:
- High accuracy on image data
- Captures visual patterns
- Pretrained features

**Training**:
```python
cnn_model = SkinLesionCNN()
history = cnn_model.train(train_loader, val_loader)
cnn_model.save("models/cnn_skin.pth")
```

### 4. Neuro-Symbolic Fusion

**Location**: `src/fusion.py`

**Purpose**: Combine symbolic and neural predictions.

**Fusion Strategies**:

1. **Weighted Average** (default):
   ```
   final_score = w_rule × rule_score + w_rf × rf_score + w_cnn × cnn_score
   ```
   Default weights: {rule: 0.3, RF: 0.5, CNN: 0.2}

2. **Maximum Confidence**:
   ```
   final_score = max(rule_score, rf_score, cnn_score)
   ```

3. **Stacking** (advanced):
   Meta-model learns optimal combination

**Risk Assessment**:
- Combines disease probability with rule-based risk flags
- Levels: low, moderate, high, severe
- Triggers clinical recommendations

### 5. Explainability Module

**Location**: `src/explainability.py`

**Purpose**: Make all decisions transparent and interpretable.

**SHAP (SHapley Additive exPlanations)**:
- Assigns importance to each feature
- Model-agnostic
- Additive property: sum = prediction margin

**Example**:
```
Feature Contributions:
1. Platelet Count (75,000): ↓ decreases probability (-0.25)
2. Fever (39.5°C): ↑ increases probability (+0.18)
3. Rash (present): ↑ increases probability (+0.12)
```

**Rule Traces**:
- Shows which rules fired
- Includes rule confidence
- Links to guidelines

**Combined Report**:
- Integrates symbolic and neural explanations
- Patient-friendly language
- Clinical recommendations

## Data Flow

### Patient Input → Diagnosis

1. **Input**: Patient data (JSON)
   ```json
   {
     "symptoms": {"fever": true, "rash": true, ...},
     "vitals": {"temperature": 39.5, ...},
     "labs": {"platelet_count": 95000, ...}
   }
   ```

2. **Preprocessing**:
   - Missing value imputation
   - Feature encoding
   - Normalization

3. **Parallel Reasoning**:
   - Rule engine evaluates → rule_scores
   - Random Forest predicts → rf_scores
   - CNN analyzes (if image) → cnn_scores

4. **Fusion**:
   - Weighted combination
   - Risk assessment
   - Confidence scoring

5. **Explanation**:
   - SHAP feature importance
   - Rule traces
   - Combined narrative

6. **Output**: Diagnosis report
   ```python
   {
     'diagnosis': ('dengue', 0.82),
     'risk_level': 'high',
     'recommendations': [...],
     'explanation': "..."
   }
   ```

## Performance Metrics

### Evaluation Framework

**Clinical Diseases** (dengue, COVID-19, pneumonia):
- Accuracy: Overall correctness
- Precision: True positives / all positives
- Recall: True positives / actual positives
- F1 Score: Harmonic mean of precision/recall
- AUC-ROC: Area under ROC curve

**Skin Diseases** (melanoma, eczema, psoriasis, acne):
- Same metrics as above
- Per-class accuracy

**Rule Engine**:
- Coverage: % of cases where rules fire
- Accuracy: Correctness on known cases

**Hybrid System**:
- Overall accuracy across all diseases
- Component-wise contributions
- Explanation quality

### Expected Performance

Based on similar systems in literature:
- Random Forest: ~85-90% accuracy
- CNN (skin): ~85-95% accuracy (dermatologist-level)
- Rule Engine: 100% on guideline-adherent cases
- Hybrid System: ~90-95% accuracy with high explainability

## Scalability and Extensions

### Adding New Diseases

1. **Update Rules** (`config/rules.yaml`):
   ```yaml
   malaria_rules:
     - name: "Malaria_Classic"
       conditions: {...}
       conclusion: {...}
   ```

2. **Update Configuration** (`config/model_config.yaml`):
   ```yaml
   diseases:
     clinical:
       - dengue
       - covid19
       - pneumonia
       - malaria  # New
   ```

3. **Retrain Models**:
   ```bash
   python train.py --train-all
   ```

### Customizing Fusion Weights

Edit `config/model_config.yaml`:
```yaml
fusion:
  weights:
    rule_based: 0.4  # Increase if rules are highly accurate
    random_forest: 0.4
    cnn: 0.2
```

### Improving Models

**Random Forest**:
- Increase n_estimators
- Tune max_depth
- Use GridSearchCV

**CNN**:
- Try different architectures (ResNet50, DenseNet)
- Increase training epochs
- Use more augmentation

**Rules**:
- Add more specific rules
- Include lab value thresholds
- Incorporate symptom combinations

## Citation and References

This system implements techniques from:

1. Rule-based reasoning: WHO/CDC diagnostic guidelines
2. Random Forests: Breiman (2001)
3. CNNs for dermatology: Esteva et al. (Nature 2017)
4. SHAP: Lundberg & Lee (2017)
5. Neuro-symbolic AI: Recent advances in hybrid AI systems

## Technical Requirements

- Python 3.8+
- PyTorch 2.0+
- Scikit-learn 1.3+
- SHAP 0.42+
- 8GB+ RAM (16GB recommended)
- GPU recommended for CNN training (not required for inference)

## Deployment Considerations

### Production Use

**DO NOT** deploy this system for clinical use without:
1. Extensive validation on real patient data
2. Regulatory approval (FDA, CE marking, etc.)
3. Integration with EHR systems
4. Continuous monitoring and updates
5. Medical professional oversight

### Research Use

Appropriate for:
- Educational demonstrations
- Research experiments
- Algorithm development
- Proof-of-concept studies

## Limitations

1. **Training Data**: System accuracy depends on training data quality
2. **Generalization**: May not generalize to all populations
3. **Rare Diseases**: Limited by available training examples
4. **Comorbidities**: May struggle with multiple simultaneous conditions
5. **Data Quality**: Requires complete, accurate patient data

## Future Enhancements

- [ ] Multi-lingual support
- [ ] Integration with EHR systems
- [ ] Real-time learning from new cases
- [ ] Uncertainty quantification
- [ ] Treatment recommendation
- [ ] Prognosis prediction
- [ ] Clinical trial matching

## Glossary

- **SHAP**: SHapley Additive exPlanations - method for explaining predictions
- **Forward Chaining**: Inference method that starts with facts and applies rules
- **Transfer Learning**: Using pretrained models as starting point
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
- **Neuro-Symbolic**: Combining neural networks with symbolic reasoning
- **EHR**: Electronic Health Record

---

For more information, see README.md or contact the development team.
