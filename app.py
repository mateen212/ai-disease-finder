"""
Gradio Web Interface for Hybrid Clinical Decision Support System

Allows users to:
- Input patient symptoms, vitals, and lab values
- Upload skin lesion images
- Get predictions from Rule Engine, Random Forest, CNN
- View hybrid decision with confidence scores
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
import gradio as gr
import shap
import matplotlib.pyplot as plt
import io
import base64
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.rule_engine import RuleEngine
from src.ml_models import RandomForestDiagnostic, SkinLesionCNN
from src.data_preprocessing import DataPreprocessor
from src.hybrid_system import HybridDiagnosticSystem
from src.explainability import ExplainabilityModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalDiagnosisApp:
    """Main application class for the diagnosis system"""
    
    def __init__(self):
        """Initialize all models and components"""
        logger.info("Initializing Clinical Diagnosis System...")
        
        # Skin lesion class names (define BEFORE loading models)
        # Updated to match new 5-class training dataset (including healthy skin)
        self.skin_classes = [
            "Acne and Rosacea Photos",
            "Eczema Photos",
            "Melanoma Skin Cancer Nevi and Moles",
            "Normal Healthy Skin",
            "Psoriasis pictures Lichen Planus and related diseases"
        ]
        
        # Initialize components
        self.rule_engine = RuleEngine()
        self.rf_model = RandomForestDiagnostic()
        self.cnn_model = SkinLesionCNN()
        self.preprocessor = DataPreprocessor()
        self.hybrid_system = HybridDiagnosticSystem()
        # Explainability helper (guidelines + textual fusion)
        self.explainer = ExplainabilityModule()
        
        # Load trained models
        self._load_models()
        
        logger.info("✓ System initialized successfully")
    
    def _load_models(self):
        """Load trained models from disk"""
        # Load Random Forest
        rf_path = "models/random_forest_clinical.pkl"
        if Path(rf_path).exists():
            try:
                self.rf_model.load(rf_path)
                logger.info(f"✓ Loaded Random Forest from {rf_path}")
            except Exception as e:
                logger.warning(f"Could not load RF model: {e}")
        
        # Load CNN (5-class model with Normal/Healthy Skin)
        cnn_path = "models/cnn_skin_lesion.pth"
        if Path(cnn_path).exists():
            try:
                self.cnn_model.load(cnn_path)
                self.cnn_model.class_names = self.skin_classes
                logger.info(f"✓ Loaded CNN from {cnn_path}")
            except Exception as e:
                logger.warning(f"Could not load CNN model: {e}")
    
    def diagnose_clinical(
        self,
        # Symptoms
        fever: bool, cough: bool, fatigue: bool, headache: bool,
        shortness_of_breath: bool, loss_of_taste: bool, sore_throat: bool,
        body_aches: bool, nausea: bool, vomiting: bool, diarrhea: bool,
        rash: bool, joint_pain: bool, retro_orbital_pain: bool,
        # Vitals
        temperature: float, heart_rate: float, respiratory_rate: float,
        blood_pressure_systolic: float, blood_pressure_diastolic: float,
        oxygen_saturation: float,
        # Labs
        wbc_count: float, platelet_count: float, hemoglobin: float,
        crp: float, ferritin: float,
        # Demographics
        age: int, sex: str,
        # Filter
        disease_filter: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Perform clinical diagnosis using rule engine and ML models
        
        Returns:
            Tuple of (results_text, chart_html, confidence_text)
        """
        try:
            # Prepare patient data
            patient_data = {
                'symptoms': {
                    'fever': fever,
                    'cough': cough,
                    'fatigue': fatigue,
                    'headache': headache,
                    'shortness_of_breath': shortness_of_breath,
                    'loss_of_taste': loss_of_taste,
                    'sore_throat': sore_throat,
                    'body_aches': body_aches,
                    'nausea': nausea,
                    'vomiting': vomiting,
                    'diarrhea': diarrhea,
                    'rash': rash,
                    'joint_pain': joint_pain,
                    'retro_orbital_pain': retro_orbital_pain
                },
                'vitals': {
                    'temperature': temperature,
                    'heart_rate': heart_rate,
                    'respiratory_rate': respiratory_rate,
                    'blood_pressure_systolic': blood_pressure_systolic,
                    'blood_pressure_diastolic': blood_pressure_diastolic,
                    'oxygen_saturation': oxygen_saturation
                },
                'labs': {
                    'wbc_count': wbc_count,
                    'platelet_count': platelet_count,
                    'hemoglobin': hemoglobin,
                    'crp': crp,
                    'ferritin': ferritin
                },
                'demographics': {
                    'age': age,
                    'sex': sex
                }
            }
            
            # Get hybrid diagnosis
            diagnosis_result = self.hybrid_system.diagnose(patient_data)
            
            # Filter results if disease_filter is specified
            if disease_filter:
                diagnosis_result = self._filter_disease_results(diagnosis_result, disease_filter)
            
            # Format results
            results_text = self._format_diagnosis_results(diagnosis_result, disease_filter)
            chart_html = self._create_probability_chart(diagnosis_result['probabilities'])
            confidence_text = self._format_confidence(diagnosis_result)

            # Append disease-specific guideline excerpt
            try:
                primary = diagnosis_result.get('diagnosis')
                if primary:
                    guidance_data = self.explainer.get_guidance_for_disease(primary)
                    if guidance_data and guidance_data.get('content') and len(guidance_data['content']) > 50:
                        disease_display = primary.upper().replace('_', ' ')
                        source = guidance_data.get('source', 'clinical guidelines')
                        results_text += f"\n\n### 📋 {disease_display} Clinical Guidelines\n\n"
                        results_text += guidance_data['content']
                        results_text += f"\n\n*Source: {source}*"
            except Exception as e:
                logger.debug(f"Could not append clinical guidance: {e}")

            return results_text, chart_html, confidence_text
            
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            return f"Error: {str(e)}", "", ""
    
    def diagnose_from_json(self, json_str: str, disease_filter: Optional[str] = None) -> Tuple[str, str, str, str]:
        """
        Diagnose from JSON input
        
        Returns:
            Tuple of (results_text, chart_html, confidence_text, json_output)
        """
        try:
            data = json.loads(json_str)
            
            # Extract with defaults
            symptoms = data.get('symptoms', {})
            vitals = data.get('vitals', {})
            labs = data.get('labs', {})
            demographics = data.get('demographics', {})
            
            results = self.diagnose_clinical(
                fever=symptoms.get('fever', False),
                cough=symptoms.get('cough', False),
                fatigue=symptoms.get('fatigue', False),
                headache=symptoms.get('headache', False),
                shortness_of_breath=symptoms.get('shortness_of_breath', False),
                loss_of_taste=symptoms.get('loss_of_taste', False),
                sore_throat=symptoms.get('sore_throat', False),
                body_aches=symptoms.get('body_aches', False),
                nausea=symptoms.get('nausea', False),
                vomiting=symptoms.get('vomiting', False),
                diarrhea=symptoms.get('diarrhea', False),
                rash=symptoms.get('rash', False),
                joint_pain=symptoms.get('joint_pain', False),
                retro_orbital_pain=symptoms.get('retro_orbital_pain', False),
                temperature=vitals.get('temperature', 37.0),
                heart_rate=vitals.get('heart_rate', 75.0),
                respiratory_rate=vitals.get('respiratory_rate', 16.0),
                blood_pressure_systolic=vitals.get('blood_pressure_systolic', 120.0),
                blood_pressure_diastolic=vitals.get('blood_pressure_diastolic', 80.0),
                oxygen_saturation=vitals.get('oxygen_saturation', 98.0),
                wbc_count=labs.get('wbc_count', 7000.0),
                platelet_count=labs.get('platelet_count', 250000.0),
                hemoglobin=labs.get('hemoglobin', 14.0),
                crp=labs.get('crp', 5.0),
                ferritin=labs.get('ferritin', 100.0),
                age=demographics.get('age', 30),
                sex=demographics.get('sex', 'Male'),
                disease_filter=disease_filter
            )
            
            # Create output JSON (pretty formatted for display)
            output_json = json.dumps(data, indent=2)
            
            return results[0], results[1], results[2], output_json
            
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {str(e)}", "", "", ""
        except Exception as e:
            logger.error(f"JSON diagnosis error: {e}")
            return f"Error: {str(e)}", "", "", ""
    
    def _filter_disease_results(self, result: Dict[str, Any], disease_filter: str) -> Dict[str, Any]:
        """Filter diagnosis results to focus on specific disease"""
        disease_map = {
            'COVID-19': ['covid', 'covid-19', 'coronavirus'],
            'Dengue': ['dengue'],
            'Pneumonia': ['pneumonia']
        }
        
        if disease_filter not in disease_map:
            return result
        
        # Filter probabilities to only relevant diseases
        keywords = disease_map[disease_filter]
        filtered_probs = {
            disease: prob 
            for disease, prob in result['probabilities'].items()
            if any(kw in disease.lower() for kw in keywords)
        }
        
        # If no matches, keep original
        if not filtered_probs:
            filtered_probs = result['probabilities']
        
        # Update result
        filtered_result = result.copy()
        filtered_result['probabilities'] = filtered_probs
        
        return filtered_result
    
    def diagnose_skin_lesion(self, image: Optional[Image.Image], explain_mode: str = 'Fast (saliency)') -> Tuple[str, str, str, str]:
        """
        Diagnose skin lesion from uploaded image
        
        Returns:
            Tuple of (results_text, chart_html, confidence_text)
        """
        if image is None:
            return "Please upload an image", "", "", ""
        
        try:
            # Preprocess image
            print("Preprocessing image...")
            image_tensor = self.preprocessor.test_transform(image)
            print("Image preprocessed successfully", image_tensor.shape)
            # Get CNN predictions
            probabilities = self.cnn_model.predict_proba(image_tensor)
            print("CNN prediction probabilities:", probabilities)
            
            # Find top prediction
            top_class = max(probabilities.items(), key=lambda x: x[1])
            print(f"Top predicted class: {top_class[0]} with probability {top_class[1]:.4f}")
            
            
            # Format results
            results_text = f"## 🔬 Skin Lesion Analysis\n\n"
            results_text += f"**Primary Diagnosis:** {top_class[0]}\n\n"
            results_text += f"**Confidence:** {top_class[1]*100:.1f}%\n\n"
            results_text += "### All Probabilities:\n"
            print("Formatting all probabilities for display...")
            
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            print("Sorted probabilities:", sorted_probs)
            for disease, prob in sorted_probs:
                bar = "█" * int(prob * 20)
                results_text += f"- **{disease}**: {prob*100:.1f}% {bar}\n"
            print("Formatted results text:", results_text)
            # Create chart
            chart_html = self._create_probability_chart(probabilities)

            # Always append a detailed textual explanation (guidance + probabilities)
            try:
                top_name = max(probabilities.items(), key=lambda x: x[1])[0]
                detailed_text = self.explainer.generate_narrative_for_image(probabilities, top_name)
                results_text += "\n\n### Detailed Explanation and Guidance:\n\n" + detailed_text
            except Exception as e:
                logger.debug(f"Could not generate detailed guidance text: {e}")
            
            # Confidence assessment
            if top_class[1] >= 0.8:
                confidence = "High confidence"
                color = "green"
            elif top_class[1] >= 0.6:
                confidence = "Moderate confidence"
                color = "orange"
            else:
                confidence = "Low confidence - consider additional tests"
                color = "red"
            
            confidence_text = f"<div style='padding:10px; background-color:{color}; color:white; border-radius:5px;'>{confidence}</div>"
            
            # Default explanation: fast saliency overlay (selected via `explain_mode` argument)
            explain_html = ""

            if explain_mode in ('Fast', 'Fast (saliency)'):
                try:
                    explain_html = self._generate_saliency_image(image_tensor)
                except Exception as e:
                    logger.warning(f"Fast explanation failed: {e}")
                    explain_html = ""
            elif explain_mode in ('SHAP', 'SHAP (slow)'):
                try:
                    explain_html = self._generate_shap_image(image_tensor)
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
                    explain_html = ""
            elif explain_mode in ('Textual',):
                # Use ExplainabilityModule to build textual explanation combining
                # guideline text (from data/guidelines) and model probabilities.
                top = max(probabilities.items(), key=lambda x: x[1])[0]
                # explain_html = self.explainer.generate_text_for_image(probabilities, top)

            return results_text, chart_html, confidence_text, explain_html
            
        except Exception as e:
            logger.error(f"Skin lesion diagnosis error: {e}")
            return f"Error: {str(e)}", "", "", ""
    
    def _format_diagnosis_results(self, result: Dict[str, Any], disease_filter: Optional[str] = None) -> str:
        """Format diagnosis results as markdown"""
        diagnosis = result['diagnosis']
        confidence = result['confidence']
        
        # Select emoji based on disease
        emoji = "🩺"  # Stethoscope default
        if 'covid' in diagnosis.lower():
            emoji = "💉"  # Syringe
        elif 'dengue' in diagnosis.lower() or 'malaria' in diagnosis.lower():
            emoji = "🦟"  # Mosquito
        elif 'pneumonia' in diagnosis.lower():
            emoji = "🫁"  # Lungs
        elif any(x in diagnosis.lower() for x in ['melanoma', 'acne', 'eczema', 'psoriasis']):
            emoji = "👨‍⚕️"  # Health worker
        
        diagnosis_display = diagnosis.upper().replace('_', ' ')
        text = f"## {emoji} Clinical Diagnosis Results\n\n"
        
        # Primary diagnosis
        text += f"### Primary Diagnosis: **{diagnosis_display}**\n\n"
        text += f"**Overall Confidence:** {confidence*100:.1f}%\n\n"
        
        # Component predictions
        text += "### 🔍 Component Model Predictions:\n\n"
        
        for component, data in result['component_predictions'].items():
            text += f"#### {component}\n"
            if 'top_disease' in data:
                text += f"- Prediction: **{data['top_disease']}**\n"
                text += f"- Confidence: {data['top_score']*100:.1f}%\n"
            
            # Show rule-specific details
            if component == "Rule Engine" and 'metadata' in data:
                metadata = data['metadata']
                fired_rules = metadata.get('fired_rules', [])
                
                if fired_rules:
                    # Group by disease
                    rules_by_disease = {}
                    for rule in fired_rules:
                        disease = rule.conclusion.get('disease', 'unknown')
                        if disease not in rules_by_disease:
                           rules_by_disease[disease] = []
                        rules_by_disease[disease].append(rule)
                    
                    primary_disease = result.get('diagnosis', '')
                    
                    # Show primary rules
                    if primary_disease in rules_by_disease:
                        text += f"- **✅ {len(rules_by_disease[primary_disease])} rules support PRIMARY diagnosis** ({primary_disease})\n"
                        for rule in rules_by_disease[primary_disease]:
                            risk = rule.conclusion.get('risk_level', '')
                            risk_text = f" [{risk.upper()}]" if risk else ""
                            boost = rule.conclusion.get('probability_boost', 0)
                            text += f"  - {rule.name}{risk_text} (+{boost*100:.0f}%)\n"
                    
                    # Show competing
                    other_diseases = [d for d in rules_by_disease.keys() if d != primary_disease]
                    if other_diseases:
                        text += f"- ⚠️ Competing diagnoses (ruled out by lower confidence):\n"
                        for disease in sorted(other_diseases):
                            final_prob = result['probabilities'].get(disease, 0)
                            text += f"  - {disease}: {len(rules_by_disease[disease])} rules fired, but final confidence only {final_prob*100:.1f}%\n"
            
            text += "\n"
        
        # Top probabilities
        if disease_filter:
            text += f"### {disease_filter} Probability:\n\n"
        else:
            text += "### Disease Probabilities:\n\n"
        
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for disease, prob in sorted_probs[:5]:
            bar = "█" * int(prob * 20)
            text += f"- **{disease.upper()}**: {prob*100:.1f}% {bar}\n"
        
        # Explanation
        if 'explanation' in result:
            text += f"\n### Explanation:\n\n{result['explanation']}\n"
        
        return text
    
    def _format_confidence(self, result: Dict[str, Any]) -> str:
        """Format confidence level with color coding"""
        confidence = result['confidence']
        
        if confidence >= 0.8:
            level = "HIGH CONFIDENCE"
            color = "#28a745"
            message = "Strong evidence for this diagnosis"
        elif confidence >= 0.6:
            level = "MODERATE CONFIDENCE"
            color = "#ffc107"
            message = "Reasonable evidence, consider additional tests"
        else:
            level = "LOW CONFIDENCE"
            color = "#dc3545"
            message = "Uncertain diagnosis - additional evaluation recommended"
        
        html = f"""
        <div style='padding: 20px; background-color: {color}; color: white; 
                    border-radius: 10px; text-align: center; font-weight: bold;'>
            <h2>{level}</h2>
            <p>{message}</p>
        </div>
        """
        
        return html
    
    def _create_probability_chart(self, probabilities: Dict[str, float]) -> str:
        """Create HTML bar chart for probabilities"""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        html = "<div style='padding: 10px;'>"
        
        for disease, prob in sorted_probs:
            percentage = prob * 100
            color = self._get_color_for_probability(prob)
            
            html += f"""
            <div style='margin: 10px 0;'>
                <div style='font-weight: bold; margin-bottom: 5px;'>{disease}</div>
                <div style='background-color: #e0e0e0; border-radius: 5px; overflow: hidden;'>
                    <div style='width: {percentage}%; background-color: {color}; 
                                padding: 5px; color: white; text-align: right; 
                                border-radius: 5px;'>
                        {percentage:.1f}%
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _get_color_for_probability(self, prob: float) -> str:
        """Get color based on probability value"""
        if prob >= 0.7:
            return "#28a745"  # Green
        elif prob >= 0.5:
            return "#ffc107"  # Yellow
        elif prob >= 0.3:
            return "#fd7e14"  # Orange
        else:
            return "#6c757d"  # Gray

    def _generate_shap_image(self, image_tensor: torch.Tensor) -> str:
        """Create SHAP explanation image for the given image tensor and return HTML <img> data URI."""
        try:
            # Ensure model in eval mode
            model = self.cnn_model.model
            model.eval()

            # Prepare background (single zero image) on same device
            device = next(model.parameters()).device
            if image_tensor.dim() == 3:
                img = image_tensor.unsqueeze(0)
            else:
                img = image_tensor

            # Create a small background set - zero image (works as baseline)
            background = torch.zeros_like(img).to(device)

            # Use GradientExplainer for PyTorch models
            explainer = shap.GradientExplainer(model, background)

            # Compute SHAP values (may take a few seconds)
            shap_values = explainer.shap_values(img.to(device))

            # shap_values is a list (one array per class) of shape (N, C, H, W)
            # We'll sum absolute values across classes to visualise important pixels
            import numpy as _np

            summed = _np.sum(_np.abs(_np.stack([_np.transpose(sv[0], (1,2,0)) for sv in shap_values])), axis=0)

            # Normalize for display
            summed = (summed - summed.min()) / (summed.max() - summed.min() + 1e-8)

            # Create heatmap plot over original image
            fig, ax = plt.subplots(figsize=(4,4), dpi=100)
            # Original image in HWC [0,1]
            orig = _np.transpose(img.detach()[0].cpu().numpy(), (1,2,0))
            ax.imshow(orig)
            ax.imshow(summed, cmap='jet', alpha=0.5)
            ax.axis('off')

            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            b64 = base64.b64encode(buf.read()).decode('utf-8')
            html_img = f"data:image/png;base64,{b64}"
            return html_img
        except Exception as e:
            logger.warning(f"Error generating SHAP image: {e}")
            return ""

    def _generate_saliency_image(self, image_tensor: torch.Tensor) -> str:
        """Fast saliency (input-gradient) explanation overlaid on the image.

        This is much faster than SHAP for a single image: one forward and one backward pass.
        Returns a data-URI PNG string suitable for Gradio HTML/Image display.
        """
        try:
            model = self.cnn_model.model
            model.eval()

            device = next(model.parameters()).device
            if image_tensor.dim() == 3:
                img = image_tensor.unsqueeze(0).to(device)
            else:
                img = image_tensor.to(device)

            # Ensure gradients
            img.requires_grad = True

            outputs = model(img)
            pred_idx = int(outputs.argmax(dim=1).item())
            score = outputs[0, pred_idx]

            model.zero_grad()
            if img.grad is not None:
                img.grad.zero_()
            score.backward(retain_graph=False)

            grad = img.grad.detach().cpu().numpy()[0]  # C,H,W
            import numpy as _np

            saliency = _np.sum(_np.abs(grad), axis=0)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

            # Original image HWC in [0,1]
            orig = _np.transpose(img[0].cpu().numpy(), (1,2,0))

            fig, ax = plt.subplots(figsize=(4,4), dpi=100)
            ax.imshow(orig)
            ax.imshow(saliency, cmap='jet', alpha=0.5)
            ax.axis('off')

            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            b64 = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            logger.warning(f"Saliency generation failed: {e}")
            return ""

    def _generate_textual_explanation(self, probabilities: Dict[str, float]) -> str:
        """Generate a concise textual explanation based on model probabilities."""
        try:
            # Use class order if available
            classes = self.cnn_model.class_names or list(probabilities.keys())
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

            top, second = sorted_probs[0], sorted_probs[1] if len(sorted_probs) > 1 else (None, None)

            templates = {
                'Melanoma Skin Cancer Nevi and Moles': (
                    "Model assigns {p:.1f}% probability to Melanoma. "
                    "Visual patterns associated by the model include irregular borders and color variegation. "
                    "Recommend dermatology referral and dermoscopic examination."
                ),
                'Eczema Photos': (
                    "Model assigns {p:.1f}% probability to Eczema. "
                    "Patterns the model finds include diffuse redness and scaling. "
                    "Consider clinical correlation with history and topical treatment guidance."
                ),
                'Psoriasis pictures Lichen Planus and related diseases': (
                    "Model assigns {p:.1f}% probability to Psoriasis or related conditions. "
                    "The model highlights well-demarcated plaques and silvery scales. "
                    "Consider dermatologist evaluation and possible biopsy if uncertain."
                ),
                'Acne and Rosacea Photos': (
                    "Model assigns {p:.1f}% probability to Acne/Rosacea. "
                    "Typical model features include pustules, comedones, or centrofacial erythema. "
                    "Treatment depends on severity; consider dermatologic care."
                ),
                'Normal Healthy Skin': (
                    "Model assigns {p:.1f}% probability to Normal/Healthy Skin. "
                    "Image appears consistent with non-pathologic skin. "
                    "No specific dermatologic intervention suggested based on image alone."
                )
            }

            expl = "Explanation\n\n"
            for cls, prob in sorted_probs:
                full_name = cls
                p = prob * 100
                if full_name in templates:
                    expl += f"- **{full_name}**: {templates[full_name].format(p=p)}\n\n"
                else:
                    expl += f"- **{full_name}**: Model probability {p:.1f}%.\n\n"

            # Add short recommendation based on top class confidence
            if top and top[1] >= 0.8:
                expl += "**Recommendation:** High-confidence result; consider specialist follow-up."
            elif top and top[1] >= 0.5:
                expl += "**Recommendation:** Moderate confidence; correlate clinically and consider expert review."
            else:
                expl += "**Recommendation:** Low confidence; obtain additional clinical information or specialist review."

            return expl
        except Exception as e:
            logger.warning(f"Textual explanation failed: {e}")
            return ""


def create_interface():
    """Create Gradio interface"""
    app = ClinicalDiagnosisApp()
    
    # Custom CSS
    custom_css = """
    .gradio-container { font-family: 'Inter', 'Arial', sans-serif; }
    .gradio-container .panel { box-shadow: 0 6px 18px rgba(36, 37, 38, 0.08); border-radius: 12px; }
    .output-markdown h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    .gr-button.primary { background: linear-gradient(90deg,#357edd,#1b6ed8); color: white; border: none; }
    .gradio-container .footer { color: #7f8c8d; }
    img[alt="SHAP Explanation"] { max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    """
    
    with gr.Blocks(css=custom_css, title="Clinical Diagnosis System") as interface:
        gr.Markdown("""
        # 🏥 Hybrid Neuro-Symbolic Clinical Decision Support System
        
        This system combines **Rule-Based Reasoning**, **Machine Learning (Random Forest)**, 
        and **Deep Learning (CNN)** to provide accurate disease diagnosis.
        
        ### Select a tab below to start:
        """)
        
        with gr.Tabs():
            # Tab 1: Clinical Diagnosis (with disease-specific sub-tabs)
            with gr.Tab("🩺 Clinical Diagnosis"):
                gr.Markdown("### Select a disease-specific check below")
                
                with gr.Tabs():
                    # Sub-tab 1: COVID-19 Check
                    with gr.Tab("🦠 COVID-19 Check"):
                        gr.Markdown("**Primary Symptoms:** Fever, cough, loss of taste/smell, shortness of breath")
                        
                        with gr.Tabs():
                            # Form Input
                            with gr.Tab("📝 Form Input"):
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("#### COVID-19 Specific Symptoms")
                                        covid_fever = gr.Checkbox(label="Fever", value=False)
                                        covid_cough = gr.Checkbox(label="Dry Cough")
                                        covid_fatigue = gr.Checkbox(label="Fatigue")
                                        covid_loss_taste = gr.Checkbox(label="Loss of Taste/Smell")
                                        covid_shortness = gr.Checkbox(label="Shortness of Breath")
                                        covid_sore_throat = gr.Checkbox(label="Sore Throat")
                                        covid_body_aches = gr.Checkbox(label="Body Aches")
                                        covid_headache = gr.Checkbox(label="Headache")
                                        covid_diarrhea = gr.Checkbox(label="Diarrhea")
                                    
                                    with gr.Column():
                                        gr.Markdown("#### Vital Signs")
                                        covid_temp = gr.Slider(35, 42, value=37, label="Temperature (°C)")
                                        covid_hr = gr.Slider(40, 180, value=75, label="Heart Rate (bpm)")
                                        covid_rr = gr.Slider(8, 40, value=16, label="Respiratory Rate")
                                        covid_o2 = gr.Slider(70, 100, value=98, label="Oxygen Saturation (%)")
                                        covid_bp_sys = gr.Slider(80, 200, value=120, label="Systolic BP")
                                        covid_bp_dia = gr.Slider(40, 130, value=80, label="Diastolic BP")
                                    
                                    with gr.Column():
                                        gr.Markdown("#### Demographics & Labs")
                                        covid_age = gr.Slider(0, 120, value=30, label="Age")
                                        covid_sex = gr.Radio(["Male", "Female", "Other"], label="Sex", value="Male")
                                        covid_wbc = gr.Slider(1000, 20000, value=7000, label="WBC Count")
                                        covid_crp = gr.Slider(0, 200, value=5, label="CRP (mg/L)")
                                        covid_ferritin = gr.Slider(10, 1000, value=100, label="Ferritin (ng/mL)")
                                
                                covid_btn = gr.Button("🔍 Check COVID-19", variant="primary", size="lg")
                            
                            # JSON Input
                            with gr.Tab("📋 JSON Input"):
                                gr.Markdown("### Paste or edit patient data in JSON format")
                                covid_json_input = gr.Code(
                                    value="""{
  "symptoms": {
    "fever": true,
    "cough": true,
    "fatigue": false,
    "headache": false,
    "shortness_of_breath": false,
    "loss_of_taste": true,
    "sore_throat": false,
    "body_aches": false,
    "nausea": false,
    "vomiting": false,
    "diarrhea": false,
    "rash": false,
    "joint_pain": false,
    "retro_orbital_pain": false
  },
  "vitals": {
    "temperature": 38.5,
    "heart_rate": 85,
    "respiratory_rate": 18,
    "blood_pressure_systolic": 120,
    "blood_pressure_diastolic": 80,
    "oxygen_saturation": 96
  },
  "labs": {
    "wbc_count": 7000,
    "platelet_count": 250000,
    "hemoglobin": 14,
    "crp": 15,
    "ferritin": 200
  },
  "demographics": {
    "age": 35,
    "sex": "Male"
  }
}""",
                                    language="json",
                                    label="Patient Data (JSON)"
                                )
                                covid_json_btn = gr.Button("🔍 Check COVID-19 from JSON", variant="primary", size="lg")
                                covid_json_output = gr.Code(language="json", label="Processed Input")
                        
                        with gr.Row():
                            with gr.Column():
                                covid_results = gr.Markdown()
                            with gr.Column():
                                covid_confidence = gr.HTML()
                        covid_chart = gr.HTML()
                        
                        # Form button handler
                        covid_btn.click(
                            fn=lambda *args: app.diagnose_clinical(*args, disease_filter='COVID-19'),
                            inputs=[
                                covid_fever, covid_cough, covid_fatigue, covid_headache, covid_shortness,
                                covid_loss_taste, covid_sore_throat, covid_body_aches, 
                                gr.State(False), gr.State(False), covid_diarrhea, gr.State(False), 
                                gr.State(False), gr.State(False),
                                covid_temp, covid_hr, covid_rr, covid_bp_sys, covid_bp_dia, covid_o2,
                                covid_wbc, gr.State(250000), gr.State(14), covid_crp, covid_ferritin,
                                covid_age, covid_sex
                            ],
                            outputs=[covid_results, covid_chart, covid_confidence]
                        )
                        
                        # JSON button handler
                        covid_json_btn.click(
                            fn=lambda json_str: app.diagnose_from_json(json_str, disease_filter='COVID-19'),
                            inputs=[covid_json_input],
                            outputs=[covid_results, covid_chart, covid_confidence, covid_json_output]
                        )
                    
                    # Sub-tab 2: Dengue Check
                    with gr.Tab("🦟 Dengue Check"):
                        gr.Markdown("**Primary Symptoms:** High fever, severe headache, retro-orbital pain, joint/muscle pain, rash")
                        
                        with gr.Tabs():
                            # Form Input
                            with gr.Tab("📝 Form Input"):
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("#### Dengue Specific Symptoms")
                                        dengue_fever = gr.Checkbox(label="High Fever (>39°C)")
                                        dengue_headache = gr.Checkbox(label="Severe Headache")
                                        dengue_retro = gr.Checkbox(label="Retro-orbital Pain (Behind Eyes)")
                                        dengue_joint = gr.Checkbox(label="Severe Joint/Muscle Pain")
                                        dengue_rash = gr.Checkbox(label="Skin Rash")
                                        dengue_nausea = gr.Checkbox(label="Nausea/Vomiting")
                                        dengue_fatigue = gr.Checkbox(label="Fatigue")
                                        dengue_body_aches = gr.Checkbox(label="Body Aches")
                                    
                                    with gr.Column():
                                        gr.Markdown("#### Vital Signs")
                                        dengue_temp = gr.Slider(35, 42, value=38.5, label="Temperature (°C)")
                                        dengue_hr = gr.Slider(40, 180, value=75, label="Heart Rate (bpm)")
                                        dengue_rr = gr.Slider(8, 40, value=16, label="Respiratory Rate")
                                        dengue_bp_sys = gr.Slider(80, 200, value=110, label="Systolic BP")
                                        dengue_bp_dia = gr.Slider(40, 130, value=70, label="Diastolic BP")
                                        dengue_o2 = gr.Slider(70, 100, value=98, label="Oxygen Saturation (%)")
                                    
                                    with gr.Column():
                                        gr.Markdown("#### Demographics & Labs")
                                        dengue_age = gr.Slider(0, 120, value=30, label="Age")
                                        dengue_sex = gr.Radio(["Male", "Female", "Other"], label="Sex", value="Male")
                                        dengue_wbc = gr.Slider(1000, 20000, value=4000, label="WBC Count (often low)")
                                        dengue_platelet = gr.Slider(20000, 500000, value=100000, label="Platelet Count (often low)")
                                        dengue_hgb = gr.Slider(5, 20, value=14, label="Hemoglobin (g/dL)")
                                
                                dengue_btn = gr.Button("🔍 Check Dengue", variant="primary", size="lg")
                            
                            # JSON Input
                            with gr.Tab("📋 JSON Input"):
                                gr.Markdown("### Paste or edit patient data in JSON format")
                                dengue_json_input = gr.Code(
                                    value="""{
  "symptoms": {
    "fever": true,
    "cough": false,
    "fatigue": true,
    "headache": true,
    "shortness_of_breath": false,
    "loss_of_taste": false,
    "sore_throat": false,
    "body_aches": true,
    "nausea": true,
    "vomiting": false,
    "diarrhea": false,
    "rash": true,
    "joint_pain": true,
    "retro_orbital_pain": true
  },
  "vitals": {
    "temperature": 39.5,
    "heart_rate": 90,
    "respiratory_rate": 18,
    "blood_pressure_systolic": 110,
    "blood_pressure_diastolic": 70,
    "oxygen_saturation": 98
  },
  "labs": {
    "wbc_count": 4000,
    "platelet_count": 80000,
    "hemoglobin": 13,
    "crp": 10,
    "ferritin": 100
  },
  "demographics": {
    "age": 28,
    "sex": "Female"
  }
}""",
                                    language="json",
                                    label="Patient Data (JSON)"
                                )
                                dengue_json_btn = gr.Button("🔍 Check Dengue from JSON", variant="primary", size="lg")
                                dengue_json_output = gr.Code(language="json", label="Processed Input")
                        
                        with gr.Row():
                            with gr.Column():
                                dengue_results = gr.Markdown()
                            with gr.Column():
                                dengue_confidence = gr.HTML()
                        dengue_chart = gr.HTML()
                        
                        # Form button handler
                        dengue_btn.click(
                            fn=lambda *args: app.diagnose_clinical(*args, disease_filter='Dengue'),
                            inputs=[
                                dengue_fever, gr.State(False), dengue_fatigue, dengue_headache, gr.State(False),
                                gr.State(False), gr.State(False), dengue_body_aches, 
                                dengue_nausea, gr.State(False), gr.State(False), dengue_rash, 
                                dengue_joint, dengue_retro,
                                dengue_temp, dengue_hr, dengue_rr, dengue_bp_sys, dengue_bp_dia, dengue_o2,
                                dengue_wbc, dengue_platelet, dengue_hgb, gr.State(5), gr.State(100),
                                dengue_age, dengue_sex
                            ],
                            outputs=[dengue_results, dengue_chart, dengue_confidence]
                        )
                        
                        # JSON button handler
                        dengue_json_btn.click(
                            fn=lambda json_str: app.diagnose_from_json(json_str, disease_filter='Dengue'),
                            inputs=[dengue_json_input],
                            outputs=[dengue_results, dengue_chart, dengue_confidence, dengue_json_output]
                        )
                    
                    # Sub-tab 3: Pneumonia Check
                    with gr.Tab("🫁 Pneumonia Check"):
                        gr.Markdown("**Primary Symptoms:** Cough, fever, chest pain, difficulty breathing, fatigue")
                        
                        with gr.Tabs():
                            # Form Input
                            with gr.Tab("📝 Form Input"):
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("#### Pneumonia Specific Symptoms")
                                        pneumonia_cough = gr.Checkbox(label="Persistent Cough")
                                        pneumonia_fever = gr.Checkbox(label="Fever")
                                        pneumonia_shortness = gr.Checkbox(label="Shortness of Breath")
                                        pneumonia_fatigue = gr.Checkbox(label="Fatigue/Weakness")
                                        pneumonia_body_aches = gr.Checkbox(label="Chest Pain/Body Aches")
                                        pneumonia_headache = gr.Checkbox(label="Headache")
                                        pneumonia_nausea = gr.Checkbox(label="Nausea")
                                        pneumonia_sore_throat = gr.Checkbox(label="Sore Throat")
                                    
                                    with gr.Column():
                                        gr.Markdown("#### Vital Signs")
                                        pneumonia_temp = gr.Slider(35, 42, value=38, label="Temperature (°C)")
                                        pneumonia_hr = gr.Slider(40, 180, value=85, label="Heart Rate (bpm)")
                                        pneumonia_rr = gr.Slider(8, 40, value=22, label="Respiratory Rate (often elevated)")
                                        pneumonia_o2 = gr.Slider(70, 100, value=94, label="Oxygen Saturation (%)") 
                                        pneumonia_bp_sys = gr.Slider(80, 200, value=120, label="Systolic BP")
                                        pneumonia_bp_dia = gr.Slider(40, 130, value=80, label="Diastolic BP")
                                    
                                    with gr.Column():
                                        gr.Markdown("#### Demographics & Labs")
                                        pneumonia_age = gr.Slider(0, 120, value=30, label="Age")
                                        pneumonia_sex = gr.Radio(["Male", "Female", "Other"], label="Sex", value="Male")
                                        pneumonia_wbc = gr.Slider(1000, 20000, value=12000, label="WBC Count (often high)")
                                        pneumonia_crp = gr.Slider(0, 200, value=50, label="CRP (mg/L) - often elevated")
                                        pneumonia_hgb = gr.Slider(5, 20, value=14, label="Hemoglobin (g/dL)")
                                
                                pneumonia_btn = gr.Button("🔍 Check Pneumonia", variant="primary", size="lg")
                            
                            # JSON Input
                            with gr.Tab("📋 JSON Input"):
                                gr.Markdown("### Paste or edit patient data in JSON format")
                                pneumonia_json_input = gr.Code(
                                    value="""{
  "symptoms": {
    "fever": true,
    "cough": true,
    "fatigue": true,
    "headache": true,
    "shortness_of_breath": true,
    "loss_of_taste": false,
    "sore_throat": false,
    "body_aches": true,
    "nausea": false,
    "vomiting": false,
    "diarrhea": false,
    "rash": false,
    "joint_pain": false,
    "retro_orbital_pain": false
  },
  "vitals": {
    "temperature": 38.5,
    "heart_rate": 95,
    "respiratory_rate": 24,
    "blood_pressure_systolic": 125,
    "blood_pressure_diastolic": 80,
    "oxygen_saturation": 92
  },
  "labs": {
    "wbc_count": 13000,
    "platelet_count": 250000,
    "hemoglobin": 13.5,
    "crp": 75,
    "ferritin": 150
  },
  "demographics": {
    "age": 45,
    "sex": "Male"
  }
}""",
                                    language="json",
                                    label="Patient Data (JSON)"
                                )
                                pneumonia_json_btn = gr.Button("🔍 Check Pneumonia from JSON", variant="primary", size="lg")
                                pneumonia_json_output = gr.Code(language="json", label="Processed Input")
                        
                        with gr.Row():
                            with gr.Column():
                                pneumonia_results = gr.Markdown()
                            with gr.Column():
                                pneumonia_confidence = gr.HTML()
                        pneumonia_chart = gr.HTML()
                        
                        # Form button handler
                        pneumonia_btn.click(
                            fn=lambda *args: app.diagnose_clinical(*args, disease_filter='Pneumonia'),
                            inputs=[
                                pneumonia_fever, pneumonia_cough, pneumonia_fatigue, pneumonia_headache, 
                                pneumonia_shortness, gr.State(False), pneumonia_sore_throat, pneumonia_body_aches, 
                                pneumonia_nausea, gr.State(False), gr.State(False), gr.State(False), 
                                gr.State(False), gr.State(False),
                                pneumonia_temp, pneumonia_hr, pneumonia_rr, pneumonia_bp_sys, 
                                pneumonia_bp_dia, pneumonia_o2,
                                pneumonia_wbc, gr.State(250000), pneumonia_hgb, pneumonia_crp, gr.State(100),
                                pneumonia_age, pneumonia_sex
                            ],
                            outputs=[pneumonia_results, pneumonia_chart, pneumonia_confidence]
                        )
                        
                        # JSON button handler
                        pneumonia_json_btn.click(
                            fn=lambda json_str: app.diagnose_from_json(json_str, disease_filter='Pneumonia'),
                            inputs=[pneumonia_json_input],
                            outputs=[pneumonia_results, pneumonia_chart, pneumonia_confidence, pneumonia_json_output]
                        )
            
            # Tab 2: Skin Lesion Analysis
            with gr.Tab("🔬 Skin Lesion Analysis"):
                gr.Markdown("""
                ### Upload a dermoscopic image for skin lesion analysis
                
                The CNN model can identify:
                - **Melanoma**: Skin cancer, moles, and nevi
                - **Eczema**: Atopic dermatitis and related conditions
                - **Psoriasis**: Including lichen planus and related diseases
                - **Acne**: Including rosacea and acne vulgaris
                - **Normal/Healthy Skin**: Detects healthy skin to avoid false positives
                
                Model: EfficientNet-B0 trained on 5 disease categories including normal skin.
                Dataset: ~819MB unified skin disease dataset with balanced classes.
                """)
                
                with gr.Row():
                    with gr.Column(scale=6):
                        skin_image = gr.Image(type="pil", label="Upload Skin Lesion Image", elem_id="skin-upload")
                        analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

                    with gr.Column(scale=6):
                        skin_confidence = gr.HTML()
                        skin_results = gr.Markdown()
                        explain_mode = gr.Radio(
                            choices=["None", "Fast (saliency)", "SHAP (slow)", "Textual"],
                            label="Explanation",
                            value="Textual"
                        )

                skin_chart = gr.HTML()
                shap_output = gr.HTML(label="Explanation")

                analyze_btn.click(
                    fn=app.diagnose_skin_lesion,
                    inputs=[skin_image, explain_mode],
                    outputs=[skin_results, skin_chart, skin_confidence, shap_output]
                )
            
            # Tab 3: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This System
                
                This **Hybrid Neuro-Symbolic Clinical Decision Support System** combines three 
                complementary AI approaches:
                
                ### 🧠 Components
                
                1. **Rule-Based Engine** (30% weight)
                   - Expert-defined clinical rules
                   - Forward chaining inference
                   - Explainable reasoning
                
                2. **Random Forest Classifier** (50% weight)
                   - Trained on clinical data (symptoms, vitals, labs)
                   - Handles 5 diseases: COVID-19, Dengue, Malaria, Pneumonia, Flu
                   - 99%+ accuracy
                
                3. **Convolutional Neural Network** (20% weight)
                   - EfficientNet-B0 architecture
                   - Trained on unified skin disease dataset (~819MB)
                   - 5 classes: Melanoma, Eczema, Psoriasis, Acne, Normal/Healthy Skin
                   - Transfer learning from ImageNet
                   - Trained on TPU v5e-1 / GPU for optimal performance
                
                ### 📊 Fusion Strategy
                
                The system uses **weighted voting** to combine predictions from all three 
                components, providing more robust and reliable diagnoses than any single method.
                
                ### ⚠️ Disclaimer
                
                This system is designed for **educational and research purposes only**. 
                It should not replace professional medical advice, diagnosis, or treatment. 
                Always consult qualified healthcare professionals for medical decisions.
                
                ### 🔧 Technical Details
                
                - **Framework**: PyTorch, scikit-learn, Gradio
                - **Dataset**: Unified Kaggle skin disease dataset (mateenzahid/skin-diesease)
                - **Models**: Trained with cross-validation and TPU/GPU acceleration
                - **Last Updated**: April 2026
                """)
        
        gr.Markdown("""
        ---
        <div style='text-align: center; color: #7f8c8d;'>
        <p>Hybrid Neuro-Symbolic Clinical Decision Support System | 
        Built with PyTorch, scikit-learn, and Gradio</p>
        </div>
        """)
    
    return interface


def main():
    """Launch the Gradio interface"""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )


if __name__ == "__main__":
    main()
