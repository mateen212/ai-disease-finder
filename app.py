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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.rule_engine import RuleEngine
from src.ml_models import RandomForestDiagnostic, SkinLesionCNN
from src.data_preprocessing import DataPreprocessor
from src.hybrid_system import HybridDiagnosticSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalDiagnosisApp:
    """Main application class for the diagnosis system"""
    
    def __init__(self):
        """Initialize all models and components"""
        logger.info("Initializing Clinical Diagnosis System...")
        
        # Skin lesion class names (define BEFORE loading models)
        self.skin_classes = [
            'Melanoma', 'Melanocytic Nevus', 'Basal Cell Carcinoma',
            'Actinic Keratosis', 'Benign Keratosis',
            'Dermatofibroma', 'Vascular Lesion'
        ]
        
        # Initialize components
        self.rule_engine = RuleEngine()
        self.rf_model = RandomForestDiagnostic()
        self.cnn_model = SkinLesionCNN()
        self.preprocessor = DataPreprocessor()
        self.hybrid_system = HybridDiagnosticSystem()
        
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
        
        # Load CNN
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
        age: int, sex: str
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
            
            # Format results
            results_text = self._format_diagnosis_results(diagnosis_result)
            chart_html = self._create_probability_chart(diagnosis_result['probabilities'])
            confidence_text = self._format_confidence(diagnosis_result)
            
            return results_text, chart_html, confidence_text
            
        except Exception as e:
            logger.error(f"Diagnosis error: {e}")
            return f"Error: {str(e)}", "", ""
    
    def diagnose_skin_lesion(self, image: Optional[Image.Image]) -> Tuple[str, str, str]:
        """
        Diagnose skin lesion from uploaded image
        
        Returns:
            Tuple of (results_text, chart_html, confidence_text)
        """
        if image is None:
            return "Please upload an image", "", ""
        
        try:
            # Preprocess image
            image_tensor = self.preprocessor.test_transform(image)
            
            # Get CNN predictions
            probabilities = self.cnn_model.predict_proba(image_tensor)
            
            # Find top prediction
            top_class = max(probabilities.items(), key=lambda x: x[1])
            
            # Format results
            results_text = f"## 🔬 Skin Lesion Analysis\n\n"
            results_text += f"**Primary Diagnosis:** {top_class[0]}\n\n"
            results_text += f"**Confidence:** {top_class[1]*100:.1f}%\n\n"
            results_text += "### All Probabilities:\n"
            
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for disease, prob in sorted_probs:
                bar = "█" * int(prob * 20)
                results_text += f"- **{disease}**: {prob*100:.1f}% {bar}\n"
            
            # Create chart
            chart_html = self._create_probability_chart(probabilities)
            
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
            
            return results_text, chart_html, confidence_text
            
        except Exception as e:
            logger.error(f"Skin lesion diagnosis error: {e}")
            return f"Error: {str(e)}", "", ""
    
    def _format_diagnosis_results(self, result: Dict[str, Any]) -> str:
        """Format diagnosis results as markdown"""
        text = "## 🏥 Diagnosis Results\n\n"
        
        # Primary diagnosis
        text += f"### Primary Diagnosis: **{result['diagnosis']}**\n\n"
        text += f"**Overall Confidence:** {result['confidence']*100:.1f}%\n\n"
        
        # Component predictions
        text += "### Component Model Predictions:\n\n"
        
        for component, data in result['component_predictions'].items():
            text += f"#### {component}\n"
            if 'top_disease' in data:
                text += f"- Prediction: **{data['top_disease']}**\n"
                text += f"- Confidence: {data['top_score']*100:.1f}%\n"
            text += "\n"
        
        # Top probabilities
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


def create_interface():
    """Create Gradio interface"""
    app = ClinicalDiagnosisApp()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-markdown h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Clinical Diagnosis System") as interface:
        gr.Markdown("""
        # 🏥 Hybrid Neuro-Symbolic Clinical Decision Support System
        
        This system combines **Rule-Based Reasoning**, **Machine Learning (Random Forest)**, 
        and **Deep Learning (CNN)** to provide accurate disease diagnosis.
        
        ### Select a tab below to start:
        """)
        
        with gr.Tabs():
            # Tab 1: Clinical Diagnosis
            with gr.Tab("🩺 Clinical Diagnosis"):
                gr.Markdown("### Enter patient information for clinical diagnosis")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Symptoms")
                        fever = gr.Checkbox(label="Fever")
                        cough = gr.Checkbox(label="Cough")
                        fatigue = gr.Checkbox(label="Fatigue")
                        headache = gr.Checkbox(label="Headache")
                        shortness_of_breath = gr.Checkbox(label="Shortness of Breath")
                        loss_of_taste = gr.Checkbox(label="Loss of Taste/Smell")
                        sore_throat = gr.Checkbox(label="Sore Throat")
                        body_aches = gr.Checkbox(label="Body Aches")
                        nausea = gr.Checkbox(label="Nausea")
                        vomiting = gr.Checkbox(label="Vomiting")
                        diarrhea = gr.Checkbox(label="Diarrhea")
                        rash = gr.Checkbox(label="Rash")
                        joint_pain = gr.Checkbox(label="Joint Pain")
                        retro_orbital_pain = gr.Checkbox(label="Retro-orbital Pain")
                    
                    with gr.Column():
                        gr.Markdown("#### Vital Signs")
                        temperature = gr.Slider(35, 42, value=37, label="Temperature (°C)")
                        heart_rate = gr.Slider(40, 180, value=75, label="Heart Rate (bpm)")
                        respiratory_rate = gr.Slider(8, 40, value=16, label="Respiratory Rate")
                        bp_sys = gr.Slider(80, 200, value=120, label="Systolic BP (mmHg)")
                        bp_dia = gr.Slider(40, 130, value=80, label="Diastolic BP (mmHg)")
                        oxygen_sat = gr.Slider(70, 100, value=98, label="Oxygen Saturation (%)")
                        
                        gr.Markdown("#### Demographics")
                        age = gr.Slider(0, 120, value=30, label="Age")
                        sex = gr.Radio(["Male", "Female", "Other"], label="Sex", value="Male")
                    
                    with gr.Column():
                        gr.Markdown("#### Laboratory Values")
                        wbc = gr.Slider(1000, 20000, value=7000, label="WBC Count (cells/μL)")
                        platelet = gr.Slider(20000, 500000, value=250000, label="Platelet Count (cells/μL)")
                        hemoglobin = gr.Slider(5, 20, value=14, label="Hemoglobin (g/dL)")
                        crp = gr.Slider(0, 200, value=5, label="CRP (mg/L)")
                        ferritin = gr.Slider(10, 1000, value=100, label="Ferritin (ng/mL)")
                
                diagnose_btn = gr.Button("🔍 Diagnose", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        clinical_results = gr.Markdown()
                    with gr.Column():
                        clinical_confidence = gr.HTML()
                
                clinical_chart = gr.HTML()
                
                diagnose_btn.click(
                    fn=app.diagnose_clinical,
                    inputs=[
                        fever, cough, fatigue, headache, shortness_of_breath,
                        loss_of_taste, sore_throat, body_aches, nausea, vomiting,
                        diarrhea, rash, joint_pain, retro_orbital_pain,
                        temperature, heart_rate, respiratory_rate, bp_sys, bp_dia,
                        oxygen_sat, wbc, platelet, hemoglobin, crp, ferritin,
                        age, sex
                    ],
                    outputs=[clinical_results, clinical_chart, clinical_confidence]
                )
            
            # Tab 2: Skin Lesion Analysis
            with gr.Tab("🔬 Skin Lesion Analysis"):
                gr.Markdown("""
                ### Upload a dermoscopic image for skin lesion analysis
                
                The CNN model can identify:
                - Melanoma
                - Melanocytic Nevus
                - Basal Cell Carcinoma
                - Actinic Keratosis
                - Benign Keratosis
                - Dermatofibroma
                - Vascular Lesion
                """)
                
                with gr.Row():
                    with gr.Column():
                        skin_image = gr.Image(type="pil", label="Upload Skin Lesion Image")
                        analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                    
                    with gr.Column():
                        skin_confidence = gr.HTML()
                        skin_results = gr.Markdown()
                
                skin_chart = gr.HTML()
                
                analyze_btn.click(
                    fn=app.diagnose_skin_lesion,
                    inputs=[skin_image],
                    outputs=[skin_results, skin_chart, skin_confidence]
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
                   - Trained on 10,015 dermoscopic images
                   - Transfer learning from ImageNet
                   - 85%+ accuracy on skin lesions
                
                ### 📊 Fusion Strategy
                
                The system uses **weighted voting** to combine predictions from all three 
                components, providing more robust and reliable diagnoses than any single method.
                
                ### ⚠️ Disclaimer
                
                This system is designed for **educational and research purposes only**. 
                It should not replace professional medical advice, diagnosis, or treatment. 
                Always consult qualified healthcare professionals for medical decisions.
                
                ### 🔧 Technical Details
                
                - **Framework**: PyTorch, scikit-learn, Gradio
                - **Dataset**: HAM10000, synthetic clinical data
                - **Models**: Trained with cross-validation
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
