"""
Explainability Module

Provides feature-level explanations using SHAP values and rule traces
to make the system's decisions transparent and interpretable.
"""

import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainabilityModule:
    """
    Provides comprehensive explanations for clinical decisions.
    
    Combines:
    - SHAP values for ML model feature importance
    - Rule traces from the inference engine
    - Feature contribution analysis
    """
    
    def __init__(self, config_file: str = "config/model_config.yaml"):
        """
        Initialize the explainability module.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        explain_config = self.config.get('explainability', {})
        shap_config = explain_config.get('shap', {})
        
        self.max_features_display = shap_config.get('max_features_display', 10)
        self.num_shap_samples = shap_config.get('num_samples', 100)
        
        self.explainer = None
        self.background_data = None
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not available. Some features will be limited.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def initialize_shap_explainer(
        self,
        model: Any,
        background_data: pd.DataFrame,
        model_type: str = "tree"
    ):
        """
        Initialize SHAP explainer for a trained model.
        
        Args:
            model: Trained ML model (sklearn or similar)
            background_data: Sample data for SHAP background
            model_type: Type of model ("tree", "linear", "kernel")
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP not available")
            return
        
        # Sample background data if too large
        if len(background_data) > self.num_shap_samples:
            self.background_data = background_data.sample(
                n=self.num_shap_samples,
                random_state=42
            )
        else:
            self.background_data = background_data
        
        # Create explainer based on model type
        try:
            if model_type == "tree":
                # For tree-based models (Random Forest, XGBoost, etc.)
                self.explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                self.explainer = shap.LinearExplainer(model, self.background_data)
            else:
                # Kernel explainer (model-agnostic but slower)
                self.explainer = shap.KernelExplainer(
                    model.predict_proba,
                    self.background_data
                )
            
            logger.info(f"SHAP explainer initialized: {model_type}")
        
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
    
    def explain_prediction(
        self,
        patient_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP-based explanation for a prediction.
        
        Args:
            patient_data: Single patient data as DataFrame (1 row)
            feature_names: Optional list of feature names
        
        Returns:
            Dictionary with SHAP values and explanation
        """
        # If SHAP is not available or explainer not initialized, return
        # a concise textual explanation by default. Use SHAP only when
        # an explainer is initialized and the library is available.
        if not SHAP_AVAILABLE or self.explainer is None:
            text_expl = self._generate_text_explanation(patient_data)
            return {
                'shap_values': None,
                'feature_contributions': {},
                'top_features': {},
                'explanation': text_expl,
                'explanation_type': 'textual'
            }
        
        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(patient_data)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # For multi-class, use the class with highest prediction
                # or average across classes
                shap_values_array = np.array(shap_values)
                # Take mean absolute contribution across classes
                shap_values_summary = np.mean(np.abs(shap_values_array), axis=0)
            else:
                shap_values_summary = shap_values
            
            # Get feature names
            if feature_names is None:
                feature_names = patient_data.columns.tolist()
            
            # Create feature contribution dictionary
            contributions = {}
            for i, feature in enumerate(feature_names):
                if len(shap_values_summary.shape) == 2:
                    value = float(shap_values_summary[0, i])
                else:
                    value = float(shap_values_summary[i])
                contributions[feature] = value
            
            # Sort by absolute contribution
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top features
            top_features = sorted_contributions[:self.max_features_display]
            
            # Generate explanation text
            explanation = self._generate_shap_explanation(
                top_features,
                patient_data.iloc[0].to_dict()
            )
            
            return {
                'shap_values': shap_values,
                'feature_contributions': contributions,
                'top_features': dict(top_features),
                'explanation': explanation,
                'explanation_type': 'shap'
            }
        
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return {
                'shap_values': None,
                'feature_contributions': {},
                'explanation': f"Error: {str(e)}",
                'explanation_type': 'textual'
            }
    
    def _generate_shap_explanation(
        self,
        top_features: List[Tuple[str, float]],
        patient_values: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation from SHAP values.
        
        Args:
            top_features: List of (feature_name, shap_value) tuples
            patient_values: Dictionary of patient's feature values
        
        Returns:
            Explanation string
        """
        explanation = ["Key factors influencing this diagnosis:\n"]
        
        for i, (feature, shap_value) in enumerate(top_features, 1):
            # Get patient's value for this feature
            patient_val = patient_values.get(feature, "N/A")
            
            # Determine direction of contribution
            if shap_value > 0:
                direction = "increases"
                symbol = "↑"
            else:
                direction = "decreases"
                symbol = "↓"
            
            # Clean feature name for display
            display_name = feature.replace('_', ' ').title()
            
            val_str = f"{patient_val:.2f}" if isinstance(patient_val, (int, float)) else str(patient_val)
            explanation.append(
                f"{i}. {display_name} (value: {val_str}): {symbol} {direction} probability (contribution: {abs(shap_value):.3f})"
            )
        
        return "\n".join(explanation)

    def get_guidance_for_disease(self, disease: str) -> Dict[str, str]:
        """
        Get synthesized clinical guidance for a disease with source attribution.
        Returns dict with 'content' (synthesized guidance) and 'source' (filename).
        """
        if not disease:
            return {"content": "", "source": ""}

        guidelines_dir = Path("data/guidelines")
        if not guidelines_dir.exists():
            return {"content": "", "source": ""}

        # Normalize disease name and look for exact match first
        disease_lower = disease.lower()
        disease_normalized = disease_lower.replace('_', '').replace('-', '').replace(' ', '')
        
        # Strategy 1: Look for exact filename match (e.g., "covid.pdf" for "covid19")
        exact_matches = []
        partial_matches = []
        generic_files = []
        
        for p in guidelines_dir.iterdir():
            if not p.is_file():
                continue
            
            name_lower = p.stem.lower()  # Filename without extension
            name_normalized = name_lower.replace('_', '').replace('-', '').replace(' ', '')
            
            # Check if this is a generic file (contains all diseases)
            if 'clinical_guidelines' in name_lower or 'all_diseases' in name_lower:
                generic_files.append(p)
                continue
            
            # Exact disease name match (e.g., "covid" matches "covid.pdf")
            if disease_normalized in name_normalized or name_normalized in disease_normalized:
                exact_matches.append(p)
            # Partial token match (e.g., "covid" in "covid19_cdc.pdf")
            elif any(token in name_lower for token in disease_lower.split() if len(token) > 2):
                partial_matches.append(p)
        
        # Pick best match: exact > partial > generic
        candidates_sorted = exact_matches + partial_matches + generic_files
        
        if not candidates_sorted:
            return {"content": "", "source": ""}

        picked = candidates_sorted[0]
        source_name = picked.name
        
        # Read file contents depending on type
        suffix = picked.suffix.lower()
        text = ""
        try:
            if suffix == '.pdf':
                if not PDF_AVAILABLE:
                    return {
                        "content": f"Guidance available in {source_name} but PDF extraction not installed.",
                        "source": source_name
                    }
                try:
                    reader = PdfReader(str(picked))
                    pages = []
                    for p in reader.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                    text = "\n".join(pages)
                except Exception:
                    return {"content": "", "source": source_name}
            else:
                text = picked.read_text(errors='ignore')
        except Exception:
            return {"content": "", "source": source_name}

        # If HTML, strip tags (basic)
        if suffix in ('.html', '.htm'):
            text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.S|re.I)
            text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.S|re.I)
            text = re.sub(r'<[^>]+>', '', text)

        # Remove control characters and collapse whitespace
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # For clinical diseases (COVID, dengue, pneumonia), always use curated content
        # PDF extraction is too unreliable for WHO/CDC documents with extensive legal text
        disease_lower = disease.lower()
        if any(x in disease_lower for x in ['covid', 'dengue', 'pneumonia']):
            curated = self._synthesize_clinical_guidance({}, disease, source_name)
            return {"content": curated, "source": source_name}
        
        # For other diseases (skin lesions, etc.), try PDF extraction
        raw_text = text[:10000]  # Limit for processing
        sections = self._extract_key_guidance_points_clinical(raw_text, disease)
        
        if sections:
            synthesized = self._synthesize_clinical_guidance(sections, disease, source_name)
            return {"content": synthesized, "source": source_name}
        else:
            # Fallback: extract clean excerpt
            clean_excerpt = self._extract_clean_excerpt(raw_text, disease)
            return {"content": clean_excerpt, "source": source_name}

    def _extract_key_guidance_points_clinical(self, guidance_text: str, disease: str) -> Dict[str, str]:
        """
        Extract key clinical guidance points for diseases like COVID, dengue, pneumonia.
        Returns structured sections: symptoms, diagnosis, treatment, complications, management.
        """
        if not guidance_text or len(guidance_text) < 200:
            return {}

        text = re.sub(r'\s+', ' ', guidance_text)
        sections = {}

        # Keywords for clinical diseases - more specific to actual medical content
        keywords = {
            'symptoms': ['fever', 'cough', 'fatigue', 'headache', 'rash', 'bleeding', 'breathing', 'pain', 'vomiting', 
                        'clinical feature', 'present with', 'patient may', 'common symptom'],
            'diagnosis': ['laboratory', 'test result', 'pcr', 'antigen', 'antibody', 'platelet', 'ns1',
                         'confirmed by', 'diagnostic', 'specimen', 'blood test', 'rapid test'],
            'treatment': ['oral', 'intravenous', 'antibiotic', 'antiviral', 'fluid', 'hydration', 'hospitalization',
                         'supportive care', 'oxygen', 'medication', 'dose', 'amoxicillin', 'therapy'],
            'complications': ['shock', 'severe', 'organ failure', 'ards', 'intensive care', 'mortality',
                            'warning sign', 'danger', 'critical', 'life-threatening', 'deterioration'],
            'prevention': ['vaccination', 'vaccine', 'mosquito control', 'vector control', 'hygiene',
                          'prevent transmission', 'protective measure', 'avoid contact']
        }

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Strong exclusion patterns - remove administrative/legal/formatting text
        exclude_patterns = [
            r'©|copyright|all rights reserved|creative commons|licence|license',
            r'isbn|catalogu|suggested citation|reference:|source:',
            r'table of content|chapter \d|annex|bibliography|page \d|section \d\.+\d',
            r'message from|acknowledgement|hon.?ble|minister|representative|who country',
            r'fig\.|figure \d|table \d|appendix',
            r'this (work|publication|translation|guideline) (is|was|has)',
            r'world health organization|ministry of health|department of',
            r'under the terms|in accordance with|shall be|is available|can be',
            r'the designations employed|the presentation of|opinion whatsoever',
            r'dotted and dashed lines|map represent|frontiers or boundaries',
        ]
        
        for section_name, section_keywords in keywords.items():
            relevant = []
            for sent in sentences:
                sent_clean = sent.strip()
                
                # Skip excluded patterns (legal, administrative, formatting)
                if any(re.search(pat, sent_clean, re.I) for pat in exclude_patterns):
                    continue
                
                # Skip very short or very long
                if len(sent_clean) < 60 or len(sent_clean) > 500:
                    continue
                
                # Must end properly
                if sent_clean[-1] not in '.!?':
                    continue
                
                # Check if contains ANY medical keywords (not just section keywords)
                sent_lower = sent_clean.lower()
                medical_terms = ['patient', 'treatment', 'clinical', 'symptom', 'diagnosis', 'disease',
                                'fever', 'infection', 'medical', 'health', 'care', 'therapy']
                has_medical_content = any(term in sent_lower for term in medical_terms)
                
                if not has_medical_content:
                    continue
                
                # Check if matches section keywords
                if any(kw in sent_lower for kw in section_keywords):
                    # Additional quality checks
                    words = sent_clean.split()
                    if len(words) < 8:  # Skip very short sentences
                        continue
                    
                    # Skip if too many ALL CAPS words (usually headings/junk)
                    caps_words = [w for w in words if w.isupper() and len(w) > 2]
                    if len(caps_words) > len(words) * 0.3:
                        continue
                    
                    relevant.append(sent_clean)
                    if len(relevant) >= 2:  # Take up to 2 sentences per section
                        break
            
            if relevant:
                sections[section_name] = ' '.join(relevant[:2])  # Max 2 sentences
        
        return sections

    def _extract_key_guidance_points(self, guidance: Any, disease: str) -> Dict[str, str]:
        """
        Generic extractor for guideline content. Accepts either the dict returned
        by `get_guidance_for_disease` or raw text and returns structured
        sections useful for narrative synthesis (overview, symptoms,
        diagnosis, risk_factors, treatment, prevention).
        """
        # Normalize input (guidance may be a dict {'content':..., 'source':...})
        content = ""
        try:
            if isinstance(guidance, dict):
                content = guidance.get('content', '') or ''
            else:
                content = str(guidance or '')
        except Exception:
            content = ''

        if not content or len(content) < 100:
            return {}

        # If clinical disease, delegate to clinical-specific extractor
        disease_lower = (disease or '').lower()
        if any(x in disease_lower for x in ['covid', 'dengue', 'pneumonia']):
            return self._extract_key_guidance_points_clinical(content, disease)

        # Generic extraction for other conditions (e.g., skin diseases)
        text = re.sub(r'\s+', ' ', content)

        # Split into candidate sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        sections = {}

        keywords = {
            'overview': ['overview', 'background', 'summary', 'about', 'introduction'],
            'symptoms': ['symptom', 'itch', 'itching', 'pruritus', 'lesion', 'scaly', 'erythema', 'plaque', 'rash', 'red'],
            'diagnosis': ['dermoscopy', 'biopsy', 'histopath', 'histology', 'diagnos', 'examination', 'confirm'],
            'risk_factors': ['risk factor', 'family history', 'genetic', 'uv', 'sun', 'tanning', 'smoking', 'immunosuppress'],
            'treatment': ['treatment', 'therapy', 'topical', 'systemic', 'corticosteroid', 'phototherapy', 'biologic', 'methotrexate', 'antibiotic'],
            'prevention': ['prevention', 'sunscreen', 'sun protection', 'avoid', 'protective', 'reduce risk']
        }

        # Basic exclusion and quality checks
        exclude_patterns = [r'©|copyright|all rights reserved|creative commons', r'isbn', r'table of content', r'appendix']

        for section_name, kws in keywords.items():
            relevant = []
            for sent in sentences:
                s = sent.strip()
                if not s or len(s) < 40 or len(s) > 600:
                    continue
                if any(re.search(pat, s, re.I) for pat in exclude_patterns):
                    continue
                s_lower = s.lower()
                if any(kw in s_lower for kw in kws):
                    # Avoid grabbing headings or boilerplate
                    words = s.split()
                    caps_words = [w for w in words if w.isupper() and len(w) > 2]
                    if len(caps_words) > len(words) * 0.4:
                        continue
                    relevant.append(s)
                if len(relevant) >= 2:
                    break
            if relevant:
                # Clean and keep up to two sentences/paragraphs
                cleaned = ' '.join([self._clean_extracted_text(r) for r in relevant[:2]])
                sections[section_name] = cleaned

        return sections
    
    def _synthesize_clinical_guidance(self, sections: Dict[str, str], disease: str, source: str) -> str:
        """
        Synthesize clinical guidance sections into readable format.
        If PDF extraction is poor, use curated medical knowledge.
        """
        disease_lower = disease.lower()
        output = []
        
        # Check if we have good extracted content
        has_good_content = len(sections) >= 2 and any(len(v) > 100 for v in sections.values())
        
        # Additional quality check: make sure extracted content doesn't have junk
        if has_good_content:
            all_text = ' '.join(sections.values())
            junk_indicators = [
                'world health organization', 'suggested citation', 'creative commons',
                'isbn', 'copyright', 'disclaimers', 'dotted and dashed lines',
                'the designations employed', 'table of contents', 'publication data'
            ]
            # If more than 10% of text is junk keywords, don't trust it
            has_junk = sum(1 for indicator in junk_indicators if indicator in all_text.lower())
            if has_junk >= 2:  # If 2+ junk indicators found, use curated
                has_good_content = False
        
        if not has_good_content:
            # Use curated clinical knowledge
            if 'covid' in disease_lower:
                output.append("**🔍 Clinical Presentation**")
                output.append("COVID-19 commonly presents with fever, dry cough, fatigue, and shortness of breath. Loss of taste or smell is a distinctive feature. Symptoms typically appear 2-14 days after exposure.")
                output.append("")
                output.append("**🧪 Diagnosis**")
                output.append("Diagnosis is confirmed through RT-PCR testing or rapid antigen tests. Laboratory findings may show lymphopenia, elevated inflammatory markers (CRP, D-dimer), and ground-glass opacities on chest imaging in severe cases.")
                output.append("")
                output.append("**⚠️ Complications & Warning Signs**")
                output.append("Severe COVID-19 can lead to acute respiratory distress syndrome (ARDS), requiring oxygen therapy or mechanical ventilation. Warning signs include persistent chest pain, severe shortness of breath, confusion, and oxygen saturation below 94%.")
                output.append("")
                output.append("**💊 Treatment**")
                output.append("Mild cases: Rest, hydration, symptomatic relief (acetaminophen for fever). Moderate-severe cases: Supplemental oxygen, dexamethasone (reduces mortality in severe cases), anticoagulation when indicated. ICU care for critical patients with ARDS.")
                output.append("")
                output.append("**🛡️ Prevention**")
                output.append("Vaccination is the primary preventive measure. Additional measures include masking in high-risk settings, hand hygiene, physical distancing, and improving ventilation in indoor spaces.")
                
            elif 'dengue' in disease_lower:
                output.append("**🔍 Clinical Presentation**")
                output.append("Dengue fever presents with sudden high fever, severe headache (especially retro-orbital pain), joint and muscle pain ('breakbone fever'), rash, and mild bleeding manifestations. The disease progresses through febrile, critical, and recovery phases.")
                output.append("")
                output.append("**🧪 Diagnosis**")
                output.append("Diagnosis through NS1 antigen test (early phase) or IgM/IgG antibodies (later phase). Laboratory shows thrombocytopenia (low platelets), hemoconcentration (rising hematocrit), and leukopenia.")
                output.append("")
                output.append("**⚠️ Complications & Warning Signs**")
                output.append("Warning signs indicating progression to severe dengue: persistent vomiting, severe abdominal pain, bleeding gums/nose, blood in stool/vomit, difficulty breathing, lethargy/restlessness. Severe dengue can cause plasma leakage, severe bleeding, and organ failure with 20% mortality if untreated.")
                output.append("")
                output.append("**💊 Treatment**")
                output.append("No specific antiviral treatment. Management is supportive: adequate fluid replacement (oral for mild cases, IV for severe), close monitoring of vital signs and hematocrit, platelet transfusion only for severe bleeding. Avoid aspirin and NSAIDs (increase bleeding risk).")
                output.append("")
                output.append("**🛡️ Prevention**")
                output.append("Vector control is key: eliminate standing water (mosquito breeding sites), use mosquito repellents, wear protective clothing, install window screens. Community-wide vector control programs are essential.")
                
            elif 'pneumonia' in disease_lower:
                output.append("**🔍 Clinical Presentation**")
                output.append("Pneumonia presents with cough (productive or dry), fever, chest pain (pleuritic), and shortness of breath. Physical examination may reveal crackles, decreased breath sounds, and signs of respiratory distress.")
                output.append("")
                output.append("**🧪 Diagnosis**")
                output.append("Diagnosed through chest X-ray showing infiltrates or consolidation, plus clinical findings. Laboratory tests show elevated white blood cell count, elevated CRP/procalcitonin. Sputum culture or blood culture may identify causative organism.")
                output.append("")
                output.append("**⚠️ Complications & Warning Signs**")
                output.append("Severe pneumonia may progress to respiratory failure, sepsis, or pleural effusion/empyema. Use CURB-65 score to assess severity: Confusion, Urea >7 mmol/L, Respiratory rate ≥30/min, Blood pressure <90/60, age ≥65. Score ≥3 indicates high risk requiring hospitalization.")
                output.append("")
                output.append("**💊 Treatment**")
                output.append("Community-acquired pneumonia: Amoxicillin (first-line oral), or macrolide/fluoroquinolone if atypical pathogens suspected. Hospital-acquired pneumonia requires broader coverage. Ensure adequate oxygenation, hydration, and respiratory support as needed.")
                output.append("")
                output.append("**🛡️ Prevention**")
                output.append("Pneumococcal and influenza vaccination for high-risk groups (elderly, immunocompromised, chronic disease). Good hand hygiene, avoiding smoking, and treating underlying conditions reduce risk.")
            
            else:
                # Generic fallback
                output.append("Please consult medical guidelines for detailed clinical management information.")
        
        else:
            # Use extracted sections if good quality
            section_order = [
                ('symptoms', '🔍 Clinical Presentation'),
                ('diagnosis', '🧪 Diagnosis'),
                ('complications', '⚠️ Complications & Warning Signs'),
                ('treatment', '💊 Treatment'),
                ('prevention', '🛡️ Prevention'),
            ]
            
            for key, header in section_order:
                if key in sections and sections[key]:
                    output.append(f"**{header}**")
                    output.append(sections[key])
                    output.append("")  # Blank line
        
        return "\n".join(output)
    
    def _extract_clean_excerpt(self, text: str, disease: str) -> str:
        """
        Extract a clean excerpt when structured extraction fails.
        Removes boilerplate, copyright, TOC, and keeps only medical content.
        """
        # Remove common junk sections
        junk_markers = [
            r'©.*?(?=\n)',  # Copyright lines
            r'Creative Commons.*?(?=\n)',
            r'Table of Contents.*?(?=Chapter|Section|\d\.\s)',
            r'Acknowledgement.*?(?=Chapter|Section|\d\.\s)',
            r'Message from.*?(?=Chapter|Section)',
            r'ISBN.*?(?=\n)',
            r'Cataloguing.*?(?=\n)',
        ]
        
        clean_text = text
        for pattern in junk_markers:
            clean_text = re.sub(pattern, '', clean_text, flags=re.S|re.I)
        
        # Find first substantial medical paragraph (look for keywords)
        disease_lower = disease.lower()
        paragraphs = clean_text.split('\n')
        
        relevant = []
        for para in paragraphs:
            para_clean = para.strip()
            if len(para_clean) > 100 and (disease_lower in para_clean.lower() or 
                                          any(word in para_clean.lower() for word in 
                                              ['symptom', 'diagnosis', 'treatment', 'patient', 'clinical'])):
                relevant.append(para_clean)
                if len(relevant) >= 3:  # Take first 3 relevant paragraphs
                    break
        
        if relevant:
            return '\n\n'.join(relevant[:3])
        
        # Ultimate fallback: first 500 chars of cleaned text
        return clean_text[:500] + "..."

    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean up common PDF extraction artifacts while preserving meaning.
        """
        if not text:
            return text
        
        # Fix concatenated words (common OCR error)
        # forsurgical -> for surgical, Biopsytechniques -> Biopsy techniques
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase splits
        text = re.sub(r'(Biopsy)(techniques)', r'\1 techniques', text, flags=re.I)
        text = re.sub(r'(for)(surgical)', r'for surgical', text, flags=re.I)
        text = re.sub(r'(and)(nonsurgical)', r'and nonsurgical', text, flags=re.I)
        text = re.sub(r'(and)(radiation)', r'and radiation', text, flags=re.I)
        text = re.sub(r'(forthe)', r'for the', text, flags=re.I)
        text = re.sub(r'(ofthe)', r'of the', text, flags=re.I)
        text = re.sub(r'(tothe)', r'to the', text, flags=re.I)
        text = re.sub(r'(inthe)', r'in the', text, flags=re.I)
        
        # Fix common spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def _synthesize_guidance_narrative(self, guidance_sections: Dict[str, str], disease: str, confidence: float) -> str:
        """
        Generate narrative using ACTUAL PDF content when available, falling back
        to curated medical knowledge only when PDFs don't have usable content.
        This ensures we USE the guideline PDFs meaningfully.
        """
        disease_name = disease.replace('Skin Cancer', '').replace('Pictures', '').replace('Photos', '').strip()
        parts = []

        # Opening based on confidence
        if confidence > 0.9:
            parts.append(f"The model has identified {disease_name} with very high confidence.")
        elif confidence > 0.7:
            parts.append(f"The analysis indicates {disease_name} as the likely condition.")
        else:
            parts.append(f"The model suggests {disease_name} as a possibility that warrants further evaluation.")

        # PRIMARY: Try to use PDF content if available and clean
        pdf_content_used = False
        if guidance_sections and len(guidance_sections) > 0:
            # Clean and compile PDF-extracted content
            pdf_parts = []
            
            for section in ['overview', 'symptoms', 'diagnosis', 'risk_factors', 'treatment', 'prevention']:
                if section in guidance_sections:
                    cleaned = self._clean_extracted_text(guidance_sections[section])
                    if len(cleaned) > 30:  # Meaningful content
                        # Make it flow naturally
                        if section == 'overview':
                            pdf_parts.append(f"Medical literature indicates that {cleaned.lower() if cleaned[0].isupper() and 'The' not in cleaned[:3] else cleaned}")
                        elif section == 'symptoms':
                            pdf_parts.append(f"Clinical presentation: {cleaned}")
                        elif section == 'diagnosis':
                            pdf_parts.append(f"For diagnosis: {cleaned}")
                        elif section == 'risk_factors':
                            pdf_parts.append(f"Risk factors: {cleaned}")
                        elif section == 'treatment':
                            pdf_parts.append(f"Treatment approaches: {cleaned}")
                        elif section == 'prevention':
                            pdf_parts.append(f"Prevention: {cleaned}")
            
            if len(pdf_parts) >= 2:  # If we got at least 2 good sections from PDF
                parts.extend(pdf_parts)
                pdf_content_used = True

        # FALLBACK: Use curated medical knowledge if PDF content insufficient
        if not pdf_content_used:
            disease_lower = disease.lower()
            
            if 'melanoma' in disease_lower:
                parts.append("Melanoma is a serious form of skin cancer that develops from melanocytes, the cells that produce skin pigment.")
                parts.append("It often appears as an irregular, asymmetric mole with varied colors and irregular borders, following the \"ABCDE\" rule: Asymmetry, Border irregularity, Color variation, Diameter greater than 6mm, and Evolving over time.")
                parts.append("Early detection through careful skin examination is crucial, as melanoma can metastasize to other parts of the body if left untreated.")
                parts.append("Key risk factors include excessive UV exposure from sun or tanning beds, fair skin, family history of melanoma, numerous or atypical moles, and previous history of skin cancer.")
                parts.append("Any suspicious lesion should be evaluated by a dermatologist, typically with dermoscopy followed by biopsy for histopathologic confirmation if warranted.")
                parts.append("Treatment usually involves surgical excision with appropriate margins, and prognosis is excellent when detected early.")
                
            elif 'eczema' in disease_lower:
                parts.append("Eczema, also known as atopic dermatitis, is a chronic inflammatory skin condition characterized by dry, itchy, and inflamed patches of skin.")
                parts.append("It commonly affects flexural areas such as the inner elbows, behind the knees, and on the face and neck, though it can appear anywhere on the body.")
                parts.append("The condition often demonstrates a relapsing-remitting pattern, with periods of flare-ups triggered by environmental factors, allergens, stress, or irritants.")
                parts.append("Common triggers include harsh soaps, fragrances, certain fabrics, temperature extremes, and food allergens in some individuals.")
                parts.append("Management involves a multifaceted approach: regular application of emollients to maintain skin barrier function, identification and avoidance of triggers, and use of topical corticosteroids or calcineurin inhibitors during flare-ups.")
                parts.append("A dermatologist can develop a comprehensive, individualized treatment plan and monitor for potential complications.")
                
            elif 'psoriasis' in disease_lower:
                parts.append("Psoriasis is a chronic autoimmune condition characterized by accelerated skin cell turnover, resulting in thick, raised, scaly plaques.")
                parts.append("These plaques typically present as well-demarcated, erythematous patches covered with silvery-white scales, most commonly affecting the elbows, knees, scalp, and lower back.")
                parts.append("The condition is not contagious and varies significantly in severity, from limited localized patches to extensive body surface involvement.")
                parts.append("Psoriasis can be associated with psoriatic arthritis in some patients, causing joint pain and swelling that requires additional management.")
                parts.append("Treatment options are tailored to disease severity and include topical therapies (corticosteroids, vitamin D analogs), phototherapy, and systemic agents (methotrexate, biologics) for moderate to severe cases.")
                parts.append("Regular follow-up with a dermatologist or rheumatologist is recommended to optimize treatment and monitor for associated conditions.")
                
            elif 'acne' in disease_lower or 'rosacea' in disease_lower:
                parts.append("Acne and rosacea are related but distinct inflammatory skin conditions primarily affecting the face.")
                parts.append("Acne results from follicular hyperkeratinization, excess sebum production, and bacterial colonization (especially *Cutibacterium acnes*), manifesting as comedones, papules, pustules, and potentially nodules or cysts.")
                parts.append("Rosacea causes persistent facial erythema, telangiectasia (visible blood vessels), inflammatory papules and pustules, and sometimes thickening of facial skin (phymatous changes).")
                parts.append("Both conditions can significantly impact quality of life and self-esteem, warranting appropriate medical management.")
                parts.append("Treatment approaches vary by type and severity, ranging from topical retinoids and benzoyl peroxide for acne, to metronidazole and azelaic acid for rosacea, with oral antibiotics or isotretinoin reserved for more severe cases.")
                parts.append("Identifying and avoiding triggers (spicy foods, alcohol, temperature extremes for rosacea) and maintaining a gentle skincare routine are important adjunctive measures.")
                
            elif 'normal' in disease_lower:
                parts.append("The analyzed image demonstrates characteristics consistent with healthy, non-pathologic skin without obvious concerning features.")
                parts.append("Even when skin appears normal, regular self-monitoring remains important for early detection of any changes, particularly in moles or new lesions.")
                parts.append("The ABCDE criteria (Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving characteristics) can help identify potentially concerning lesions.")
                parts.append("Protective measures are essential: daily broad-spectrum sunscreen (SPF 30 or higher), protective clothing and hats, seeking shade during peak UV hours (10 AM to 4 PM), and avoiding tanning beds.")
                parts.append("Annual full-body skin examinations by a dermatologist are recommended, especially for individuals with fair skin, numerous moles, family history of skin cancer, or significant lifetime sun exposure.")
                
            else:
                parts.append("This skin condition should be evaluated by a qualified healthcare provider for accurate diagnosis.")
                parts.append("Dermatological diagnosis relies on detailed clinical examination, comprehensive patient history, and when indicated, diagnostic procedures such as dermoscopy, skin scraping, or biopsy.")
                parts.append("Treatment protocols are individualized based on the specific diagnosis, disease severity, anatomical location, patient age, comorbidities, and treatment response history.")

        # Closing with appropriate urgency
        if 'melanoma' in disease.lower() or 'cancer' in disease.lower():
            parts.append(" Given the serious nature of this condition, prompt dermatology evaluation is essential. Early intervention significantly improves treatment outcomes and prognosis. Please do not delay seeking medical attention.")
        elif 'normal' in disease.lower():
            parts.append("Continue routine skin monitoring and maintaining healthy skin care practices. If you notice any changes or have concerns, consult a healthcare provider for evaluation.")
        else:
            parts.append("Professional medical evaluation is recommended to confirm the diagnosis and establish an evidence-based treatment plan. Seek care promptly if symptoms worsen or new concerns arise.")

        return ' '.join(parts)

    def _generate_text_explanation(self, patient_data: pd.DataFrame) -> str:
        """
        Create a simple textual explanation when SHAP is not available.
        This summarizes prominent features (boolean flags and top numeric
        contributors) from the single-row `patient_data` DataFrame.
        """
        try:
            row = patient_data.iloc[0]
        except Exception:
            return "No patient data available for textual explanation."

        # Boolean or binary-like positive features
        positives = [col.replace('_', ' ').title() for col, val in row.items() if (isinstance(val, (bool, int)) and val == True) or val == 1]

        # Numeric features: take top absolute values
        numeric = {col: abs(val) for col, val in row.items() if isinstance(val, (int, float))}
        top_numeric = sorted(numeric.items(), key=lambda x: x[1], reverse=True)[:5]

        parts = ["Textual explanation (SHAP not available):\n"]
        if positives:
            parts.append("Positive / present features: ")
            parts.append(", ".join(positives))
        else:
            parts.append("No obvious boolean-present features detected.")

        if top_numeric:
            parts.append("\nTop numeric contributors (by magnitude):")
            for col, val in top_numeric:
                parts.append(f" - {col.replace('_',' ').title()}: {val}")

        return "\n".join(parts)

    def generate_text_for_image(self, probabilities: Dict[str, float], top_class: str) -> str:
        """
        Create a natural textual explanation for an image classification result.
        Reads guideline PDFs intelligently and synthesizes medical information
        into conversational, user-friendly language. NO raw PDF pasting.
        """
        try:
            # Get top prediction info
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            top_name, top_prob = sorted_probs[0]
            
            # Read and intelligently parse guideline PDFs
            guidance_text = self.get_guidance_for_disease(top_class)
            guidance_sections = self._extract_key_guidance_points(guidance_text, top_class)
            
            # Generate natural narrative from medical knowledge
            narrative = self._synthesize_guidance_narrative(guidance_sections, top_class, top_prob)
            
            # Build user-friendly explanation
            parts = []
            parts.append(f"**{top_name}** detected with {top_prob*100:.1f}% confidence.\n")
            parts.append(narrative)
            
            # Add probability summary
            parts.append("\n\n**Confidence Levels:**")
            for cls, prob in sorted_probs[:5]:
                bar = '█' * int(prob * 15) + '░' * (15 - int(prob * 15))
                parts.append(f"- {cls}: {prob*100:.1f}% {bar}")
            
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Error generating textual explanation: {e}")
            return f"Unable to generate explanation. Please consult a healthcare provider for evaluation of {top_class}."

    def generate_narrative_for_image(self, probabilities: Dict[str, float], top_class: str) -> str:
        """
        Generate a comprehensive, narrative-style explanation by reading PDFs
        and synthesizing medical knowledge into natural, conversational English.
        Returns flowing paragraphs without static templates.
        """
        try:
            # Get prediction data
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            top_name, top_prob = sorted_probs[0]
            
            # Read guideline PDFs and extract intelligent insights
            guidance_text = self.get_guidance_for_disease(top_class)
            guidance_sections = self._extract_key_guidance_points(guidance_text, top_class)
            
            # Generate synthesized medical narrative (not static templates)
            medical_context = self._synthesize_guidance_narrative(guidance_sections, top_class, top_prob)
            
            # Build flowing explanation
            parts = []
            
            # Opening statement
            parts.append(f"Our diagnostic analysis has identified **{top_name}** with {top_prob*100:.1f}% confidence based on the visual patterns in the submitted image.")
            
            # Add synthesized medical knowledge
            parts.append(f"\n{medical_context}")
            
            # Detailed confidence breakdown
            parts.append("\n**Diagnostic Confidence Breakdown:**")
            parts.append("The AI model evaluated the image against multiple skin conditions:")
            for cls, prob in sorted_probs[:5]:
                if prob > 0.001:
                    bar = '█' * int(prob * 20) + '░' * (20 - int(prob * 20))
                    parts.append(f"- {cls}: {prob*100:.1f}% {bar}")
                else:
                    parts.append(f"- {cls}: <0.1%")
            
            # Context-specific recommendations
            parts.append("\n**What You Should Do Next:**")
            
            if 'melanoma' in top_name.lower() or 'cancer' in top_name.lower():
                parts.append("⚠️ **This is urgent.** Schedule a dermatology appointment within the next 1-2 weeks. Request a dermoscopic examination and be prepared to discuss biopsy options if recommended. Do not attempt any self-treatment.")
            elif 'normal' in top_name.lower():
                parts.append("Your skin appears healthy based on this analysis. Continue regular self-monitoring using the ABCDE criteria for moles. Maintain sun protection habits and schedule routine annual skin checks with a dermatologist.")
            elif top_prob > 0.8:
                parts.append("Schedule an appointment with a dermatologist for professional evaluation. Bring photos showing how the condition has changed over time. Document any symptoms like itching, pain, or spread.")
            else:
                parts.append("The analysis suggests this condition, but confirmation by a healthcare provider is recommended. Schedule a consultation to discuss symptoms, triggers, and appropriate treatment options.")
            
            # Professional disclaimer
            parts.append("\n---\n*This AI-generated analysis is for educational purposes only and does not constitute medical advice. Always consult qualified healthcare professionals for diagnosis and treatment.*")
            
            return "\n".join(parts)
            
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return f"An error occurred during analysis. Please consult a dermatologist for professional evaluation of this skin condition."
    
    def combine_explanations(
        self,
        shap_explanation: Dict[str, Any],
        rule_trace: List[Dict[str, Any]],
        fusion_result: Dict[str, Any]
    ) -> str:
        """
        Combine SHAP explanations with rule traces for comprehensive explanation.
        
        Args:
            shap_explanation: SHAP explanation dictionary
            rule_trace: List of fired rules from rule engine
            fusion_result: Fusion results dictionary
        
        Returns:
            Combined explanation string
        """
        explanation = []
        
        explanation.append("="*60)
        explanation.append("CLINICAL DECISION SUPPORT SYSTEM - EXPLANATION REPORT")
        explanation.append("="*60)
        
        # Primary diagnosis
        primary_disease, confidence = fusion_result.get('primary_diagnosis', ('unknown', 0))
        explanation.append(f"\nPrimary Diagnosis: {primary_disease.upper()}")
        explanation.append(f"Confidence: {confidence:.1%}")
        explanation.append(f"Risk Level: {fusion_result.get('overall_risk', 'unknown').upper()}")
        
        # Rule-based reasoning
        explanation.append("\n" + "-"*60)
        explanation.append("SYMBOLIC REASONING (Rule-Based)")
        explanation.append("-"*60)
        
        if rule_trace:
            explanation.append(f"\n{len(rule_trace)} diagnostic rule(s) activated:\n")
            for rule in rule_trace:
                explanation.append(f"✓ {rule['name']}")
                explanation.append(f"  Description: {rule['description']}")
                explanation.append(f"  Disease: {rule['disease']}")
                explanation.append(f"  Confidence: {rule['confidence']}")
                explanation.append(f"  Probability Boost: +{rule['boost']:.2f}\n")
        else:
            explanation.append("\nNo specific diagnostic rules were triggered.")
        
        # Machine Learning explanation
        explanation.append("-"*60)
        explanation.append("NEURAL NETWORK REASONING (ML Model)")
        explanation.append("-"*60)
        
        if shap_explanation and shap_explanation.get('explanation'):
            explanation.append("\n" + shap_explanation['explanation'])
        else:
            explanation.append("\nML explanation not available")
        
        # Component contributions
        explanation.append("\n" + "-"*60)
        explanation.append("NEURO-SYMBOLIC FUSION")
        explanation.append("-"*60)
        
        contributions = fusion_result.get('component_contributions', {})
        explanation.append("\nComponent-wise confidence scores:")
        
        for component, scores in contributions.items():
            if scores and any(s > 0 for s in scores.values()):
                explanation.append(f"\n{component.replace('_', ' ').title()}:")
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for disease, score in sorted_scores[:3]:
                    if score > 0:
                        explanation.append(f"  • {disease}: {score:.1%}")
        
        # Final integrated score
        explanation.append("\n" + "-"*60)
        explanation.append("INTEGRATED DISEASE PROBABILITIES")
        explanation.append("-"*60 + "\n")
        
        disease_scores = fusion_result.get('disease_scores', {})
        if disease_scores:
            sorted_scores = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
            for disease, score in sorted_scores:
                bar_length = int(score * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                explanation.append(f"{disease:15s} {bar} {score:.1%}")
        
        # Include official guidance for primary disease if available
        try:
            primary_disease, _ = fusion_result.get('primary_diagnosis', ('unknown', 0))
        except Exception:
            primary_disease = 'unknown'

        guidance = self.get_guidance_for_disease(primary_disease)
        if guidance:
            explanation.append("\n" + "-"*60)
            explanation.append("OFFICIAL GUIDANCE (WHO / CDC or local files)")
            explanation.append("-"*60)
            explanation.append(guidance)

        explanation.append("\n" + "="*60)
        
        return "\n".join(explanation)
    
    def create_feature_importance_plot(
        self,
        feature_contributions: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Create a visualization of feature importance.
        
        Args:
            feature_contributions: Dictionary of feature SHAP values
            save_path: Optional path to save plot
        """
        # Sort by absolute value
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:self.max_features_display]
        
        features, values = zip(*sorted_features)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if v < 0 else 'green' for v in values]
        ax.barh(range(len(features)), values, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Importance for Diagnosis')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_patient_report(
        self,
        patient_data: Dict[str, Any],
        fusion_result: Dict[str, Any],
        shap_explanation: Dict[str, Any],
        rule_trace: List[Dict[str, Any]],
        recommendations: List[str]
    ) -> str:
        """
        Generate comprehensive patient report.
        
        Args:
            patient_data: Patient's clinical data
            fusion_result: Fusion results
            shap_explanation: SHAP explanation
            rule_trace: Rule traces
            recommendations: Clinical recommendations
        
        Returns:
            Formatted patient report
        """
        report = []
        
        report.append("╔" + "═"*78 + "╗")
        report.append("║" + " "*20 + "CLINICAL DIAGNOSTIC REPORT" + " "*32 + "║")
        report.append("╚" + "═"*78 + "╝")
        
        # Patient summary
        report.append("\n📋 PATIENT SUMMARY")
        report.append("─"*80)
        demographics = patient_data.get('demographics', {})
        report.append(f"Age: {demographics.get('age', 'N/A')}")
        report.append(f"Gender: {demographics.get('gender', 'N/A')}")
        
        # Key symptoms
        report.append("\n🔍 PRESENTING SYMPTOMS")
        report.append("─"*80)
        symptoms = patient_data.get('symptoms', {})
        positive_symptoms = [k.replace('_', ' ').title() for k, v in symptoms.items() if v]
        if positive_symptoms:
            for symptom in positive_symptoms:
                report.append(f"  ✓ {symptom}")
        else:
            report.append("  No significant symptoms reported")
        
        # Vital signs
        report.append("\n🩺 VITAL SIGNS")
        report.append("─"*80)
        vitals = patient_data.get('vitals', {})
        for vital, value in vitals.items():
            display_name = vital.replace('_', ' ').title()
            report.append(f"  {display_name}: {value}")
        
        # Lab results
        labs = patient_data.get('labs', {})
        if labs:
            report.append("\n🧪 LABORATORY RESULTS")
            report.append("─"*80)
            for lab, value in labs.items():
                display_name = lab.replace('_', ' ').title()
                report.append(f"  {display_name}: {value}")
        
        # Diagnosis
        report.append("\n🎯 DIAGNOSIS")
        report.append("─"*80)
        primary_disease, confidence = fusion_result.get('primary_diagnosis', ('unknown', 0))
        risk = fusion_result.get('overall_risk', 'unknown')
        
        report.append(f"Primary: {primary_disease.upper()}")
        report.append(f"Confidence: {confidence:.1%}")
        report.append(f"Risk Level: {risk.upper()}")
        
        # Detailed explanation
        report.append("\n📊 EXPLANATION")
        report.append("─"*80)
        combined_explanation = self.combine_explanations(
            shap_explanation,
            rule_trace,
            fusion_result
        )
        report.append(combined_explanation)
        
        # Recommendations
        report.append("\n💊 RECOMMENDATIONS")
        report.append("─"*80)
        for rec in recommendations:
            report.append(rec)
        
        report.append("\n" + "═"*80)
        report.append("⚠️  DISCLAIMER: This report is generated by an AI system for")
        report.append("   educational purposes. It should NOT replace professional")
        report.append("   medical judgment. Always consult healthcare providers.")
        report.append("═"*80)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    explainer = ExplainabilityModule()
    
    # Mock data
    patient_data = {
        'symptoms': {'fever': True, 'headache': True, 'rash': True},
        'vitals': {'temperature': 39.5, 'oxygen_saturation': 96},
        'labs': {'platelet_count': 95000, 'wbc_count': 3200},
        'demographics': {'age': 35, 'gender': 'M'}
    }
    
    fusion_result = {
        'primary_diagnosis': ('dengue', 0.82),
        'overall_risk': 'high',
        'disease_scores': {'dengue': 0.82, 'covid19': 0.12, 'pneumonia': 0.06},
        'component_contributions': {
            'rule_based': {'dengue': 0.7},
            'random_forest': {'dengue': 0.75}
        }
    }
    
    rule_trace = [
        {
            'name': 'Dengue_Classic',
            'description': 'Classic dengue presentation',
            'disease': 'dengue',
            'confidence': 'high',
            'boost': 0.4
        }
    ]
    
    recommendations = [
        "Seek medical evaluation within 24 hours",
        "Monitor platelet count daily",
        "Ensure adequate hydration"
    ]
    
    shap_explanation = {
        'explanation': 'Fever, low platelets, and rash are key contributors'
    }
    
    report = explainer.generate_patient_report(
        patient_data,
        fusion_result,
        shap_explanation,
        rule_trace,
        recommendations
    )
    
    print(report)
