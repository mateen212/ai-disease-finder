"""
Quick test for improved UI and guidance matching

Tests:
1. COVID-19 diagnosis shows covid.pdf (not clinical_guidelines.md)
2. Rule firing display shows primary vs. competing diagnoses clearly
"""

import json

# Import components  
from src.hybrid_system import HybridDiagnosticSystem
from src.explainability import ExplainabilityModule

# User's test case
patient_data = {
    "symptoms": {
        "fever": True,
        "cough": True,
        "fatigue": True,
        "headache": True,
        "shortness_of_breath": True,
        "loss_of_taste": True,  # COVID-specific!
        "sore_throat": True,
        "body_aches": True,
        "nausea": True,
        "vomiting": False,
        "diarrhea": True,
        "rash": False,
        "joint_pain": True,
        "retro_orbital_pain": False
    },
    "vitals": {
        "temperature": 39.5,
        "heart_rate": 110,
        "respiratory_rate": 26,
        "blood_pressure_systolic": 100,
        "blood_pressure_diastolic": 65,
        "oxygen_saturation": 88  # Critical!
    },
    "labs": {
        "wbc_count": 4000,
        "platelet_count": 180000,
        "hemoglobin": 13.5,
        "crp": 85,
        "ferritin": 900
    },
    "demographics": {
        "age": 60,
        "sex": "Male"
    }
}

print("="*80)
print("TESTING UI IMPROVEMENTS")
print("="*80)

# Test hybrid system
print("\n1. Testing diagnosis...")
system = HybridDiagnosticSystem()
result = system.diagnose(patient_data)

print(f"   Primary Diagnosis: {result['diagnosis']}")
print(f"   Confidence: {result['confidence']*100:.1f}%")

# Check component predictions
if 'Rule Engine' in result['component_predictions']:
    re_data = result['component_predictions']['Rule Engine']
    if 'metadata' in re_data:
        metadata = re_data['metadata']
        fired_rules = metadata.get('fired_rules', [])
        
        # Group by disease
        rules_by_disease = {}
        for rule in fired_rules:
            disease = rule.conclusion.get('disease', 'unknown')
            if disease not in rules_by_disease:
                rules_by_disease[disease] = []
            rules_by_disease[disease].append(rule)
        
        print(f"\n   Rules fired by disease:")
        for disease, rules in rules_by_disease.items():
            symbol = "✅ PRIMARY" if disease == result['diagnosis'] else "⚠️  competing"
            print(f"     {symbol} {disease}: {len(rules)} rules")

# Test guidance matching
print("\n2. Testing guidance file matching...")
explainer = ExplainabilityModule()

test_diseases = ['covid19', 'dengue', 'pneumonia', 'melanoma']
for disease in test_diseases:
    guidance = explainer.get_guidance_for_disease(disease)
    
    # Check which keywords appear
    has_covid = 'covid' in guidance.lower() or 'sars-cov-2' in guidance.lower()
    has_dengue = 'dengue' in guidance.lower() or 'aedes' in guidance.lower()
    has_pneumonia = 'pneumonia' in guidance.lower() or 'streptococcus' in guidance.lower()
    has_melanoma = 'melanoma' in guidance.lower() or 'abcde' in guidance.lower()
    has_skin_multi = sum([has_melanoma, 'eczema' in guidance.lower(), 'psoriasis' in guidance.lower()]) >= 2
    
    length = len(guidance)
    print(f"\n   {disease}:")
    print(f"     Length: {length} chars")
    
    if disease == 'covid19':
        if has_covid and not has_skin_multi:
            print(f"     ✅ Correct: Contains COVID content, no multi-disease content")
        else:
            print(f"     ❌ WRONG: has_covid={has_covid}, has_skin_multi={has_skin_multi}")
            print(f"     First 200 chars: {guidance[:200]}")
    elif disease == 'dengue':
        if has_dengue and not has_skin_multi:
            print(f"     ✅ Correct: Contains dengue content")
        else:
            print(f"     ⚠️  Check: has_dengue={has_dengue}, has_skin_multi={has_skin_multi}")
    elif disease == 'pneumonia':
        if has_pneumonia and not has_skin_multi:
            print(f"     ✅ Correct: Contains pneumonia content")
        else:
            print(f"     ⚠️  Check: has_pneumonia={has_pneumonia}, has_skin_multi={has_skin_multi}")
    elif disease == 'melanoma':
        if has_melanoma:
            print(f"     ✅ Correct: Contains melanoma content")
        else:
            print(f"     ⚠️  Check: has_melanoma={has_melanoma}")

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)
