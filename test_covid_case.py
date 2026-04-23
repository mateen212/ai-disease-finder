"""
Test script for severe COVID-19 case

Tests the rule engine and fusion with the user's example:
- High fever (39.5°C)
- Critical O2 saturation (88%)
- Loss of taste (COVID-specific)
- Multiple COVID symptoms
"""

from src.rule_engine import RuleEngine
from src.fusion import NeuroSymbolicFusion
import json

# User's test case
patient_data = {
    "symptoms": {
        "fever": True,
        "cough": True,
        "fatigue": True,
        "headache": True,
        "shortness_of_breath": True,
        "loss_of_taste": True,
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

print("=" * 80)
print("TESTING SEVERE COVID-19 CASE")
print("=" * 80)
print("\nPatient Profile:")
print(f"  Age: {patient_data['demographics']['age']}, Sex: {patient_data['demographics']['sex']}")
print(f"  Temperature: {patient_data['vitals']['temperature']}°C (HIGH FEVER)")
print(f"  O2 Saturation: {patient_data['vitals']['oxygen_saturation']}% (CRITICAL - Normal ≥95%)")
print(f"  Loss of taste: {patient_data['symptoms']['loss_of_taste']} (COVID-SPECIFIC)")
print(f"  CRP: {patient_data['labs']['crp']} mg/L (ELEVATED - Normal <10)")
print(f"  Ferritin: {patient_data['labs']['ferritin']} ng/mL (VERY HIGH - suggests cytokine storm)")

# Test Rule Engine
print("\n" + "=" * 80)
print("STEP 1: RULE ENGINE EVALUATION")
print("=" * 80)

engine = RuleEngine()
rule_results = engine.evaluate_rules(patient_data)

print(f"\nRules Fired: {rule_results['rule_count']}")
for rule in rule_results['fired_rules']:
    print(f"  ✓ {rule.name}")
    print(f"    → Disease: {rule.conclusion.get('disease')}")
    print(f"    → Boost: {rule.conclusion.get('probability_boost')}")
    if 'risk_level' in rule.conclusion:
        print(f"    → Risk: {rule.conclusion.get('risk_level').upper()}")

print(f"\nRule Engine Scores:")
for disease, score in rule_results['disease_scores'].items():
    if score > 0:
        print(f"  {disease}: {score * 100:.1f}%")

# Simulate Random Forest predictions (RF is misclassifying as pneumonia)
print("\n" + "=" * 80)
print("STEP 2: RANDOM FOREST SIMULATION")
print("=" * 80)
print("(RF is trained on limited COVID data, often confuses with pneumonia)")

rf_scores = {
    'dengue': 0.05,
    'covid19': 0.27,  # Low! RF is confused
    'pneumonia': 0.491  # RF thinks it's pneumonia
}

print("\nRandom Forest Predictions:")
for disease, score in sorted(rf_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {disease}: {score * 100:.1f}%")

# Test Fusion
print("\n" + "=" * 80)
print("STEP 3: NEURO-SYMBOLIC FUSION")
print("=" * 80)

fusion = NeuroSymbolicFusion()
fusion_result = fusion.fuse_predictions(
    rule_scores=rule_results['disease_scores'],
    rf_scores=rf_scores,
    rule_metadata=rule_results
)

print(f"\nFusion Configuration:")
print(f"  Strategy: {fusion_result['fusion_strategy']}")
print(f"  Weights: {fusion_result['weights_used']}")

print(f"\nFinal Fused Scores:")
for disease, score in sorted(fusion_result['disease_scores'].items(), key=lambda x: x[1], reverse=True)[:5]:
    if score > 0:
        print(f"  {disease}: {score * 100:.1f}%")

print(f"\n{'*' * 80}")
print("PRIMARY DIAGNOSIS")
print('*' * 80)
primary = fusion_result['primary_diagnosis']
print(f"Disease: {primary[0].upper()}")
print(f"Confidence: {primary[1] * 100:.1f}%")
print(f"Risk Level: {rule_results['risk_assessments'].get(primary[0], 'unknown').upper()}")

print(f"\n{'=' * 80}")
print("ANALYSIS")
print('=' * 80)

covid_score = fusion_result['disease_scores'].get('covid19', 0)
if covid_score < 0.60:
    print("⚠️  WARNING: Confidence is too low for such a clear COVID-19 case!")
    print("   Patient has:")
    print("   - Critical O2 saturation (88%)")
    print("   - Loss of taste (highly specific to COVID-19)")
    print("   - Multiple COVID rules fired")
    print("   - High fever and respiratory distress")
    print("   Expected confidence: >80%")
else:
    print("✅ Confidence is appropriate for this severe COVID-19 presentation")

print("\nComponent Breakdown:")
for disease in ['covid19', 'pneumonia', 'dengue']:
    if disease in fusion_result['component_contributions']['rule_based']:
        rule_contrib = fusion_result['component_contributions']['rule_based'].get(disease, 0)
        rf_contrib = fusion_result['component_contributions']['random_forest'].get(disease, 0)
        final = fusion_result['disease_scores'].get(disease, 0)
        print(f"\n  {disease.upper()}:")
        print(f"    Rule Engine: {rule_contrib * 100:.1f}%")
        print(f"    Random Forest: {rf_contrib * 100:.1f}%")
        print(f"    Final Fused: {final * 100:.1f}%")

print("\n" + "=" * 80)
