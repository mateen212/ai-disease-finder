"""
Final test - Launch app with COVID case to verify curated guidance display
"""

from src.hybrid_system import HybridDiagnosticSystem

# Initialize system
system = HybridDiagnosticSystem()

# Severe COVID-19 case  
test_case = {
    'fever_days': 5,
    'temperature': 39.5,
    'cough': True,
    'cough_type': 'dry',
    'fever': True,
    'shortness_of_breath': True,
    'loss_of_taste_smell': True,  # Pathognomonic
    'oxygen_saturation': 88,  # Critical hypoxemia
    'age': 55,
    'comorbidities': 'diabetes'
}

print("="*80)
print("TESTING: Severe COVID-19 Case")
print("="*80)
print("\nInput symptoms:")
for key, val in test_case.items():
    print(f"  {key}: {val}")

# Get diagnosis
result = system.diagnose(test_case)

print(f"\n🏥 DIAGNOSIS: {result['diagnosis'].upper()}")
print(f"📊 CONFIDENCE: {result['confidence']*100:.1f}%")

print(f"\n🔍 Rules fired: {len(result['component_predictions']['Rule Engine']['metadata']['fired_rules'])}")

# Get guidance
from src.explainability import ExplainabilityModule
explainer = ExplainabilityModule()
guidance_data = explainer.get_guidance_for_disease(result['diagnosis'])

print(f"\n📋 CLINICAL GUIDANCE")
print(f"Source file: {guidance_data['source']}")
print(f"Content length: {len(guidance_data['content'])} chars")

# Check if curated (should have proper structure)
content = guidance_data['content']
has_proper_sections = all(x in content for x in [
    '**🔍 Clinical Presentation**',
    '**🧪 Diagnosis**',
    '**💊 Treatment**'
])

# Check for junk
has_legal_junk = any(x in content.lower() for x in [
    'world health organization',
    'creative commons',
    'suggested citation',
    'isbn',
    'copyright'
])

print(f"\n✅ Quality checks:")
print(f"  Has proper medical sections: {has_proper_sections}")
print(f"  Free from legal/copyright junk: {not has_legal_junk}")

print(f"\n📄 Full guidance content:")
print("-"*80)
print(content)
print("-"*80)

if has_proper_sections and not has_legal_junk:
    print("\n✅ SUCCESS: Curated clinical guidance is displayed correctly!")
else:
    print("\n⚠️  WARNING: Guidance may have issues")

