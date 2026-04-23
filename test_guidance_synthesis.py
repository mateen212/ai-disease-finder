"""
Test synthesized clinical guidance
"""

from src.explainability import ExplainabilityModule

explainer = ExplainabilityModule()

test_diseases = ['covid19', 'dengue', 'pneumonia']

for disease in test_diseases:
    print("="*80)
    print(f"TESTING: {disease.upper()}")
    print("="*80)
    
    result = explainer.get_guidance_for_disease(disease)
    
    print(f"\nSource: {result.get('source', 'N/A')}")
    print(f"Content length: {len(result.get('content', ''))} chars")
    print("\nContent preview:")
    print(result.get('content', '')[:800])
    print("\n... (truncated)\n")
