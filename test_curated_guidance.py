"""
Test curated clinical guidance
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
    
    content = result.get('content', '')
    
    # Check if it's curated content (should have emoji headers)
    has_curated_format = '**🔍 Clinical Presentation**' in content
    
    if has_curated_format:
        print("✅ Using CURATED medical knowledge")
    else:
        print("⚠️  Using EXTRACTED PDF content (may have junk)")
    
    print("\nContent preview:")
    print(content[:1000])
    print("\n")
