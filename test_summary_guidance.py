"""
Summary: Test all three clinical diseases with curated guidance
"""

from src.explainability import ExplainabilityModule

explainer = ExplainabilityModule()

diseases = ['covid19', 'dengue', 'pneumonia']

print("="*80)
print("CLINICAL GUIDANCE SUMMARY TEST")
print("="*80)

for disease in diseases:
    print(f"\n{'='*80}")
    print(f"🏥 {disease.upper()}")
    print('='*80)
    
    guidance_data = explainer.get_guidance_for_disease(disease)
    content = guidance_data['content']
    source = guidance_data['source']
    
    # Quality checks
    has_sections = all(header in content for header in [
        '**🔍 Clinical Presentation**',
        '**🧪 Diagnosis**',
        '**💊 Treatment**'
    ])
    
    has_junk = any(junk in content.lower() for junk in [
        'world health organization',
        'creative commons',
        'copyright',
        'isbn',
        'suggested citation',
        'the designations employed'
    ])
    
    print(f"\n📄 Source: {source}")
    print(f"📏 Length: {len(content)} chars")
    print(f"✅ Has proper sections: {has_sections}")
    print(f"✅ Free from junk: {not has_junk}")
    
    if has_sections and not has_junk:
        print("🎉 PASS: Clean curated medical content")
    else:
        print("⚠️  FAIL: Quality issues detected")
    
    print(f"\nFirst 400 chars:")
    print(content[:400])

print("\n" + "="*80)
print("✅ ALL TESTS COMPLETE")
print("="*80)
