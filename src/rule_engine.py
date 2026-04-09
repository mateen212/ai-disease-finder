"""
Rule-Based Inference Engine with Forward Chaining

Implements medical diagnostic rules based on WHO/CDC guidelines.
Uses forward chaining to infer diagnoses from patient symptoms and findings.
"""

from typing import Dict, List, Any, Optional, Tuple
import yaml
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """Represents a single diagnostic rule"""
    name: str
    description: str
    conditions: Dict[str, Any]
    conclusion: Dict[str, Any]
    fired: bool = False
    confidence: str = "medium"


class RuleEngine:
    """
    Forward-chaining inference engine for medical diagnosis.
    
    Evaluates patient data against diagnostic rules to infer possible diseases.
    Based on established clinical guidelines from WHO/CDC.
    """
    
    def __init__(self, rules_file: str = "config/rules.yaml"):
        """
        Initialize the rule engine.
        
        Args:
            rules_file: Path to YAML file containing diagnostic rules
        """
        self.rules_file = Path(rules_file)
        self.rules: Dict[str, List[Rule]] = {}
        self.thresholds: Dict[str, Any] = {}
        self.risk_levels: Dict[str, Any] = {}
        self.fired_rules: List[Rule] = []
        
        self._load_rules()
    
    def _load_rules(self):
        """Load rules from YAML configuration file"""
        try:
            with open(self.rules_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load disease-specific rules
            for disease in ['dengue', 'covid19', 'pneumonia']:
                rule_key = f"{disease}_rules"
                if rule_key in config:
                    self.rules[disease] = []
                    for rule_data in config[rule_key]:
                        rule = Rule(
                            name=rule_data['name'],
                            description=rule_data['description'],
                            conditions=rule_data['conditions'],
                            conclusion=rule_data['conclusion']
                        )
                        self.rules[disease].append(rule)
            
            # Load thresholds and risk levels
            self.thresholds = config.get('thresholds', {})
            self.risk_levels = config.get('risk_levels', {})
            
            logger.info(f"Loaded {sum(len(r) for r in self.rules.values())} rules for {len(self.rules)} diseases")
        
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            raise
    
    def _evaluate_condition(self, condition: Dict[str, Any], patient_data: Dict[str, Any]) -> bool:
        """
        Evaluate a single condition against patient data.
        
        Args:
            condition: Condition specification with operator and value
            patient_data: Patient's clinical data
        
        Returns:
            True if condition is satisfied, False otherwise
        """
        # Determine data category (symptom, vital, lab)
        category = None
        field_name = None
        
        for cat in ['symptom', 'vital', 'lab', 'demographic']:
            if cat in condition:
                category = cat + 's'  # Pluralize
                field_name = condition[cat]
                break
        
        if not category or not field_name:
            return False
        
        # Get patient value for this field
        patient_value = patient_data.get(category, {}).get(field_name)
        
        if patient_value is None:
            return False
        
        # Apply operator
        operator = condition.get('operator', '==')
        expected_value = condition.get('value')
        
        try:
            if operator == '==':
                return patient_value == expected_value
            elif operator == '!=':
                return patient_value != expected_value
            elif operator == '>':
                return float(patient_value) > float(expected_value)
            elif operator == '>=':
                return float(patient_value) >= float(expected_value)
            elif operator == '<':
                return float(patient_value) < float(expected_value)
            elif operator == '<=':
                return float(patient_value) <= float(expected_value)
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not evaluate condition: {e}")
            return False
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], patient_data: Dict[str, Any]) -> bool:
        """
        Evaluate complex condition logic (all, any, any_N_of).
        
        Args:
            conditions: Condition structure with logical operators
            patient_data: Patient's clinical data
        
        Returns:
            True if overall conditions are satisfied
        """
        # Handle 'all' conditions (AND logic)
        if 'all' in conditions:
            all_conditions = conditions['all']
            if not all(self._evaluate_condition(c, patient_data) for c in all_conditions):
                return False
        
        # Handle 'any' conditions (OR logic)
        if 'any' in conditions:
            any_conditions = conditions['any']
            if not any(self._evaluate_condition(c, patient_data) for c in any_conditions):
                return False
        
        # Handle 'any_N_of' conditions (at least N must be true)
        for key in conditions:
            if key.startswith('any_') and key.endswith('_of'):
                # Extract N from 'any_N_of'
                n_required = int(key.split('_')[1])
                any_n_conditions = conditions[key]
                satisfied_count = sum(
                    self._evaluate_condition(c, patient_data) 
                    for c in any_n_conditions
                )
                if satisfied_count < n_required:
                    return False
        
        return True
    
    def evaluate_rules(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform forward chaining inference on patient data.
        
        Args:
            patient_data: Dictionary containing symptoms, vitals, labs, demographics
        
        Returns:
            Dictionary with disease scores and fired rules
        """
        self.fired_rules = []
        disease_scores = {}
        risk_assessments = {}
        
        # Iterate through all disease rules
        for disease, rules in self.rules.items():
            disease_score = 0.0
            disease_risk = "low"
            disease_fired_rules = []
            
            for rule in rules:
                # Evaluate rule conditions
                if self._evaluate_conditions(rule.conditions, patient_data):
                    # Rule fired!
                    rule.fired = True
                    self.fired_rules.append(rule)
                    disease_fired_rules.append(rule)
                    
                    # Add probability boost from conclusion
                    conclusion = rule.conclusion
                    disease_score += conclusion.get('probability_boost', 0.0)
                    
                    # Update risk level if more severe
                    if 'risk_level' in conclusion:
                        rule_risk = conclusion['risk_level']
                        if self._is_more_severe(rule_risk, disease_risk):
                            disease_risk = rule_risk
                    
                    logger.info(f"Rule fired: {rule.name} for {disease}")
            
            # Normalize score to [0, 1]
            disease_scores[disease] = min(disease_score, 1.0)
            risk_assessments[disease] = disease_risk
        
        return {
            'disease_scores': disease_scores,
            'risk_assessments': risk_assessments,
            'fired_rules': self.fired_rules,
            'rule_count': len(self.fired_rules)
        }
    
    def _is_more_severe(self, risk1: str, risk2: str) -> bool:
        """Compare severity of two risk levels"""
        severity_order = ['low', 'moderate', 'high', 'severe']
        try:
            return severity_order.index(risk1) > severity_order.index(risk2)
        except ValueError:
            return False
    
    def get_rule_trace(self, include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Get trace of fired rules for explanation.
        
        Args:
            include_all: If True, include non-fired rules as well
        
        Returns:
            List of rule information dictionaries
        """
        trace = []
        
        rules_to_include = self.fired_rules if not include_all else [
            rule for rules_list in self.rules.values() 
            for rule in rules_list
        ]
        
        for rule in rules_to_include:
            trace.append({
                'name': rule.name,
                'description': rule.description,
                'fired': rule.fired,
                'disease': rule.conclusion.get('disease'),
                'confidence': rule.conclusion.get('confidence', 'medium'),
                'boost': rule.conclusion.get('probability_boost', 0.0)
            })
        
        return trace
    
    def reset(self):
        """Reset all fired rule flags"""
        self.fired_rules = []
        for disease_rules in self.rules.values():
            for rule in disease_rules:
                rule.fired = False


# Example usage
if __name__ == "__main__":
    # Test data
    test_patient = {
        'symptoms': {
            'fever': True,
            'headache': True,
            'rash': True,
            'nausea': True,
            'retro_orbital_pain': True
        },
        'vitals': {
            'temperature': 39.5,
            'oxygen_saturation': 96
        },
        'labs': {
            'platelet_count': 95000,
            'wbc_count': 3200
        }
    }
    
    engine = RuleEngine()
    results = engine.evaluate_rules(test_patient)
    
    print("\n=== Rule Engine Results ===")
    print(f"Disease Scores: {results['disease_scores']}")
    print(f"Risk Assessments: {results['risk_assessments']}")
    print(f"Fired Rules: {len(results['fired_rules'])}")
    
    print("\n=== Rule Trace ===")
    for rule_info in engine.get_rule_trace():
        print(f"- {rule_info['name']}: {rule_info['description']}")
