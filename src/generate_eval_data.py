"""
Generate synthetic evaluation data for Canadian ER Triage Assessment using AI Foundry SDK.

This script creates diverse patient scenarios with varying CTAS levels (1-5) for testing
the multi-agent triage system. Each scenario includes patient demographics, chief complaints,
symptoms, and expected triage outcomes.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class SyntheticTriageDataGenerator:
    """Generate synthetic evaluation data for ER triage assessment."""
    
    def __init__(self):
        self.output_dir = Path("src/foundry_agents/config")
        self.output_file = self.output_dir / "evaluation_data.jsonl"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Patient names from the FHIR dataset
        self.patient_names = [
            "Aaron697 Stanton715",
            "Abdul218 Gusikowski974", 
            "Abel832 Keebler762",
            "Abram53 Kihn564",
            "Ada662 Nader710",
            "Adalberto916 Feil794",
            "Adan632 Bode78",
            "Adelaide981 Tremblay80",
            "Adelina682 Cruickshank494",
            "Adella39 Morar593",
            "Adina377 Effertz744",
            "Adolfo777 McLaughlin530",
            "Adolph80 Jakubowski832",
            "Agustin437 Gorczany269",
            "Ahmad985 Cruickshank494",
            "Ahmed109 Boyle917",
            "Aisha756 Harris789",
            "Akiko835 Schaden604",
            "Alaine226 Skiles927",
            "Alan320 Cormier289"
        ]
        
        # Birth dates (ages 25-85)
        self.birth_dates = self._generate_birth_dates()
        
    def _generate_birth_dates(self) -> List[str]:
        """Generate realistic birth dates for different age groups."""
        birth_dates = []
        current_year = datetime.now().year
        
        for _ in range(len(self.patient_names)):
            # Generate ages between 25-85
            age = random.randint(25, 85)
            birth_year = current_year - age
            birth_month = random.randint(1, 12)
            birth_day = random.randint(1, 28)  # Safe day range
            birth_dates.append(f"{birth_year}-{birth_month:02d}-{birth_day:02d}")
            
        return birth_dates

    def generate_ctas_1_scenarios(self) -> List[Dict[str, Any]]:
        """Generate CTAS Level 1 (Resuscitation) scenarios - Life-threatening."""
        scenarios = [
            {
                "chief_complaint": "Cardiac arrest - patient found unconscious, no pulse",
                "symptoms": [
                    "Unconscious",
                    "No pulse palpable", 
                    "Not breathing",
                    "Cyanotic",
                    "CPR in progress"
                ],
                "vitals": {
                    "pulse": "0",
                    "bp": "Not obtainable",
                    "respiratory_rate": "0",
                    "temperature": "Unknown",
                    "oxygen_saturation": "Undetectable"
                },
                "pain_level": "Unable to assess",
                "onset": "Found down 5 minutes ago",
                "expected_ctas": 1,
                "expected_actions": ["Immediate resuscitation", "CPR", "Advanced cardiac life support"],
                "expected_wait_time": "Immediate (0 minutes)"
            },
            {
                "chief_complaint": "Severe trauma from motor vehicle accident with multiple injuries",
                "symptoms": [
                    "Altered level of consciousness",
                    "Severe abdominal pain",
                    "Obvious deformity left leg",
                    "Heavy bleeding from head laceration",
                    "Difficulty breathing"
                ],
                "vitals": {
                    "pulse": "120",
                    "bp": "80/50",
                    "respiratory_rate": "28",
                    "temperature": "36.2°C", 
                    "oxygen_saturation": "88%"
                },
                "pain_level": "Unable to assess reliably",
                "onset": "30 minutes ago",
                "expected_ctas": 1,
                "expected_actions": ["Trauma team activation", "IV access", "Blood transfusion preparation"],
                "expected_wait_time": "Immediate (0 minutes)"
            },
            {
                "chief_complaint": "Anaphylactic reaction after eating shellfish",
                "symptoms": [
                    "Severe difficulty breathing",
                    "Facial and throat swelling",
                    "Full body hives",
                    "Vomiting",
                    "Feeling of impending doom"
                ],
                "vitals": {
                    "pulse": "140",
                    "bp": "70/40", 
                    "respiratory_rate": "32",
                    "temperature": "37.1°C",
                    "oxygen_saturation": "85%"
                },
                "pain_level": "7/10",
                "onset": "15 minutes ago",
                "expected_ctas": 1,
                "expected_actions": ["Epinephrine administration", "Airway management", "IV steroids"],
                "expected_wait_time": "Immediate (0 minutes)"
            }
        ]
        return scenarios

    def generate_ctas_2_scenarios(self) -> List[Dict[str, Any]]:
        """Generate CTAS Level 2 (Emergent) scenarios - Potential threat to life/limb."""
        scenarios = [
            {
                "chief_complaint": "Severe chest pain with radiation to left arm and jaw",
                "symptoms": [
                    "Crushing chest pain radiating to left arm",
                    "Shortness of breath",
                    "Nausea and vomiting", 
                    "Diaphoresis",
                    "Anxiety"
                ],
                "vitals": {
                    "pulse": "110",
                    "bp": "160/100",
                    "respiratory_rate": "24",
                    "temperature": "37.0°C",
                    "oxygen_saturation": "92%"
                },
                "pain_level": "9/10",
                "onset": "2 hours ago",
                "expected_ctas": 2,
                "expected_actions": ["12-lead ECG", "Cardiac enzymes", "Cardiology consult"],
                "expected_wait_time": "Within 15 minutes"
            },
            {
                "chief_complaint": "Severe abdominal pain with vomiting and fever",
                "symptoms": [
                    "Severe right lower quadrant pain",
                    "Nausea and vomiting",
                    "Fever and chills",
                    "Unable to walk upright",
                    "Pain worse with movement"
                ],
                "vitals": {
                    "pulse": "105",
                    "bp": "130/85",
                    "respiratory_rate": "22", 
                    "temperature": "38.8°C",
                    "oxygen_saturation": "96%"
                },
                "pain_level": "8/10",
                "onset": "6 hours ago, worsening",
                "expected_ctas": 2,
                "expected_actions": ["CT abdomen", "Surgical consultation", "IV antibiotics"],
                "expected_wait_time": "Within 15 minutes"
            },
            {
                "chief_complaint": "Difficulty breathing and wheezing after exposure to smoke",
                "symptoms": [
                    "Severe shortness of breath",
                    "Audible wheezing",
                    "Coughing up black sputum",
                    "Chest tightness",
                    "Singed nasal hairs noted"
                ],
                "vitals": {
                    "pulse": "115",
                    "bp": "140/90",
                    "respiratory_rate": "30",
                    "temperature": "37.2°C",
                    "oxygen_saturation": "89%"
                },
                "pain_level": "6/10",
                "onset": "1 hour ago",
                "expected_ctas": 2,
                "expected_actions": ["High-flow oxygen", "Bronchodilators", "Chest X-ray"],
                "expected_wait_time": "Within 15 minutes"
            }
        ]
        return scenarios

    def generate_ctas_3_scenarios(self) -> List[Dict[str, Any]]:
        """Generate CTAS Level 3 (Urgent) scenarios - Potentially serious."""
        scenarios = [
            {
                "chief_complaint": "Severe headache with visual changes and neck stiffness",
                "symptoms": [
                    "Worst headache of life",
                    "Blurred vision",
                    "Neck stiffness",
                    "Photophobia",
                    "Mild confusion"
                ],
                "vitals": {
                    "pulse": "95",
                    "bp": "150/95",
                    "respiratory_rate": "18",
                    "temperature": "37.8°C",
                    "oxygen_saturation": "97%"
                },
                "pain_level": "8/10",
                "onset": "4 hours ago",
                "expected_ctas": 3,
                "expected_actions": ["CT head", "Lumbar puncture consideration", "Neurological assessment"],
                "expected_wait_time": "Within 30 minutes"
            },
            {
                "chief_complaint": "Deep laceration to forearm from broken glass",
                "symptoms": [
                    "4-inch deep laceration on forearm",
                    "Active bleeding controlled with pressure",
                    "Possible tendon involvement",
                    "Numbness in fingers",
                    "Pain with finger movement"
                ],
                "vitals": {
                    "pulse": "88",
                    "bp": "125/80",
                    "respiratory_rate": "16",
                    "temperature": "36.8°C",
                    "oxygen_saturation": "99%"
                },
                "pain_level": "7/10",
                "onset": "45 minutes ago",
                "expected_ctas": 3,
                "expected_actions": ["Wound exploration", "Tetanus update", "Orthopedic consultation"],
                "expected_wait_time": "Within 30 minutes"
            },
            {
                "chief_complaint": "Diabetic with high blood sugar and vomiting",
                "symptoms": [
                    "Blood glucose >400 mg/dL",
                    "Persistent vomiting",
                    "Fruity breath odor", 
                    "Dehydration",
                    "Weakness and fatigue"
                ],
                "vitals": {
                    "pulse": "102",
                    "bp": "110/70",
                    "respiratory_rate": "20",
                    "temperature": "37.1°C",
                    "oxygen_saturation": "98%"
                },
                "pain_level": "4/10",
                "onset": "12 hours ago",
                "expected_ctas": 3,
                "expected_actions": ["Blood gas analysis", "IV insulin", "Fluid resuscitation"],
                "expected_wait_time": "Within 30 minutes"
            }
        ]
        return scenarios

    def generate_ctas_4_scenarios(self) -> List[Dict[str, Any]]:
        """Generate CTAS Level 4 (Less Urgent) scenarios."""
        scenarios = [
            {
                "chief_complaint": "Ankle injury from fall with swelling and pain",
                "symptoms": [
                    "Ankle swelling and bruising",
                    "Unable to bear weight",
                    "Pain with movement",
                    "No obvious deformity",
                    "Good circulation and sensation"
                ],
                "vitals": {
                    "pulse": "82",
                    "bp": "130/85",
                    "respiratory_rate": "16",
                    "temperature": "36.7°C",
                    "oxygen_saturation": "99%"
                },
                "pain_level": "6/10",
                "onset": "2 hours ago",
                "expected_ctas": 4,
                "expected_actions": ["X-ray ankle", "Pain management", "Orthopedic assessment"],
                "expected_wait_time": "Within 60 minutes"
            },
            {
                "chief_complaint": "Urinary tract infection symptoms with fever",
                "symptoms": [
                    "Burning with urination",
                    "Frequent urination",
                    "Cloudy urine",
                    "Low-grade fever",
                    "Lower abdominal discomfort"
                ],
                "vitals": {
                    "pulse": "78",
                    "bp": "120/75",
                    "respiratory_rate": "14", 
                    "temperature": "37.6°C",
                    "oxygen_saturation": "99%"
                },
                "pain_level": "4/10",
                "onset": "2 days ago, worsening",
                "expected_ctas": 4,
                "expected_actions": ["Urinalysis", "Urine culture", "Antibiotic therapy"],
                "expected_wait_time": "Within 60 minutes"
            },
            {
                "chief_complaint": "Migraine headache with nausea in known migraineur",
                "symptoms": [
                    "Throbbing headache",
                    "Nausea without vomiting",
                    "Sensitivity to light",
                    "Typical migraine pattern",
                    "No neurological deficits"
                ],
                "vitals": {
                    "pulse": "72",
                    "bp": "125/82",
                    "respiratory_rate": "15",
                    "temperature": "36.6°C",
                    "oxygen_saturation": "99%"
                },
                "pain_level": "7/10",
                "onset": "6 hours ago",
                "expected_ctas": 4,
                "expected_actions": ["Migraine medications", "IV hydration", "Dark quiet room"],
                "expected_wait_time": "Within 60 minutes"
            }
        ]
        return scenarios

    def generate_ctas_5_scenarios(self) -> List[Dict[str, Any]]:
        """Generate CTAS Level 5 (Non-urgent) scenarios."""
        scenarios = [
            {
                "chief_complaint": "Minor cut on finger needing wound care",
                "symptoms": [
                    "Small laceration on index finger",  
                    "Minimal bleeding",
                    "Good sensation and movement",
                    "No signs of infection",
                    "Clean wound edges"
                ],
                "vitals": {
                    "pulse": "68",
                    "bp": "118/72",
                    "respiratory_rate": "14",
                    "temperature": "36.5°C",
                    "oxygen_saturation": "99%"
                },
                "pain_level": "2/10",
                "onset": "3 hours ago",
                "expected_ctas": 5,
                "expected_actions": ["Wound cleaning", "Steri-strips or sutures", "Tetanus status"],
                "expected_wait_time": "Within 120 minutes"
            },
            {
                "chief_complaint": "Prescription refill request for chronic medication",
                "symptoms": [
                    "Ran out of blood pressure medication",
                    "No acute symptoms",
                    "Regular follow-up with family doctor",
                    "Stable chronic condition",
                    "No medication side effects"
                ],
                "vitals": {
                    "pulse": "75",
                    "bp": "135/88",
                    "respiratory_rate": "16",
                    "temperature": "36.8°C",
                    "oxygen_saturation": "99%"
                },
                "pain_level": "0/10",
                "onset": "N/A",
                "expected_ctas": 5,
                "expected_actions": ["Medication review", "Prescription renewal", "Family doctor referral"],
                "expected_wait_time": "Within 120 minutes"
            },
            {
                "chief_complaint": "Mild cold symptoms and cough for 3 days",
                "symptoms": [
                    "Runny nose",
                    "Mild cough",
                    "Slight sore throat",
                    "No fever",
                    "Able to carry out normal activities"
                ],
                "vitals": {
                    "pulse": "70",
                    "bp": "115/75",
                    "respiratory_rate": "15",
                    "temperature": "36.4°C",
                    "oxygen_saturation": "99%"
                },
                "pain_level": "1/10",
                "onset": "3 days ago",
                "expected_ctas": 5,
                "expected_actions": ["Symptomatic care", "Rest and fluids", "Return if worsening"],
                "expected_wait_time": "Within 120 minutes"
            }
        ]
        return scenarios

    def create_evaluation_record(self, patient_name: str, birth_date: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete evaluation record for a patient scenario."""
        
        # Create the patient triage request prompt
        symptoms_text = "\n- ".join(scenario["symptoms"])
        vitals_text = "\n- ".join([f"{k}: {v}" for k, v in scenario["vitals"].items()])
        
        user_prompt = f"""Patient Triage Request:

Patient: {patient_name} (DOB: {birth_date})

Chief Complaint: {scenario["chief_complaint"]}

Current Symptoms:
- {symptoms_text}

Vital Signs:
- {vitals_text}

Pain Level: {scenario["pain_level"]}
Onset: {scenario["onset"]}

Please coordinate the triage assessment by:
1. First retrieving the patient's medical history, immunizations, and diagnostic reports
2. Then performing a CTAS triage assessment based on the current presentation and historical data
3. Provide a comprehensive triage decision with rationale"""

        # Create the expected response
        expected_response = f"""CTAS Level: {scenario["expected_ctas"]}

Recommended Actions:
- {chr(10).join([f"• {action}" for action in scenario["expected_actions"]])}

Expected Wait Time: {scenario["expected_wait_time"]}

Rationale: Patient presents with {scenario["chief_complaint"].lower()} requiring CTAS Level {scenario["expected_ctas"]} assessment based on symptom severity and potential for deterioration."""

        return {
            "patient_name": patient_name,
            "patient_dob": birth_date,
            "input": user_prompt,
            "expected_output": expected_response,
            "expected_ctas_level": scenario["expected_ctas"],
            "scenario_type": f"CTAS Level {scenario['expected_ctas']}",
            "chief_complaint": scenario["chief_complaint"],
            "pain_level": scenario["pain_level"],
            "onset": scenario["onset"]
        }

    def generate_all_scenarios(self) -> List[Dict[str, Any]]:
        """Generate all evaluation scenarios across all CTAS levels."""
        all_scenarios = []
        
        # Get scenarios for each CTAS level
        ctas_1_scenarios = self.generate_ctas_1_scenarios()
        ctas_2_scenarios = self.generate_ctas_2_scenarios() 
        ctas_3_scenarios = self.generate_ctas_3_scenarios()
        ctas_4_scenarios = self.generate_ctas_4_scenarios()
        ctas_5_scenarios = self.generate_ctas_5_scenarios()
        
        # Combine all scenarios
        all_scenario_groups = [
            ctas_1_scenarios,
            ctas_2_scenarios, 
            ctas_3_scenarios,
            ctas_4_scenarios,
            ctas_5_scenarios
        ]
        
        patient_index = 0
        
        # Create evaluation records for each scenario
        for scenario_group in all_scenario_groups:
            for scenario in scenario_group:
                if patient_index < len(self.patient_names):
                    patient_name = self.patient_names[patient_index]
                    birth_date = self.birth_dates[patient_index]
                    
                    eval_record = self.create_evaluation_record(patient_name, birth_date, scenario)
                    all_scenarios.append(eval_record)
                    
                    patient_index += 1
        
        return all_scenarios

    def save_to_jsonl(self, scenarios: List[Dict[str, Any]]) -> None:
        """Save scenarios to JSONL file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario, ensure_ascii=False) + '\n')
        
        print(f"Generated {len(scenarios)} evaluation scenarios")
        print(f"Saved to: {self.output_file}")
        
        # Print summary statistics
        ctas_counts = {}
        for scenario in scenarios:
            ctas_level = scenario['expected_ctas_level']
            ctas_counts[ctas_level] = ctas_counts.get(ctas_level, 0) + 1
        
        print("\nScenario distribution by CTAS level:")
        for level in sorted(ctas_counts.keys()):
            print(f"  Level {level}: {ctas_counts[level]} scenarios")

    def generate(self) -> None:
        """Generate and save all evaluation data."""
        print("Generating synthetic ER triage evaluation data...")
        
        scenarios = self.generate_all_scenarios()
        self.save_to_jsonl(scenarios)
        
        print("\nEvaluation data generation complete!")
        print(f"File location: {self.output_file.absolute()}")


def main():
    """Main function to generate synthetic evaluation data."""
    generator = SyntheticTriageDataGenerator()
    generator.generate()


if __name__ == "__main__":
    main()