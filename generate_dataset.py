import pandas as pd
import random

# Diseases aur symptoms ki list (realistic)
diseases = [
    "Flu", "Pneumonia", "Migraine", "Common Cold", "Bronchitis", "Asthma", "Sinusitis", "Tonsillitis",
    "Strep Throat", "Gastroenteritis", "Urinary Tract Infection", "Allergies", "Arthritis", "Dengue",
    "Typhoid", "Malaria", "Hepatitis", "Diabetes", "Hypertension", "Anemia", "Kidney Stones",
    "Gallstones", "Appendicitis", "Tuberculosis"
]

symptoms = [
    "fever", "cough", "headache", "sore_throat", "chest_pain", "shortness_of_breath", "vision_changes",
    "runny_nose", "sneezing", "wheezing", "fatigue", "nausea", "sensitivity_to_light", "body_aches",
    "sensitivity_to_sound", "chest_tightness", "diarrhea", "vomiting", "abdominal_pain", "rash",
    "itching", "swelling", "joint_pain", "muscle_pain", "chills", "sweating", "dizziness", "weight_loss",
    "frequent_urination", "blood_in_urine", "jaundice", "dark_urine", "pale_stool", "high_blood_pressure",
    "low_blood_pressure", "anemia_symptoms", "coughing_blood", "night_sweats", "loss_of_appetite",
    "swollen_lymph_nodes", "back_pain", "side_pain", "burning_urination", "hives", "throat_irritation",
    "ear_pain", "nasal_congestion", "sore_muscles", "blurred_vision", "increased_thirst"
]

# Disease-symptom mapping (realistic combinations)
disease_symptom_map = {
    "Flu": ["fever", "cough", "headache", "sore_throat", "fatigue", "body_aches", "chills"],
    "Pneumonia": ["fever", "cough", "chest_pain", "shortness_of_breath", "fatigue", "chills", "sweating"],
    "Migraine": ["headache", "nausea", "sensitivity_to_light", "vision_changes", "sensitivity_to_sound"],
    "Common Cold": ["cough", "sore_throat", "runny_nose", "sneezing", "fever", "nasal_congestion"],
    "Bronchitis": ["cough", "fatigue", "chest_pain", "wheezing", "fever", "sore_throat"],
    "Asthma": ["wheezing", "shortness_of_breath", "chest_tightness", "cough"],
    "Sinusitis": ["headache", "nasal_congestion", "runny_nose", "sore_throat", "ear_pain"],
    "Tonsillitis": ["sore_throat", "fever", "swollen_lymph_nodes", "headache"],
    "Strep Throat": ["sore_throat", "fever", "swollen_lymph_nodes", "rash"],
    "Gastroenteritis": ["diarrhea", "vomiting", "abdominal_pain", "nausea", "fever"],
    "Urinary Tract Infection": ["burning_urination", "frequent_urination", "abdominal_pain", "blood_in_urine"],
    "Allergies": ["sneezing", "runny_nose", "itching", "hives", "swelling"],
    "Arthritis": ["joint_pain", "swelling", "muscle_pain", "fatigue"],
    "Dengue": ["fever", "headache", "rash", "joint_pain", "muscle_pain", "chills"],
    "Typhoid": ["fever", "fatigue", "abdominal_pain", "loss_of_appetite", "rash"],
    "Malaria": ["fever", "chills", "sweating", "headache", "nausea"],
    "Hepatitis": ["jaundice", "dark_urine", "pale_stool", "fatigue", "abdominal_pain"],
    "Diabetes": ["frequent_urination", "increased_thirst", "fatigue", "blurred_vision"],
    "Hypertension": ["high_blood_pressure", "headache", "dizziness", "blurred_vision"],
    "Anemia": ["fatigue", "anemia_symptoms", "dizziness", "pale_stool"],
    "Kidney Stones": ["side_pain", "blood_in_urine", "burning_urination", "nausea"],
    "Gallstones": ["abdominal_pain", "nausea", "vomiting", "jaundice"],
    "Appendicitis": ["abdominal_pain", "nausea", "vomiting", "fever"],
    "Tuberculosis": ["coughing_blood", "night_sweats", "weight_loss", "fever", "fatigue"]
}

# Dataset generate karo
data = []
for _ in range(1200):
    disease = random.choice(diseases)
    possible_symptoms = disease_symptom_map[disease]
    # Har row ke liye 3 symptoms randomly select karo
    selected_symptoms = random.sample(possible_symptoms, min(3, len(possible_symptoms)))
    if len(selected_symptoms) < 3:
        selected_symptoms.extend([""] * (3 - len(selected_symptoms)))
    data.append([disease] + selected_symptoms)

# DataFrame banao
df = pd.DataFrame(data, columns=["label", "Symptom_1", "Symptom_2", "Symptom_3"])

# CSV file mein save karo
df.to_csv("Symptom2Disease.csv", index=False)

print("Dataset generated and saved as Symptom2Disease.csv!")
print("Shape of dataset:", df.shape)
print("Unique diseases:", df['label'].nunique())
print("Sample data:")
print(df.head())