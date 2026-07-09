# ============================================
# DISEASE PREDICTION - FIXED FOR YOUR FOLDER
# Save as: Backend/disease_model.py
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("=" * 60)
print("🤖 DISEASE PREDICTION AI - KAGGLE DATASET")
print("=" * 60)

# ============================================
# STEP 1: LOAD FILES WITH CORRECT PATHS
# ============================================
print("\n📂 Loading files from data/ folder...")

try:
    # Load Kaggle dataset (Training.csv)
    df_main = pd.read_csv('data/Training.csv')
    print(f"✅ Training.csv: {df_main.shape[0]} rows, {df_main.shape[1]} cols")
    
    # Check if 'prognosis' column exists (Kaggle format)
    if 'prognosis' in df_main.columns:
        df_main = df_main.rename(columns={'prognosis': 'Disease'})
        print("✅ Renamed 'prognosis' to 'Disease'")
    
    # Get all symptom columns (all except Disease)
    symptom_cols = [col for col in df_main.columns if col != 'Disease']
    print(f"✅ {len(symptom_cols)} symptoms found")
    
    # Load description and precautions
    df_desc = pd.read_csv('data/symptom_Description.csv')
    df_prec = pd.read_csv('data/symptom_precaution.csv')
    print("✅ Description and precautions loaded")
    
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print("\n📁 Files needed in data/ folder:")
    print("   Training.csv")
    print("   symptom_Description.csv") 
    print("   symptom_precaution.csv")
    print("\n📍 Current files in data/ folder:")
    if os.path.exists('data'):
        for file in os.listdir('data'):
            print(f"   - {file}")
    exit()

# ============================================
# STEP 2: TRAIN MODEL
# ============================================
print("\n🤖 Training model...")

X = df_main.drop('Disease', axis=1)
y = df_main['Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train) * 100
test_acc = model.score(X_test, y_test) * 100

print(f"✅ Model trained!")
print(f"   Training Accuracy: {train_acc:.2f}%")
print(f"   Testing Accuracy: {test_acc:.2f}%")

# ============================================
# STEP 3: SAVE MODEL
# ============================================
print("\n💾 Saving files...")

os.makedirs('models', exist_ok=True)
os.makedirs('data_processed', exist_ok=True)

# Save model
joblib.dump(model, 'models/disease_model.pkl')
print("✅ Model saved: models/disease_model.pkl")

# Save symptoms list
symptoms = X.columns.tolist()
with open('data_processed/all_symptoms.txt', 'w') as f:
    for symptom in symptoms:
        f.write(symptom + '\n')
print(f"✅ Symptoms saved: {len(symptoms)}")

# Save diseases list
with open('data_processed/all_diseases.txt', 'w') as f:
    for disease in sorted(model.classes_):
        f.write(disease + '\n')
print(f"✅ Diseases saved: {len(model.classes_)}")

# ============================================
# STEP 4: TEST
# ============================================
print("\n🧪 Testing predictions...")

# Real test cases
tests = [
    (['itching', 'skin_rash'], "Should be Fungal infection"),
    (['continuous_sneezing', 'shivering'], "Should be Allergy"),
    (['vomiting', 'sunken_eyes'], "Should be Gastroenteritis")
]

for symptom_list, expected in tests:
    input_vec = [1 if s in symptom_list else 0 for s in symptoms]
    input_vec = np.array(input_vec).reshape(1, -1)
    
    pred = model.predict(input_vec)[0]
    conf = max(model.predict_proba(input_vec)[0]) * 100
    
    print(f"   {symptom_list}")
    print(f"     → {pred} ({conf:.1f}%)")
    print(f"     💡 {expected}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)