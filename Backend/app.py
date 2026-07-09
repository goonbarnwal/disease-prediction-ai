# ============================================
# DISEASE PREDICTION BACKEND - FIXED
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

print("=" * 60)
print("🚀 DISEASE PREDICTION AI - STARTING...")
print("=" * 60)

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD THE CORRECT MODEL
# ============================================
print("\n🔍 Loading AI model...")

# List all available model files
models_dir = 'models'
if os.path.exists(models_dir):
    print("Available model files:")
    for file in os.listdir(models_dir):
        if file.endswith('.pkl'):
            size = os.path.getsize(os.path.join(models_dir, file))
            print(f"  - {file} ({size:,} bytes)")

# Try to load the correct model (disease_model.pkl is the new one)
MODEL_PATHS = [
    ('models/disease_model.pkl', 'New Kaggle Model'),
    ('models/real_disease_model.pkl', 'Real Disease Model'),
    ('models/disease_encoder.pkl', 'Disease Encoder')
]

model = None
model_name = None

for path, name in MODEL_PATHS:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            model_name = name
            print(f"\n✅ LOADED: {name}")
            print(f"   File: {path}")
            print(f"   Size: {os.path.getsize(path):,} bytes")
            
            # Check if it's a proper model
            if hasattr(model, 'predict'):
                print(f"   Type: {type(model).__name__}")
                if hasattr(model, 'classes_'):
                    print(f"   Can predict: {len(model.classes_)} diseases")
            break
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
            continue

if model is None:
    print("\n❌ No valid model found!")
    print("   Please run disease_model.py to train a new model")
    model = None

# ============================================
# LOAD SYMPTOMS LIST
# ============================================
print("\n📋 Loading symptoms list...")

ALL_SYMPTOMS = []
symptoms_file = 'data_processed/all_symptoms.txt'

if os.path.exists(symptoms_file):
    try:
        with open(symptoms_file, 'r') as f:
            ALL_SYMPTOMS = [line.strip() for line in f]
        print(f"✅ Loaded {len(ALL_SYMPTOMS)} symptoms")
    except Exception as e:
        print(f"❌ Error loading symptoms: {e}")
        ALL_SYMPTOMS = []
else:
    print(f"❌ Symptoms file not found: {symptoms_file}")
    # Create fallback list
    ALL_SYMPTOMS = [
        'itching', 'skin_rash', 'continuous_sneezing', 'shivering', 'chills',
        'fever', 'cough', 'headache', 'fatigue', 'vomiting'
    ]
    print(f"⚠️  Using fallback: {len(ALL_SYMPTOMS)} symptoms")

# ============================================
# LOAD DISEASE DATA
# ============================================
print("\n📚 Loading disease data...")

try:
    df_desc = pd.read_csv('data/symptom_Description.csv')
    df_prec = pd.read_csv('data/symptom_precaution.csv')
    print("✅ Disease data loaded")
except Exception as e:
    print(f"⚠️  Error loading disease data: {e}")
    df_desc = pd.DataFrame()
    df_prec = pd.DataFrame()

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    return jsonify({
        "app": "Disease Prediction AI",
        "status": "running",
        "model": model_name if model_name else "not loaded",
        "model_type": type(model).__name__ if model else "none",
        "symptoms": len(ALL_SYMPTOMS),
        "api": {
            "/api/health": "GET - Server status",
            "/api/symptoms": "GET - List symptoms",
            "/api/predict": "POST - Predict disease"
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": model_name if model_name else "not loaded",
        "model_ready": model is not None,
        "symptoms_loaded": len(ALL_SYMPTOMS) > 0,
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({
        "status": "success",
        "count": len(ALL_SYMPTOMS),
        "symptoms": ALL_SYMPTOMS[:50]  # First 50 only
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # ... [EXISTING CODE: Get data, validate symptoms, convert to numerical] ...
        
        # ========== EXISTING CODE ==========
        # Get data from request
        data = request.json
        symptoms = data.get('symptoms', [])
        
        # Validate and clean symptoms
        valid_symptoms = []
        for s in symptoms:
            s_clean = str(s).strip().lower().replace(' ', '_')
            if s_clean in ALL_SYMPTOMS:
                valid_symptoms.append(s_clean)
        
        # Convert to numerical
        input_vector = [1 if s in valid_symptoms else 0 for s in ALL_SYMPTOMS]
        input_vector = np.array(input_vector).reshape(1, -1)
        
        # Get prediction
        predicted_disease = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]
        confidence = float(np.max(probabilities) * 100)
        # ========== END EXISTING CODE ==========
        
        # ========== 🚨 ADD THIS VALIDATION CODE HERE ==========
        classes = model.classes_
        
        def is_medically_valid(disease, symptoms):
            """Quick medical validation"""
            invalid_combinations = {
                'Paralysis (brain hemorrhage)': ['fever', 'cough', 'sneezing', 'runny_nose'],
                'Heart attack': ['fever', 'cough', 'itching', 'sneezing'],
                'Fungal infection': ['chest_pain', 'shortness_of_breath'],
                'Diabetes': ['chest_pain', 'shortness_of_breath'],
                'Malaria': ['chest_pain', 'shortness_of_breath'],
                'Typhoid': ['chest_pain', 'shortness_of_breath'],
                'Hepatitis A': ['chest_pain', 'shortness_of_breath'],
                'Common Cold': ['chest_pain', 'shortness_of_breath', 'weakness_of_one_body_side']
            }
            
            if disease in invalid_combinations:
                for symptom in symptoms:
                    if symptom in invalid_combinations[disease]:
                        return False
            return True
        
        # Check if prediction is medically valid
        original_disease = predicted_disease
        original_confidence = confidence
        
        if not is_medically_valid(predicted_disease, valid_symptoms):
            # Find better prediction from top 5
            top_5_idx = np.argsort(probabilities)[-5:][::-1]
            
            for idx in top_5_idx:
                alt_disease = classes[idx]
                alt_confidence = probabilities[idx] * 100
                
                if (is_medically_valid(alt_disease, valid_symptoms) and 
                    alt_confidence > 30):
                    predicted_disease = alt_disease
                    confidence = alt_confidence
                    print(f"🔄 Corrected: {original_disease} → {predicted_disease}")
                    break
            else:
                # No valid alternative found, use common sense
                if 'fever' in valid_symptoms and 'cough' in valid_symptoms:
                    predicted_disease = 'Common Cold'
                    confidence = 70.0
                    print(f"🔄 Set to Common Cold (fever+cough)")
                elif 'itching' in valid_symptoms and 'skin_rash' in valid_symptoms:
                    predicted_disease = 'Fungal infection'
                    confidence = 75.0
                    print(f"🔄 Set to Fungal infection (itching+rash)")
                elif 'chest_pain' in valid_symptoms:
                    predicted_disease = 'GERD'
                    confidence = 60.0
                    print(f"🔄 Set to GERD (chest pain)")
        # ========== END VALIDATION CODE ==========
        
        # ========== EXISTING CODE CONTINUES ==========
        # Get description
        description = "No description available"
        if not df_desc.empty:
            desc_match = df_desc[df_desc['Disease'] == predicted_disease]
            if not desc_match.empty:
                description = desc_match['Description'].values[0]
        
        # Get precautions
        precautions = ["Consult a healthcare professional"]
        if not df_prec.empty:
            prec_match = df_prec[df_prec['Disease'] == predicted_disease]
            if not prec_match.empty:
                precautions = prec_match.iloc[0, 1:].dropna().tolist()
        
        # Determine risk level
        if confidence >= 80:
            risk_level = "High"
        elif confidence >= 60:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Prepare response
        response = {
            "status": "success",
            "prediction": {
                "disease": predicted_disease,
                "confidence": round(confidence, 2),
                "description": description,
                "precautions": precautions,
                "risk_level": risk_level,
                "recommendation": "Consult a doctor for accurate diagnosis"
            },
            "input_analysis": {
                "symptoms_provided": symptoms,
                "valid_symptoms": valid_symptoms,
                "count": len(valid_symptoms)
            }
        }
        
        # Add warning if correction happened
        if original_disease != predicted_disease:
            response["validation_note"] = f"Corrected from {original_disease} (medically unlikely)"
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
# ============================================
# START
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 BACKEND READY!")
    print(f"   Model: {model_name if model_name else 'Demo Mode'}")
    print(f"   Symptoms: {len(ALL_SYMPTOMS)}")
    print("   URL: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)