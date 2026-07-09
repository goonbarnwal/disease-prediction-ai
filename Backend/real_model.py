import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictorReal:
    def __init__(self):
        self.model = None
        self.disease_encoder = None
        self.all_symptoms = None
        self.df = None
        self.load_or_create_dataset()
    
    def load_or_create_dataset(self):
        """Real dataset load ya create kare"""
        # ⭐⭐⭐ YAHAN CHANGE KARNA HAI ⭐⭐⭐
        dataset_path = 'data/Disease and symptoms dataset.csv'
        # ⭐⭐⭐ ABOVE LINE IS CHANGED ⭐⭐⭐
        
        # Agar 'data' folder nahi hai to banao
        if not os.path.exists('data'):
            os.makedirs('data')
            print("📁 'data' folder created")
        
        try:
            # Real dataset ko load karein
            print(f"🔍 Looking for dataset at: {dataset_path}")
            self.df = pd.read_csv(dataset_path)
            print(f"✅ Real dataset loaded from: {dataset_path}")
            print(f"   📊 Total records: {len(self.df)}")
            
            # Dataset ke columns check karein
            print(f"   📝 Columns in dataset: {list(self.df.columns)}")
            
            # Disease column ka naam check karo (different cases me ho sakta hai)
            disease_col = None
            for col in self.df.columns:
                if 'disease' in col.lower():
                    disease_col = col
                    break
            
            if disease_col:
                print(f"   🩺 Unique diseases: {self.df[disease_col].nunique()}")
            else:
                print("   ⚠️  'Disease' column not found, using first column")
                disease_col = self.df.columns[0]
            
            # Symptoms columns count
            symptom_cols = [col for col in self.df.columns if col != disease_col]
            print(f"   🤒 Total symptoms columns: {len(symptom_cols)}")
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found at: {dataset_path}")
            print("📝 Creating synthetic dataset as fallback...")
            self.create_realistic_dataset()
            # Synthetic dataset save karo
            self.df.to_csv(dataset_path, index=False)
            print(f"💾 Synthetic dataset saved to: {dataset_path}")
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            print("📝 Creating synthetic dataset as fallback...")
            self.create_realistic_dataset()
        
        self.prepare_model()
    
    def create_realistic_dataset(self):
        """Agar real dataset nahi mila to synthetic banaye"""
        print("🔄 Creating synthetic dataset with 1000 cases...")
        
        # Real diseases and their common symptoms
        disease_symptoms = {
            'Common Cold': ['Fever', 'Cough', 'Runny Nose', 'Sneezing', 'Sore Throat', 'Fatigue', 'Headache'],
            'Influenza': ['Fever', 'Body Ache', 'Fatigue', 'Headache', 'Cough', 'Sore Throat', 'Chills'],
            'Migraine': ['Headache', 'Nausea', 'Sensitivity to Light', 'Vomiting', 'Dizziness', 'Fatigue'],
            'Allergic Rhinitis': ['Sneezing', 'Itchy Eyes', 'Runny Nose', 'Nasal Congestion', 'Cough', 'Fatigue'],
            'Gastroenteritis': ['Vomiting', 'Diarrhea', 'Abdominal Pain', 'Fever', 'Nausea', 'Fatigue'],
            'COVID-19': ['Fever', 'Cough', 'Loss of Taste', 'Loss of Smell', 'Fatigue', 'Body Ache'],
            'Pneumonia': ['Fever', 'Cough', 'Shortness of Breath', 'Chest Pain', 'Fatigue', 'Sweating'],
            'Bronchitis': ['Cough', 'Mucus Production', 'Fatigue', 'Shortness of Breath', 'Fever', 'Chest Discomfort'],
            'Sinusitis': ['Headache', 'Facial Pain', 'Nasal Congestion', 'Cough', 'Fever', 'Fatigue'],
            'Urinary Tract Infection': ['Burning Urination', 'Frequent Urination', 'Pelvic Pain', 'Fever', 'Fatigue']
        }
        
        # Generate synthetic cases
        cases = []
        for disease, symptoms in disease_symptoms.items():
            for _ in range(100):  # 100 cases per disease
                # Randomly select 3-5 symptoms
                num_symptoms = np.random.randint(3, 6)
                selected_symptoms = np.random.choice(symptoms, num_symptoms, replace=False)
                
                # Create a row with disease and symptoms
                row = {'Disease': disease}
                for i, symptom in enumerate(selected_symptoms, 1):
                    row[f'Symptom_{i}'] = symptom
                
                # Fill remaining symptom columns with NaN
                for i in range(len(selected_symptoms) + 1, 6):
                    row[f'Symptom_{i}'] = np.nan
                
                cases.append(row)
        
        self.df = pd.DataFrame(cases)
        print(f"✅ Synthetic dataset created with {len(self.df)} cases")
    
    def prepare_model(self):
        """Dataset se ML model prepare kare"""
        print("🔄 Preparing ML model from dataset...")
        
        # Disease column find karo
        disease_col = None
        for col in self.df.columns:
            if 'disease' in col.lower():
                disease_col = col
                break
        
        if not disease_col:
            disease_col = self.df.columns[0]
        
        # Symptoms columns
        symptom_cols = [col for col in self.df.columns if col != disease_col]
        
        # All unique symptoms collect kare
        all_symptoms = set()
        for col in symptom_cols:
            unique_symptoms = self.df[col].dropna().unique()
            all_symptoms.update(unique_symptoms)
        
        self.all_symptoms = list(all_symptoms)
        print(f"📊 Total unique symptoms found: {len(self.all_symptoms)}")
        print(f"🩺 Total unique diseases: {self.df[disease_col].nunique()}")
        
        # Create binary features for symptoms
        X = np.zeros((len(self.df), len(self.all_symptoms)))
        
        for idx, row in self.df.iterrows():
            for col in symptom_cols:
                symptom = row[col]
                if pd.notna(symptom) and symptom in self.all_symptoms:
                    X[idx, self.all_symptoms.index(symptom)] = 1
        
        # Encode diseases
        self.disease_encoder = LabelEncoder()
        y = self.disease_encoder.fit_transform(self.df[disease_col])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📈 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        print("🤖 Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Model Accuracy: {accuracy:.2%}")
        
        # Save model
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        joblib.dump(self.model, os.path.join(models_dir, 'real_disease_model.pkl'))
        joblib.dump(self.disease_encoder, os.path.join(models_dir, 'disease_encoder.pkl'))
        joblib.dump(self.all_symptoms, os.path.join(models_dir, 'symptoms_list.pkl'))
        
        print("✅ Real ML model trained and saved in 'models/' folder")
        print(f"💾 Model files saved:")
        print(f"   - {os.path.join(models_dir, 'real_disease_model.pkl')}")
        print(f"   - {os.path.join(models_dir, 'disease_encoder.pkl')}")
        print(f"   - {os.path.join(models_dir, 'symptoms_list.pkl')}")
    
    def predict(self, symptoms_text):
        """Real prediction with ML model"""
        # Convert symptoms text to list
        symptoms_list = [s.strip().title() for s in symptoms_text.lower().split(',')]
        print(f"🔍 Predicting for symptoms: {symptoms_list}")
        
        # Create input vector
        input_vector = np.zeros(len(self.all_symptoms))
        matched_symptoms = []
        
        for symptom in symptoms_list:
            # Try exact match first
            if symptom in self.all_symptoms:
                input_vector[self.all_symptoms.index(symptom)] = 1
                matched_symptoms.append(symptom)
            else:
                # Try partial match
                for idx, known_symptom in enumerate(self.all_symptoms):
                    if symptom.lower() in known_symptom.lower() or known_symptom.lower() in symptom.lower():
                        input_vector[idx] = 1
                        matched_symptoms.append(known_symptom)
                        break
        
        print(f"   Matched {len(matched_symptoms)} symptoms: {matched_symptoms}")
        
        # If no symptoms matched, return default
        if input_vector.sum() == 0:
            print("⚠️  No symptoms matched, returning default prediction")
            return self.get_default_prediction(symptoms_text)
        
        # Make prediction
        prediction = self.model.predict([input_vector])[0]
        disease = self.disease_encoder.inverse_transform([prediction])[0]
        
        # Get confidence
        probabilities = self.model.predict_proba([input_vector])[0]
        confidence = np.max(probabilities) * 100
        
        # Get top 3 diseases
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = self.disease_encoder.inverse_transform(top_3_indices)
        top_3_confidences = probabilities[top_3_indices] * 100
        
        # Get risk level
        risk_level = self.get_risk_level(disease)
        
        # Get description and recommendations
        description, recommendations = self.get_disease_info(disease)
        
        return {
            'disease': disease,
            'confidence': round(confidence, 2),
            'risk_level': risk_level,
            'description': description,
            'recommendations': recommendations,
            'matched_symptoms': matched_symptoms,
            'top_predictions': [
                {'disease': str(d), 'confidence': round(float(c), 2)} 
                for d, c in zip(top_3_diseases, top_3_confidences)
            ]
        }
    
    def get_risk_level(self, disease):
        """Disease ke hisaab se risk level determine kare"""
        risk_mapping = {
            'high': ['COVID-19', 'Pneumonia'],
            'medium': ['Influenza', 'Bronchitis', 'Gastroenteritis', 'Urinary Tract Infection'],
            'low': ['Common Cold', 'Allergic Rhinitis', 'Sinusitis', 'Migraine']
        }
        
        for risk, diseases in risk_mapping.items():
            if disease in diseases:
                return risk
        return 'low'
    
    def get_disease_info(self, disease):
        """Disease ka information aur recommendations"""
        disease_info = {
            'Common Cold': {
                'desc': 'Viral infection of the nose and throat (upper respiratory tract). Usually mild and resolves within 7-10 days.',
                'recommendations': [
                    'Get plenty of rest',
                    'Drink lots of fluids (water, soup)',
                    'Use over-the-counter cold medications',
                    'Gargle with warm salt water for sore throat',
                    'Use a humidifier to add moisture to the air'
                ]
            },
            'Influenza': {
                'desc': 'Viral infection affecting the respiratory system. More severe than common cold with sudden onset of symptoms.',
                'recommendations': [
                    'Rest and allow your body to recover',
                    'Stay hydrated with water and electrolyte solutions',
                    'Consider antiviral medication if prescribed early',
                    'Use fever reducers like acetaminophen or ibuprofen',
                    'Consult a doctor if symptoms worsen'
                ]
            },
            'COVID-19': {
                'desc': 'Respiratory illness caused by coronavirus SARS-CoV-2. Can range from mild to severe, with potential for long-term effects.',
                'recommendations': [
                    'Self-isolate to prevent spreading',
                    'Get tested to confirm diagnosis',
                    'Monitor oxygen levels with a pulse oximeter',
                    'Seek emergency care if difficulty breathing',
                    'Follow local health department guidelines'
                ]
            },
            'Pneumonia': {
                'desc': 'Infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.',
                'recommendations': [
                    'Seek immediate medical attention',
                    'Complete full course of prescribed antibiotics',
                    'Get plenty of rest',
                    'Stay well-hydrated',
                    'Use a humidifier to help with breathing'
                ]
            },
            'Migraine': {
                'desc': 'Neurological condition characterized by intense, debilitating headaches often accompanied by nausea and sensitivity to light/sound.',
                'recommendations': [
                    'Rest in a quiet, dark room',
                    'Apply cold or warm compresses to head/neck',
                    'Stay hydrated',
                    'Consider prescription migraine medication',
                    'Identify and avoid personal migraine triggers'
                ]
            }
        }
        
        # Default agar disease info nahi hai
        default_info = {
            'desc': f'{disease}. Consult a healthcare professional for accurate diagnosis and personalized treatment plan.',
            'recommendations': [
                'Consult a doctor for proper diagnosis',
                'Monitor and track your symptoms',
                'Get adequate rest and sleep',
                'Stay hydrated with water',
                'Avoid self-medication without medical advice'
            ]
        }
        
        info = disease_info.get(disease, default_info)
        return info['desc'], info['recommendations']
    
    def get_default_prediction(self, symptoms_text):
        """Agar model kuch predict nahi kar paya"""
        return {
            'disease': 'General Medical Consultation Required',
            'confidence': 50,
            'risk_level': 'medium',
            'description': 'Symptoms pattern not clearly matching known disease patterns in our database. Please consult a healthcare professional for accurate diagnosis.',
            'recommendations': [
                'Schedule an appointment with a doctor',
                'Keep a symptom diary with timing and severity',
                'Monitor for any new or worsening symptoms',
                'Rest and maintain hydration',
                'Avoid self-diagnosis from unreliable sources'
            ],
            'matched_symptoms': [],
            'top_predictions': []
        }

# Global instance
real_predictor = DiseasePredictorReal()