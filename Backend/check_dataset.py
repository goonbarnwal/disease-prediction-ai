# check_dataset.py
import pandas as pd
import os

print("=" * 50)
print("🔍 CHECKING DATASET")
print("=" * 50)

# Check current folder
print("Current folder:", os.getcwd())

# Check what files exist in data folder
data_folder = 'data'
if os.path.exists(data_folder):
    print(f"\n📂 Files in {data_folder}/:")
    for file in os.listdir(data_folder):
        print(f"   - {file}")
else:
    print(f"\n❌ {data_folder} folder not found!")

# Try to load dataset
print("\n📊 TRYING TO LOAD DATASET...")
dataset_files = [
    'real_dataset.csv',
    'dataset.csv', 
    'new_dataset.csv',
    'disease_symptoms.csv'
]

df = None
for file in dataset_files:
    file_path = f'data/{file}'
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded: {file}")
            break
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

if df is None:
    print("\n❌ No dataset file found!")
    print("Please make sure your dataset is in data/ folder")
    exit()

# Print dataset info
print(f"\n📈 DATASET INFO:")
print(f"   Shape: {df.shape} (rows, columns)")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n👀 FIRST 3 ROWS:")
print(df.head(3))

# Check for disease column
disease_cols = [col for col in df.columns if 'disease' in col.lower()]
if disease_cols:
    disease_col = disease_cols[0]
    print(f"\n🏥 DISEASE COLUMN: '{disease_col}'")
    print(f"   Unique diseases: {df[disease_col].nunique()}")
    print(f"   Sample diseases: {df[disease_col].unique()[:10]}")
else:
    print("\n⚠️  No disease column found!")
    print("   Dataset columns:", df.columns.tolist())

# Check first value to understand format
print(f"\n🔍 FORMAT ANALYSIS:")
first_val = df.iloc[0, 0] if len(df) > 0 else "Empty"
print(f"   First cell value: {first_val}")
print(f"   Type: {type(first_val)}")

if df.shape[1] > 10:
    print(f"\n📝 Sample data (first 2 rows, first 5 columns):")
    print(df.iloc[:2, :5])