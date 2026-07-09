# quick_check.py
import pandas as pd

print("=" * 50)
print("🔍 CHECKING KAGGLE DATASET FILES")
print("=" * 50)

print("\n1. Checking Training.csv...")
train_df = pd.read_csv('data/Training.csv')
print(f"   ✅ Loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

print(f"\n   First 5 column names:")
for i, col in enumerate(train_df.columns[:5], 1):
    print(f"   {i}. {col}")

print(f"\n   Disease column (likely 'prognosis'):")
disease_cols = [col for col in train_df.columns if 'prognosis' in col.lower() or 'disease' in col.lower()]
if disease_cols:
    print(f"   Found: {disease_cols[0]}")
    print(f"   Unique values: {train_df[disease_cols[0]].nunique()}")
    print(f"   Sample: {train_df[disease_cols[0]].unique()[:5]}")
else:
    print("   ⚠️  No disease column found")

print("\n2. Checking Testing.csv...")
test_df = pd.read_csv('data/Testing.csv')
print(f"   ✅ Loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

print("\n3. Dataset format check:")
print("   First row of data (first 10 values):")
first_row = train_df.iloc[0]
print(f"   Disease: {first_row[disease_cols[0]] if disease_cols else 'N/A'}")
print(f"   Symptoms (0/1): {first_row.drop(disease_cols).values[:10] if disease_cols else first_row.values[:10]}")

print("\n" + "=" * 50)
print("📊 SUMMARY:")
print(f"   • Training samples: {train_df.shape[0]}")
print(f"   • Symptoms/Features: {train_df.shape[1] - 1}")
print(f"   • Likely disease column: {disease_cols[0] if disease_cols else 'Not found'}")
print("=" * 50)