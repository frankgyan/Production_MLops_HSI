# fix_labels_clear.py
"""
Extract correct labels from filename nutrient codes.
HNHP=1, HNLP=2, LNHP=3, LNLP=4

The numbers like 10, 20, 30 are SAMPLE NUMBERS, NOT labels!
"""

import pandas as pd
import re
from pathlib import Path

# Your Excel path
excel_path = Path(r"C:/Frank/New_folder/Quinoa_dataset-hyper/stages/quinoa_hsi_labels.xlsx")

# Load
df = pd.read_excel(excel_path, engine='openpyxl')

print("="*70)
print("FIXING LABELS FROM FILENAME NUTRIENT CODES")
print("="*70)
print(f"\n📊 Loaded {len(df)} samples from Excel")

# Define nutrient code to label mapping (1-4 ONLY)
nutrient_to_label = {
    'HNHP': 1,  # High N, High P
    'HNLP': 2,  # High N, Low P  
    'LNHP': 3,  # Low N, High P
    'LNLP': 4   # Low N, Low P
}

# Reverse mapping for display
label_to_nutrient = {v: k for k, v in nutrient_to_label.items()}

def extract_nutrient_code(filename):
    """
    Extract nutrient code from filename.
    Only looks for: HNHP, HNLP, LNHP, LNLP
    Ignores numbers like 10, 20, 30, 50 (these are sample numbers)
    """
    filename = str(filename)
    
    # Check each nutrient code
    for code in nutrient_to_label.keys():
        if code in filename:
            return code
    
    return None

# Apply extraction
df['nutrient_code'] = df['File_Name'].apply(extract_nutrient_code)
df['correct_label'] = df['nutrient_code'].map(nutrient_to_label)

# Show what we found
print(f"\n📋 Nutrient codes found in filenames:")
nutrient_counts = df['nutrient_code'].value_counts()
for code, count in nutrient_counts.items():
    label = nutrient_to_label[code]
    print(f"   {code} → Label {label}: {count} samples")

# Show sample numbers (10, 20, etc.) are NOT labels
print(f"\n📋 Sample numbers found (these are NOT labels):")
# Extract sample numbers from filename
def extract_sample_number(filename):
    match = re.search(r'__(\d+)$', str(filename))
    if match:
        return int(match.group(1))
    return None

df['sample_number'] = df['File_Name'].apply(extract_sample_number)
print(f"   Sample numbers range from {df['sample_number'].min()} to {df['sample_number'].max()}")
print(f"   These are sample IDs, NOT class labels!")

# Check for files with no nutrient code
missing = df[df['nutrient_code'].isna()]
if len(missing) > 0:
    print(f"\n⚠️ {len(missing)} files have NO nutrient code in filename:")
    print("   First 5 problematic files:")
    for f in missing['File_Name'].head(5):
        print(f"     {f}")

# Compare with original labels
df['original_label'] = df['Label']
mismatches = df[df['Label'] != df['correct_label']]

print(f"\n📊 COMPARISON:")
print(f"   Original Excel labels distribution:")
orig_counts = df['original_label'].value_counts().sort_index()
for label, count in orig_counts.items():
    nutrient = label_to_nutrient.get(label, "UNKNOWN")
    print(f"     Label {label} ({nutrient}): {count} samples")

print(f"\n   Corrected labels from filename:")
corr_counts = df['correct_label'].value_counts().sort_index()
for label, count in corr_counts.items():
    nutrient = label_to_nutrient[label]
    print(f"     Label {label} ({nutrient}): {count} samples")

if len(mismatches) > 0:
    print(f"\n⚠️ Found {len(mismatches)} MISMATCHES between Excel labels and filename nutrient codes!")
    print("\n📋 First 10 mismatches (showing Excel Label → Correct Label):")
    for idx, row in mismatches.head(10).iterrows():
        sample_num = extract_sample_number(row['File_Name'])
        print(f"  {row['File_Name'][:60]}... → Excel: {row['Label']} (wrong), Filename: {row['nutrient_code']} (correct)")
else:
    print("\n✅ NO mismatches found! Your Excel labels are already correct.")

# Create corrected file
df_corrected = df.copy()
df_corrected['Label'] = df_corrected['correct_label']

# Remove rows with no label (no nutrient code found)
df_corrected = df_corrected.dropna(subset=['Label'])
df_corrected['Label'] = df_corrected['Label'].astype(int)

# Save corrected file
output_path = excel_path.parent / "quinoa_hsi_labels_CORRECTED.xlsx"
df_corrected[['File_Name', 'Label']].to_excel(output_path, index=False)

print("\n" + "="*70)
print("✅ CORRECTED LABEL FILE SAVED!")
print("="*70)
print(f"   File: {output_path}")
print(f"   Total samples: {len(df_corrected)}")
print(f"\n   Final Class Distribution (Labels 1-4 ONLY):")
for label in sorted(df_corrected['Label'].unique()):
    count = len(df_corrected[df_corrected['Label'] == label])
    nutrient = label_to_nutrient[label]
    print(f"     {label} ({nutrient}): {count} samples")

print("\n" + "="*70)
print("📝 NEXT STEPS:")
print("="*70)
print('1. Update config/production.yaml with:')
print('   label_filename: "quinoa_hsi_labels_CORRECTED.xlsx"')
print("\n2. Delete old artifacts to start fresh:")
print("   rmdir /s artifacts\\dataset")
print("   rmdir /s artifacts\\preprocessed")
print("   rmdir /s artifacts\\model")
print("\n3. Re-run pipeline:")
print("   python stages/stage1_ingest.py --config config/production.yaml")
print("   python stages/stage2_preprocess.py --config config/production.yaml")
print("   python stages/stage3_train.py --config config/production.yaml --model spectral_spatial")