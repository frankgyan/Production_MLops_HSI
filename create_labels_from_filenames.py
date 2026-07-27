# create_labels_from_filenames.py
"""
Create Excel label file from filenames in directory structure.
Extracts nutrient codes (HNHP, HNLP, LNHP, LNLP) from filenames
and assigns corresponding labels (1, 2, 3, 4).

Filename format: AVN_1_RUN3A_2020-10-08_21_LNLP__27
                                    └────┬────┘
                                    Nutrient Code

Nutrient Code → Label:
    HNHP → 1 (High N, High P)
    HNLP → 2 (High N, Low P)
    LNHP → 3 (Low N, High P)
    LNLP → 4 (Low N, Low P)
"""

import os
import re
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

# ============================================================
# NUTRIENT CODE MAPPING
# ============================================================

NUTRIENT_TO_LABEL = {
    'HNHP': 1,  # High N, High P (Healthy)
    'HNLP': 2,  # High N, Low P
    'LNHP': 3,  # Low N, High P
    'LNLP': 4   # Low N, Low P (Severe stress)
}

LABEL_TO_NUTRIENT = {
    1: 'HNHP',
    2: 'HNLP',
    3: 'LNHP',
    4: 'LNLP'
}

# ============================================================
# EXTRACT FUNCTIONS
# ============================================================

def extract_nutrient_code(filename):
    """
    Extract nutrient code from filename.
    
    Args:
        filename: e.g., "AVN_1_RUN3A_2020-10-08_21_LNLP__27"
    
    Returns:
        Nutrient code: "HNHP", "HNLP", "LNHP", or "LNLP"
        None if not found
    """
    # Search for known nutrient codes in filename
    for code in NUTRIENT_TO_LABEL.keys():
        if code in filename:
            return code
    return None

def extract_run_number(filename):
    """Extract run number from filename."""
    match = re.search(r'RUN(\d+[A-Z])', filename)
    return match.group(1) if match else None

def extract_date(filename):
    """Extract date from filename."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None

def extract_sample_number(filename):
    """Extract sample number from filename."""
    match = re.search(r'__(\d+)$', filename)
    return int(match.group(1)) if match else None

def parse_filename(filename):
    """
    Parse filename into components.
    
    Args:
        filename: e.g., "AVN_1_RUN3A_2020-10-08_21_LNLP__27"
    
    Returns:
        Dictionary with parsed components
    """
    return {
        'nutrient_code': extract_nutrient_code(filename),
        'run_number': extract_run_number(filename),
        'date': extract_date(filename),
        'sample_number': extract_sample_number(filename),
        'label': NUTRIENT_TO_LABEL.get(extract_nutrient_code(filename))
    }

# ============================================================
# SCAN DIRECTORY
# ============================================================

def scan_directory(data_dir, extensions=None):
    """
    Recursively scan directory for HSI files.
    
    Args:
        data_dir: Path to data directory
        extensions: List of file extensions to include
    
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.tiff', '.tif', '.npy', '.npz']
    
    data_dir = Path(data_dir)
    files = []
    
    print(f"Scanning directory: {data_dir}")
    print(f"Looking for extensions: {extensions}")
    
    for ext in extensions:
        # Recursively find files with given extension
        for file_path in data_dir.rglob(f'*{ext}'):
            # Get filename without extension
            filename = file_path.stem
            files.append({
                'full_path': str(file_path),
                'filename': filename,
                'extension': file_path.suffix
            })
    
    print(f"Found {len(files)} files")
    return files

# ============================================================
# CREATE LABEL FILE
# ============================================================

def create_label_file(data_dir, output_path=None, extensions=None):
    """
    Create Excel label file from files in directory.
    
    Args:
        data_dir: Path to data directory
        output_path: Path to save Excel file (optional)
        extensions: List of file extensions to include
    
    Returns:
        DataFrame with labels
    """
    # Scan directory
    files = scan_directory(data_dir, extensions)
    
    if not files:
        print("No files found!")
        return None
    
    # Parse each file
    records = []
    nutrient_counts = {code: 0 for code in NUTRIENT_TO_LABEL.keys()}
    skipped = 0
    skipped_files = []
    
    print("\nProcessing files...")
    
    for file_info in files:
        filename = file_info['filename']
        parsed = parse_filename(filename)
        
        if parsed['nutrient_code'] is None:
            skipped += 1
            skipped_files.append(filename)
            continue
        
        # Count nutrient codes
        nutrient_counts[parsed['nutrient_code']] += 1
        
        records.append({
            'File_Name': filename,
            'Label': parsed['label'],
            'Nutrient_Code': parsed['nutrient_code'],
            'Run_Number': parsed['run_number'],
            'Date': parsed['date'],
            'Sample_Number': parsed['sample_number']
        })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by date, run, sample number for consistency
    if 'Date' in df.columns and 'Sample_Number' in df.columns:
        df = df.sort_values(['Date', 'Run_Number', 'Sample_Number'])
    
    # Save to Excel (only File_Name and Label columns)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(data_dir) / f'quinoa_labels_{timestamp}.xlsx'
    else:
        output_path = Path(output_path)
    
    # Save full dataframe to Excel
    output_full = output_path.parent / f"{output_path.stem}_full{output_path.suffix}"
    df.to_excel(output_full, index=False)
    print(f"\nFull data saved to: {output_full}")
    
    # Save simplified version (only File_Name and Label)
    df_simple = df[['File_Name', 'Label']]
    df_simple.to_excel(output_path, index=False)
    print(f"Simplified labels saved to: {output_path}")
    
    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    
    print("\n" + "="*60)
    print("LABEL CREATION SUMMARY")
    print("="*60)
    
    print(f"\n Total files found: {len(files)}")
    print(f"   Valid files: {len(records)}")
    print(f"   Skipped files: {skipped}")
    
    if skipped_files:
        print(f"\n First 10 skipped files:")
        for f in skipped_files[:10]:
            print(f"   - {f}")
    
    print(f"\n Nutrient code distribution:")
    for code in NUTRIENT_TO_LABEL.keys():
        label = NUTRIENT_TO_LABEL[code]
        count = nutrient_counts[code]
        pct = (count / len(records) * 100) if len(records) > 0 else 0
        print(f"   {code} (Label {label}): {count} ({pct:.1f}%)")
    
    print(f"\n Total samples: {len(records)}")
    print(f"   Class distribution:")
    for label in sorted(df['Label'].unique()):
        count = len(df[df['Label'] == label])
        nutrient = LABEL_TO_NUTRIENT[label]
        pct = (count / len(records) * 100) if len(records) > 0 else 0
        print(f"   {label} ({nutrient}): {count} ({pct:.1f}%)")
    
    # Check class balance
    if len(records) > 0:
        max_count = df['Label'].value_counts().max()
        min_count = df['Label'].value_counts().min()
        balance_ratio = max_count / min_count if min_count > 0 else 0
        print(f"\n Class balance ratio: {balance_ratio:.2f}x")
        if balance_ratio < 1.1:
            print("Classes are very well balanced!")
        elif balance_ratio < 1.3:
            print("Classes are reasonably balanced.")
        else:
            print("Classes are imbalanced. Consider weighting or augmentation.")
    
    print("\n" + "="*60)
    
    return df

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create Excel label file from HSI filenames'
    )
    parser.add_argument('--data_dir', 
                        default='C:/Frank/New_folder/HSI_AVNData',
                        help='Path to data directory')
    parser.add_argument('--output', 
                        help='Path to output Excel file')
    parser.add_argument('--extensions', nargs='+',
                        default=['.tiff', '.tif', '.npy', '.npz'],
                        help='File extensions to include')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HSI LABEL CREATOR")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    
    if not Path(args.data_dir).exists():
        print(f"Error: Directory not found: {args.data_dir}")
        return
    
    create_label_file(
        data_dir=args.data_dir,
        output_path=args.output,
        extensions=args.extensions
    )

if __name__ == "__main__":
    main()