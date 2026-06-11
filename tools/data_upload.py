"""Data Upload & Validation Script

Script untuk:
1. Validate format data
2. Check data quality
3. Merge dengan existing data
4. Generate statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data_format(filepath):
    """
    Validate CSV format dan struktur data
    
    Parameters:
    -----------
    filepath : str
        Path ke CSV file
    
    Returns:
    --------
    bool : True jika valid, False jika invalid
    """
    
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    try:
        # Load data
        print(f"\n[1] Loading file: {filepath}")
        df = pd.read_csv(filepath)
        print(f"    ✓ File loaded successfully")
        print(f"    Records: {len(df):,}")
        print(f"    Columns: {list(df.columns)}")
        
        # Check required columns
        print(f"\n[2] Checking required columns...")
        required_cols = ['date', 'blood_type', 'demand', 'collection']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"    ✗ Missing columns: {missing_cols}")
            return False
        print(f"    ✓ All required columns present")
        
        # Parse dates
        print(f"\n[3] Validating date format...")
        try:
            df['date'] = pd.to_datetime(df['date'])
            print(f"    ✓ Date format valid (YYYY-MM-DD)")
            print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
            days = (df['date'].max() - df['date'].min()).days
            print(f"    Duration: {days} days (~{days/365:.1f} years)")
            
            if days < 730:
                print(f"    ⚠ Warning: Less than 2 years of data (minimum recommended)")
        except Exception as e:
            print(f"    ✗ Date parsing error: {e}")
            return False
        
        # Validate blood types
        print(f"\n[4] Validating blood types...")
        valid_types = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
        actual_types = df['blood_type'].unique()
        invalid_types = [bt for bt in actual_types if bt not in valid_types]
        
        if invalid_types:
            print(f"    ✗ Invalid blood types: {invalid_types}")
            return False
        
        print(f"    ✓ Blood types valid")
        print(f"    Found: {', '.join(sorted(actual_types))}")
        
        # Check for all blood types
        missing_types = [bt for bt in valid_types if bt not in actual_types]
        if missing_types:
            print(f"    ⚠ Missing blood types: {missing_types}")
        
        # Validate demand and collection
        print(f"\n[5] Validating demand & collection values...")
        
        if (df['demand'] < 0).any():
            print(f"    ✗ Demand contains negative values")
            return False
        
        if (df['collection'] < 0).any():
            print(f"    ✗ Collection contains negative values")
            return False
        
        if not pd.api.types.is_numeric_dtype(df['demand']):
            print(f"    ✗ Demand column is not numeric")
            return False
        
        if not pd.api.types.is_numeric_dtype(df['collection']):
            print(f"    ✗ Collection column is not numeric")
            return False
        
        print(f"    ✓ Demand & collection values valid")
        print(f"    Demand range: {df['demand'].min():.0f} - {df['demand'].max():.0f}")
        print(f"    Collection range: {df['collection'].min():.0f} - {df['collection'].max():.0f}")
        
        # Check for missing values
        print(f"\n[6] Checking for missing values...")
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"    ✗ Contains null values:")
            print(null_counts[null_counts > 0])
            return False
        print(f"    ✓ No null values")
        
        # Check for duplicates
        print(f"\n[7] Checking for duplicates...")
        dupes = df.duplicated(subset=['date', 'blood_type'])
        if dupes.any():
            print(f"    ✗ Found {dupes.sum()} duplicate date-blood_type combinations")
            return False
        print(f"    ✓ No duplicates")
        
        # Summary statistics
        print(f"\n[8] Summary Statistics")
        print(f"\n    DEMAND:")
        print(f"      Mean: {df['demand'].mean():.2f}")
        print(f"      Std:  {df['demand'].std():.2f}")
        print(f"      Min:  {df['demand'].min():.0f}")
        print(f"      Max:  {df['demand'].max():.0f}")
        
        print(f"\n    COLLECTION:")
        print(f"      Mean: {df['collection'].mean():.2f}")
        print(f"      Std:  {df['collection'].std():.2f}")
        print(f"      Min:  {df['collection'].min():.0f}")
        print(f"      Max:  {df['collection'].max():.0f}")
        
        collection_rate = (df['collection'].sum() / df['demand'].sum() * 100)
        print(f"\n    Collection Rate: {collection_rate:.2f}%")
        
        print(f"\n" + "="*80)
        print("✓ VALIDATION PASSED")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        return False


# ============================================================================
# DATA MERGING
# ============================================================================

def merge_data(new_data_path, existing_data_path=None, output_path='data/processed/blood_demand_data.csv'):
    """
    Merge new data dengan existing data (jika ada)
    
    Parameters:
    -----------
    new_data_path : str
        Path ke new data
    existing_data_path : str
        Path ke existing data (optional)
    output_path : str
        Path untuk output merged data
    """
    
    print(f"\n" + "="*80)
    print("DATA MERGING")
    print("="*80)
    
    # Load new data
    print(f"\n[1] Loading new data...")
    df_new = pd.read_csv(new_data_path)
    df_new['date'] = pd.to_datetime(df_new['date'])
    print(f"    ✓ Loaded {len(df_new):,} records")
    
    # Load existing data if provided
    if existing_data_path and Path(existing_data_path).exists():
        print(f"\n[2] Loading existing data...")
        df_existing = pd.read_csv(existing_data_path)
        df_existing['date'] = pd.to_datetime(df_existing['date'])
        print(f"    ✓ Loaded {len(df_existing):,} records")
        
        # Merge
        print(f"\n[3] Merging data...")
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Remove duplicates (keep first)
        df_merged = df_merged.drop_duplicates(subset=['date', 'blood_type'], keep='first')
        print(f"    ✓ Merged: {len(df_merged):,} records (duplicates removed)")
    else:
        df_merged = df_new
        print(f"\n[2] No existing data found, using new data only")
    
    # Sort by date and blood type
    df_merged = df_merged.sort_values(['date', 'blood_type']).reset_index(drop=True)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    
    print(f"\n[4] Saving merged data...")
    print(f"    ✓ Saved to: {output_path}")
    print(f"    Records: {len(df_merged):,}")
    print(f"    Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
    
    return df_merged


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# DATA UPLOAD & VALIDATION TOOL")
    print("#"*80)
    
    # Check if data file provided as argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Use sample data
        data_file = 'data/raw/sample_blood_demand_data.csv'
    
    print(f"\nData file: {data_file}")
    
    # Validate
    if validate_data_format(data_file):
        # Merge with existing data
        df = merge_data(
            new_data_path=data_file,
            existing_data_path='data/processed/blood_demand_data.csv',
            output_path='data/processed/blood_demand_data.csv'
        )
        
        print(f"\n" + "#"*80)
        print("# DATA UPLOAD COMPLETED ✓")
        print("#"*80)
        print(f"\nNext step: Run 'python src/01_data_preparation.py'")
        print("#"*80 + "\n")
    else:
        print(f"\nPlease fix data format issues before uploading.")
        print(f"See DATA_GUIDE.md for format requirements.")
        sys.exit(1)
