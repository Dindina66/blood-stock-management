"""Data Preparation untuk Blood Demand Forecasting

Tahap ini meliputi:
1. Load historis data dari SatuSehat/CSV
2. Exploratory Data Analysis (EDA)
3. Data cleaning & normalisasi
4. Time series decomposition
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 1: Generate Sample Historical Data (untuk demo)
# ============================================================================

def generate_sample_demand_data(start_date='2020-01-01', 
                               end_date='2025-12-31',
                               blood_types=['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']):
    """
    Generate synthetic demand data yang mengikuti pola musiman dan trend
    
    Parameters:
    -----------
    start_date : str
        Tanggal awal (format YYYY-MM-DD)
    end_date : str
        Tanggal akhir (format YYYY-MM-DD)
    blood_types : list
        Daftar golongan darah
    
    Returns:
    --------
    pd.DataFrame : Data demand dengan kolom [date, blood_type, demand, collection]
    """
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    
    data = []
    
    for blood_type in blood_types:
        # Base demand per golongan darah (O+ paling tinggi)
        base_demand = {'O+': 150, 'O-': 80, 'A+': 120, 'A-': 60,
                       'B+': 100, 'B-': 50, 'AB+': 40, 'AB-': 30}[blood_type]
        
        for date in date_range:
            # Trend: sedikit peningkatan seiring waktu
            trend = 0.001 * (date - date_range[0]).days
            
            # Seasonality: pola mingguan (lebih tinggi di hari kerja)
            day_of_week = date.dayofweek
            seasonality = 1.2 if day_of_week < 5 else 0.8  # Lebih tinggi hari kerja
            
            # Demand dengan noise
            demand = base_demand * (1 + trend) * seasonality + np.random.normal(0, 10)
            demand = max(0, int(demand))  # Non-negative
            
            # Collection (sedikit lebih rendah dari demand di awal, naik seiring waktu)
            collection_factor = 0.8 + 0.002 * (date - date_range[0]).days
            collection = int(base_demand * collection_factor * seasonality + np.random.normal(0, 5))
            collection = max(0, collection)
            
            data.append({
                'date': date,
                'blood_type': blood_type,
                'demand': demand,
                'collection': collection
            })
    
    df = pd.DataFrame(data)
    return df.sort_values(['blood_type', 'date']).reset_index(drop=True)


# ============================================================================
# STEP 2: Exploratory Data Analysis
# ============================================================================

def perform_eda(df, output_dir='results/eda'):
    """
    Perform Exploratory Data Analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data dengan kolom [date, blood_type, demand, collection]
    output_dir : str
        Direktori untuk simpan hasil visualisasi
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - BLOOD DEMAND FORECASTING")
    print("="*80)
    
    # Basic statistics
    print("\n[1] BASIC STATISTICS")
    print(f"Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total records: {len(df):,}")
    print(f"Blood types: {', '.join(df['blood_type'].unique())}")
    
    print("\n[2] DEMAND STATISTICS BY BLOOD TYPE")
    demand_stats = df.groupby('blood_type')['demand'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('q75', lambda x: x.quantile(0.75))
    ]).round(2)
    print(demand_stats)
    
    # Time series visualization
    print("\n[3] VISUALIZING TIME SERIES DATA")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Blood Demand Time Series Analysis', fontsize=16, fontweight='bold')
    
    # All blood types
    ax = axes[0, 0]
    for blood_type in df['blood_type'].unique():
        subset = df[df['blood_type'] == blood_type]
        ax.plot(subset['date'], subset['demand'], label=blood_type, alpha=0.7)
    ax.set_title('Demand Trend - All Blood Types')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand (kantong)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # O+ blood type (most common)
    ax = axes[0, 1]
    o_plus = df[df['blood_type'] == 'O+']
    ax.plot(o_plus['date'], o_plus['demand'], color='red', label='Demand', alpha=0.7)
    ax.plot(o_plus['date'], o_plus['collection'], color='blue', label='Collection', alpha=0.7)
    ax.set_title('O+ Blood Type - Demand vs Collection')
    ax.set_xlabel('Date')
    ax.set_ylabel('Kantong')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1, 0]
    df.boxplot(column='demand', by='blood_type', ax=ax)
    ax.set_title('Demand Distribution by Blood Type')
    ax.set_xlabel('Blood Type')
    ax.set_ylabel('Demand')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    # Day of week pattern
    ax = axes[1, 1]
    df['day_of_week'] = df['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day = df.groupby('day_of_week')['demand'].mean().reindex(day_order)
    ax.bar(range(7), df_day.values, color='skyblue', edgecolor='black')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_title('Average Demand by Day of Week')
    ax.set_ylabel('Average Demand')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_time_series_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/01_time_series_analysis.png")
    plt.close()
    

# ============================================================================
# STEP 3: Data Cleaning & Normalisasi
# ============================================================================

def clean_and_normalize(df):
    """
    Clean data dan normalize values
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data
    
    Returns:
    --------
    pd.DataFrame : Cleaned data
    """
    
    print("\n" + "="*80)
    print("DATA CLEANING & NORMALIZATION")
    print("="*80)
    
    df = df.copy()
    
    # Check missing values
    print("\n[1] MISSING VALUES CHECK")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found ✓")
    
    # Remove duplicates
    print("\n[2] REMOVING DUPLICATES")
    initial_len = len(df)
    df = df.drop_duplicates(subset=['date', 'blood_type'])
    removed = initial_len - len(df)
    print(f"Removed {removed} duplicate records")
    
    # Handle outliers (using IQR method)
    print("\n[3] OUTLIER DETECTION & HANDLING")
    for blood_type in df['blood_type'].unique():
        mask = df['blood_type'] == blood_type
        Q1 = df[mask]['demand'].quantile(0.25)
        Q3 = df[mask]['demand'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[mask & ((df['demand'] < lower) | (df['demand'] > upper))]
        if len(outliers) > 0:
            print(f"  {blood_type}: {len(outliers)} outliers detected")
            # Replace outliers with median
            df.loc[outliers.index, 'demand'] = df[mask]['demand'].median()
    
    # Normalize to 0-1 scale per blood type
    print("\n[4] NORMALIZATION (per blood type)")
    scaler_dict = {}
    for blood_type in df['blood_type'].unique():
        mask = df['blood_type'] == blood_type
        scaler = StandardScaler()
        df.loc[mask, 'demand_normalized'] = scaler.fit_transform(
            df[mask][['demand']]
        )
        scaler_dict[blood_type] = scaler
    
    print(f"Data normalized with StandardScaler ✓")
    print(f"Final dataset shape: {df.shape}")
    
    return df, scaler_dict


# ============================================================================
# STEP 4: Time Series Decomposition
# ============================================================================

def decompose_time_series(df, blood_type='O+', period=7, output_dir='results/decomposition'):
    """
    Decompose time series menjadi trend, seasonal, dan residual components
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned data
    blood_type : str
        Golongan darah yang akan dianalisis
    period : int
        Seasonal period (7 = weekly)
    output_dir : str
        Direktori untuk simpan hasil
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"TIME SERIES DECOMPOSITION - {blood_type}")
    print("="*80)
    
    # Subset data untuk blood type tertentu
    subset = df[df['blood_type'] == blood_type].set_index('date')['demand']
    
    # Decompose
    decomposition = seasonal_decompose(subset, model='additive', period=period)
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    fig.suptitle(f'Time Series Decomposition - {blood_type} Blood Type', 
                 fontsize=14, fontweight='bold')
    
    # Original
    axes[0].plot(decomposition.observed, color='blue')
    axes[0].set_ylabel('Original')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Observed Data')
    
    # Trend
    axes[1].plot(decomposition.trend, color='green')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Trend Component')
    
    # Seasonal
    axes[2].plot(decomposition.seasonal, color='orange')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Seasonal Component (Weekly Pattern)')
    
    # Residual
    axes[3].plot(decomposition.resid, color='red')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Residual Component')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/decomposition_{blood_type}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/decomposition_{blood_type}.png")
    plt.close()
    
    print(f"\nDecomposition Components:")
    print(f"  Trend: Linear growth/decline over time")
    print(f"  Seasonal: Weekly pattern (period={period} days)")
    print(f"  Residual: Random fluctuations & noise")


# ============================================================================
# STEP 5: Train-Test Split
# ============================================================================

def create_train_test_split(df, test_size=0.2, output_dir='data/processed'):
    """
    Split data menjadi training dan testing set
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned data
    test_size : float
        Proporsi test set (default 20%)
    output_dir : str
        Direktori untuk simpan split data
    
    Returns:
    --------
    tuple : (train_df, test_df)
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING TRAIN-TEST SPLIT")
    print("="*80)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate split point
    n_samples = len(df)
    split_point = int(n_samples * (1 - test_size))
    
    # Split per blood type to maintain temporal order
    train_dfs = []
    test_dfs = []
    
    for blood_type in df['blood_type'].unique():
        subset = df[df['blood_type'] == blood_type].reset_index(drop=True)
        split_idx = int(len(subset) * (1 - test_size))
        
        train_dfs.append(subset[:split_idx])
        test_dfs.append(subset[split_idx:])
    
    train_df = pd.concat(train_dfs, ignore_index=True).sort_values('date')
    test_df = pd.concat(test_dfs, ignore_index=True).sort_values('date')
    
    print(f"\nTotal samples: {len(df):,}")
    print(f"Training set: {len(train_df):,} ({100*(1-test_size):.0f}%)")
    print(f"Test set: {len(test_df):,} ({100*test_size:.0f}%)")
    print(f"\nTrain period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    
    # Save to CSV
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    df.to_csv(f'{output_dir}/full_data.csv', index=False)
    
    print(f"\n✓ Saved: {output_dir}/train.csv")
    print(f"✓ Saved: {output_dir}/test.csv")
    print(f"✓ Saved: {output_dir}/full_data.csv")
    
    return train_df, test_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# BLOOD DEMAND FORECASTING - DATA PREPARATION PIPELINE")
    print("#"*80)
    
    # Step 1: Generate or load data
    print("\n[STEP 1] Loading/Generating Historical Data...")
    df = generate_sample_demand_data()
    print(f"✓ Data loaded: {df.shape[0]:,} records, {df['date'].min()} to {df['date'].max()}")
    
    # Step 2: EDA
    print("\n[STEP 2] Performing Exploratory Data Analysis...")
    perform_eda(df)
    print("✓ EDA completed")
    
    # Step 3: Cleaning & Normalization
    print("\n[STEP 3] Cleaning & Normalizing Data...")
    df_clean, scaler_dict = clean_and_normalize(df)
    print("✓ Data cleaned and normalized")
    
    # Step 4: Time Series Decomposition
    print("\n[STEP 4] Decomposing Time Series...")
    for blood_type in ['O+', 'A+', 'B+', 'AB+']:
        decompose_time_series(df_clean, blood_type=blood_type)
    print("✓ Decomposition completed")
    
    # Step 5: Train-Test Split
    print("\n[STEP 5] Creating Train-Test Split...")
    train_df, test_df = create_train_test_split(df_clean, test_size=0.2)
    print("✓ Train-test split created")
    
    print("\n" + "#"*80)
    print("# DATA PREPARATION COMPLETED ✓")
    print("#"*80)
    print(f"\nNext step: Run 'python src/02_arima_forecasting.py'")
    print("#"*80 + "\n")
