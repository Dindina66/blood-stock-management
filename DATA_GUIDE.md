# Blood Stock Management - Data Guide

## Data Structure

Data yang digunakan harus memiliki struktur berikut:

### Format CSV

```csv
date,blood_type,demand,collection
2020-01-01,O+,145,120
2020-01-01,O-,78,65
2020-01-01,A+,118,95
2020-01-01,A-,62,50
2020-01-01,B+,98,85
2020-01-01,B-,48,40
2020-01-01,AB+,38,32
2020-01-01,AB-,28,22
2020-01-02,O+,152,125
...
```

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| **date** | YYYY-MM-DD | Tanggal | 2020-01-01 |
| **blood_type** | String | Golongan darah | O+, O-, A+, A-, B+, B-, AB+, AB- |
| **demand** | Integer | Jumlah kantong darah diminta | 145 |
| **collection** | Integer | Jumlah kantong darah dikumpulkan | 120 |

### Data Requirements

- **Time Range**: Minimum 2 tahun (730 hari) untuk good ARIMA results
- **Frequency**: Daily data
- **Blood Types**: Semua 8 golongan darah harus ada
- **Coverage**: Tidak ada missing dates dalam range
- **Values**: Harus non-negative integers

## Data Sources

### Official Sources:
- Platform SatuSehat (https://satusehat.kemkes.go.id/)
- PMI Pusat Database
- Unit Transfusi Darah Rumah Sakit
- Dinas Kesehatan DKI Jakarta

### Data Collection Methods:

1. **Direct Export from SatuSehat**
   - Login ke platform
   - Navigate ke Blood Bank module
   - Export daily report as CSV

2. **Manual Entry from Records**
   - Collect dari laporan harian Unit Transfusi
   - Input ke format CSV
   - Validate sebelum upload

3. **Database Query**
   ```sql
   SELECT 
     DATE(created_at) as date,
     blood_type,
     COUNT(*) as demand,
     SUM(collected) as collection
   FROM blood_transactions
   GROUP BY DATE(created_at), blood_type
   ORDER BY date, blood_type;
   ```

## How to Upload Data

### Option 1: Direct Upload via GitHub

```bash
# 1. Clone repository
git clone https://github.com/Dindina66/blood-stock-management.git
cd blood-stock-management

# 2. Checkout to branch
git checkout dev/arima-rl-forecasting

# 3. Copy your CSV file ke folder data/raw/
cp your_data.csv data/raw/blood_demand_data.csv

# 4. Commit dan push
git add data/raw/blood_demand_data.csv
git commit -m "Add real blood demand data from SatuSehat"
git push origin dev/arima-rl-forecasting
```

### Option 2: Upload via GitHub Web Interface

1. Navigate ke https://github.com/Dindina66/blood-stock-management
2. Switch ke branch `dev/arima-rl-forecasting`
3. Open folder `data/raw/`
4. Click "Add file" → "Upload files"
5. Drag & drop CSV file
6. Commit dengan message: "Add blood demand data"

### Option 3: Modify data_preparation.py

Edit `src/01_data_preparation.py` untuk load dari custom source:

```python
def load_real_data(filepath):
    """
    Load real demand data dari CSV
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Validate columns
    required_cols = ['date', 'blood_type', 'demand', 'collection']
    assert all(col in df.columns for col in required_cols), \
        f"Missing columns. Required: {required_cols}"
    
    # Validate blood types
    valid_types = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
    assert all(bt in valid_types for bt in df['blood_type'].unique()), \
        f"Invalid blood type. Valid: {valid_types}"
    
    # Validate date range
    date_diff = (df['date'].max() - df['date'].min()).days
    assert date_diff >= 730, f"Need at least 2 years of data. Got {date_diff} days"
    
    return df

# Usage
if __name__ == "__main__":
    df = load_real_data('data/raw/blood_demand_data.csv')
    # ... continue with pipeline
```

## Data Validation Checklist

Sebelum upload, pastikan:

- [ ] File format: CSV
- [ ] Columns: date, blood_type, demand, collection
- [ ] Date format: YYYY-MM-DD
- [ ] Date range: Minimum 2 tahun
- [ ] No missing values
- [ ] All 8 blood types present
- [ ] Demand & collection values: non-negative integers
- [ ] No duplicate date + blood_type combinations
- [ ] File size: < 10MB

## Example Data Validation Script

```python
import pandas as pd

def validate_data(filepath):
    df = pd.read_csv(filepath)
    
    # Check columns
    required = ['date', 'blood_type', 'demand', 'collection']
    assert all(c in df.columns for c in required), "Missing columns"
    
    # Check date format
    df['date'] = pd.to_datetime(df['date'])
    
    # Check blood types
    valid_types = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
    invalid = df[~df['blood_type'].isin(valid_types)]
    assert len(invalid) == 0, f"Invalid blood types: {invalid['blood_type'].unique()}"
    
    # Check date range
    days = (df['date'].max() - df['date'].min()).days
    assert days >= 730, f"Need >= 2 years, got {days} days"
    
    # Check missing values
    assert df.isnull().sum().sum() == 0, "Contains null values"
    
    # Check for duplicates
    dupes = df.duplicated(subset=['date', 'blood_type'])
    assert not dupes.any(), "Contains duplicate date-blood_type combinations"
    
    # Check values
    assert (df['demand'] >= 0).all(), "Demand contains negative values"
    assert (df['collection'] >= 0).all(), "Collection contains negative values"
    
    print("✓ Data validation passed!")
    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Blood types: {', '.join(df['blood_type'].unique())}")
    
    return df

# Run validation
if __name__ == "__main__":
    df = validate_data('data/raw/your_data.csv')
    df.to_csv('data/processed/blood_demand_data.csv', index=False)
```

## Sample Data Size Guidance

| Data Period | Records | Use Case |
|------------|---------|----------|
| 6 months | ~1,440 | Testing only |
| 1 year | ~2,880 | Quick prototyping |
| 2 years | ~5,760 | **Recommended minimum** |
| 3-5 years | ~8,640-14,400 | **Optimal for production** |
| 5+ years | >14,400 | Comprehensive analysis |

## Troubleshooting

### Error: "Missing columns"
- Ensure CSV has: date, blood_type, demand, collection
- Check column names (case-sensitive)

### Error: "Invalid blood type"
- Valid types: O+, O-, A+, A-, B+, B-, AB+, AB-
- Check for typos or extra spaces

### Error: "Date parsing error"
- Use format: YYYY-MM-DD (e.g., 2020-01-15)
- Not: DD/MM/YYYY or MM/DD/YYYY

### Error: "Need at least 2 years of data"
- Current data only covers ~6-12 months
- Collect more historical data or merge multiple sources

## Support

Jika ada pertanyaan mengenai data:
1. Check format sesuai template di atas
2. Run validation script
3. Create issue di GitHub dengan sample data (anonymized)

---

**Last Updated**: 2026-06-11
