# Data Upload Guide untuk Indonesia PMI Format

## Format Data yang Didukung

### Format 1: Daily Format (Recommended)

```csv
date,blood_type,demand,collection
2024-01-01,O+,145,120
2024-01-01,O-,78,65
2024-01-01,A+,118,95
...
```

**Requirements:**
- Semicolon or comma separated
- date: YYYY-MM-DD
- blood_type: O+, O-, A+, A-, B+, B-, AB+, AB-
- demand: Integer (jumlah kantong diminta)
- collection: Integer (jumlah kantong dikumpulkan)

### Format 2: Monthly Format (Indonesia PMI)

**Supported structure:**

```
Bulan;Total Permintaan;PRC O;PRC A;PRC B;PRC AB;TC O;TC A;TC B;TC AB;FFP O;FFP A;FFP B;FFP AB;AHF O;AHF A;AHF B;AHF AB
Januari;7336;1471;1022;519;473;690;568;452;302;288;227;234;201;221;219;226;223
Februari;5986;1371;902;637;384;567;476;452;180;175;121;123;107;150;118;109;114
...
```

**Features:**
- ✅ Semicolon-separated (Indonesian Excel format)
- ✅ Monthly aggregated data
- ✅ Multiple product types: PRC, TC, FFP, AHF
- ✅ Blood type breakdown: O, A, B, AB
- ✅ Automatically converted to daily data
- ✅ Random daily variation applied (~±10%)
- ✅ Collection rate estimated (~80-90%)

---

## Upload & Convert Your Data

### Step 1: Upload File ke Repository

**Option A: Via GitHub Web**
```
1. Go: https://github.com/Dindina66/blood-stock-management
2. Branch: dev/arima-rl-forecasting
3. Folder: data/raw/
4. Click: "Add file" → "Upload files"
5. Drag: DATA DARAH.csv (or your file)
6. Commit
```

**Option B: Via Git CLI**
```bash
git clone https://github.com/Dindina66/blood-stock-management.git
cd blood-stock-management
git checkout dev/arima-rl-forecasting

# Copy your file
cp "DATA DARAH.csv" "data/raw/"

# Commit
git add "data/raw/DATA DARAH.csv"
git commit -m "Add blood demand data from PMI Jakarta Barat"
git push origin dev/arima-rl-forecasting
```

### Step 2: Convert Monthly to Daily Format

**Command:**
```bash
# Convert dengan year 2024 (default)
python tools/indonesian_data_converter.py "data/raw/DATA DARAH.csv" 2024

# Convert dengan year tertentu
python tools/indonesian_data_converter.py "data/raw/DATA DARAH.csv" 2023
```

**Output:**
```
✓ Converted monthly to daily format
✓ 365 days × 8 blood types = ~2,920 records
✓ Saved to: data/processed/blood_demand_data_converted.csv
```

### Step 3: Validate Converted Data

```bash
python tools/data_upload.py data/processed/blood_demand_data_converted.csv
```

**Validation will check:**
- ✓ Column structure
- ✓ Date format & range
- ✓ Blood type validity
- ✓ No null values
- ✓ No duplicates
- ✓ Summary statistics

---

## Full Pipeline Example

```bash
# 1. Letakkan file DATA DARAH.csv di data/raw/
ls data/raw/DATA\ DARAH.csv

# 2. Convert monthly → daily
python tools/indonesian_data_converter.py "data/raw/DATA DARAH.csv" 2024

# 3. Validate
python tools/data_upload.py data/processed/blood_demand_data_converted.csv

# 4. Jika valid, jalankan pipeline
python src/01_data_preparation.py
python src/02_arima_forecasting.py
python src/03_rl_environment.py
python src/04_rl_agent.py
python src/05_hybrid_system.py
```

---

## Data Transformation Details

### Conversion Process:

1. **Parse Monthly Data**
   - Extract: Bulan, Total Permintaan, PRC/TC/FFP/AHF per blood type
   - Aggregate all product types per blood type

2. **Expand to Daily**
   - Divide monthly demand by days in month
   - Apply ±10% random daily variation
   - Estimate collection rate 80-90%

3. **Split Blood Type**
   - O → [O+, O-]
   - A → [A+, A-]
   - B → [B+, B-]
   - AB → [AB+, AB-]
   - Random split: 40-60% untuk setiap tipe

4. **Generate Daily Records**
   - Final format: (date, blood_type, demand, collection)
   - Total records: days × 8 blood types

---

## Column Mapping (Indonesian PMI Format)

```
Column Index | Column Name | Blood Type Group
0            | Bulan       | -
1            | Total Permintaan | -
2-5          | PRC O/A/B/AB | O, A, B, AB
6-9          | TC O/A/B/AB  | O, A, B, AB
10-13        | FFP O/A/B/AB | O, A, B, AB
14-17        | AHF O/A/B/AB | O, A, B, AB
```

### Aggregation Formula:
```
Blood Type O demand = PRC_O + TC_O + FFP_O + AHF_O
Blood Type A demand = PRC_A + TC_A + FFP_A + AHF_A
Blood Type B demand = PRC_B + TC_B + FFP_B + AHF_B
Blood Type AB demand = PRC_AB + TC_AB + FFP_AB + AHF_AB
```

---

## Troubleshooting

### Error: "File not found"
```bash
# Pastikan path benar
ls data/raw/

# Coba dengan quotes jika ada spaces
python tools/indonesian_data_converter.py "data/raw/DATA DARAH.csv"
```

### Error: "Could not parse"
- Check format adalah semicolon-separated (bukan comma)
- Verify kolom PRC, TC, FFP, AHF ada
- Pastikan format angka: 7336 atau 7.336 (tidak 7,336)

### Error: "Invalid blood type"
- Valid types hanya: O+, O-, A+, A-, B+, B-, AB+, AB-
- Script auto-split dari O → [O+, O-], dst

### Small dataset warning
- Data hanya 1 tahun? Converter akan "generate" daily variation
- Recommended: 2-3 tahun historical data untuk ARIMA accuracy
- Minimal: 730 hari

---

## Expected Output

```
================================================================================
INDONESIAN BLOOD DATA CONVERSION
================================================================================

[1] Loading Indonesian format data...
    ✓ File loaded
    Rows: 12
    Columns: ['Bulan', 'Total Permintaan', ...]

[2] Extracting monthly data...
    ✓ Extracted 12 months

[3] Expanding to daily data...
    ✓ Expanded to 2,920 daily records
    Date range: 2024-01-01 to 2024-12-31

[4] Saving converted data...
    ✓ Saved to: data/processed/blood_demand_data_converted.csv
    Records: 2,920

[5] Validation Summary
    Date range: 2024-01-01 to 2024-12-31
    Blood types: O+, O-, A+, A-, B+, B-, AB+, AB-
    Avg daily demand: 8.50
    Collection rate: 85.30%

================================================================================
✓ CONVERSION COMPLETED
================================================================================
```

---

## Support

Jika ada masalah:
1. ✅ Check file encoding (UTF-8 atau Latin-1)
2. ✅ Verify semicolon separator
3. ✅ Ensure numeric columns tidak ada space/karakter aneh
4. ✅ Run dengan `--verbose` flag (jika ada)
5. ✅ Create GitHub issue dengan sample data

---

**Last Updated**: 2026-06-11
**Supported**: Indonesia PMI Format (Jakarta Barat), Daily Format
