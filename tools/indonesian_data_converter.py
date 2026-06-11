"""Data Converter untuk Blood Demand Data (Indonesia PMI Format)

Converter untuk format:
- Semicolon-separated (Excel Indonesia)
- Monthly aggregate data dengan breakdown per tipe darah
- Multiple product types (PRC, TC, FFP, AHF)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re
import sys

# ============================================================================
# INDONESIAN FORMAT PARSER
# ============================================================================

class IndonesianBloodDataConverter:
    """
    Converter untuk format data darah Indonesia (PMI Jakarta Barat style)
    
    Input format:
    - Semicolon-separated (;) bukan comma
    - Monthly data dengan breakdown per jenis darah (O, A, B, AB)
    - Multiple komponen: PRC, TC, FFP, AHF
    
    Output format:
    - CSV dengan kolom: date, blood_type, demand, collection
    - Daily data (disbanding dari monthly)
    """
    
    # Mapping tipe darah
    BLOOD_TYPE_GROUPS = {
        'O': ['O+', 'O-'],
        'A': ['A+', 'A-'],
        'B': ['B+', 'B-'],
        'AB': ['AB+', 'AB-']
    }
    
    MONTHS_ID = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
        'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
        'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }
    
    def __init__(self, input_file):
        """
        Parameters:
        -----------
        input_file : str
            Path ke file DATA DARAH.csv
        """
        self.input_file = input_file
        self.raw_data = None
        self.processed_data = None
        
        print(f"\n[INDONESIAN BLOOD DATA CONVERTER]")
        print(f"Input file: {input_file}")
    
    def load_and_parse(self):
        """
        Load dan parse format Indonesia (semicolon-separated)
        """
        print(f"\n[1] Loading Indonesian format data...")
        
        try:
            # Read dengan semicolon separator
            df = pd.read_csv(self.input_file, sep=';', encoding='utf-8')
            
            # Alternative encoding jika gagal
            if df.empty:
                df = pd.read_csv(self.input_file, sep=';', encoding='latin-1')
            
            print(f"    ✓ File loaded")
            print(f"    Rows: {len(df)}")
            print(f"    Columns: {list(df.columns[:10])}...")
            
            self.raw_data = df
            return True
        
        except Exception as e:
            print(f"    ✗ Error loading file: {e}")
            return False
    
    def extract_monthly_data(self):
        """
        Extract monthly demand dan blood type breakdown
        
        Expected columns:
        - Bulan: Nama bulan (Januari, Februari, dst)
        - Total Permintaan (Kantong): Total demand
        - PRC O, PRC A, PRC B, PRC AB: PRC per blood type
        - TC O, TC A, TC B, TC AB: TC per blood type
        - FFP O, FFP A, FFP B, FFP AB: FFP per blood type
        - AHF O, AHF A, AHF B, AHF AB: AHF per blood type
        """
        
        print(f"\n[2] Extracting monthly data...")
        
        months_data = []
        
        for idx, row in self.raw_data.iterrows():
            # Get bulan
            bulan_str = str(row.iloc[0]).strip() if len(row) > 0 else None
            
            if bulan_str not in self.MONTHS_ID:
                continue
            
            bulan_num = self.MONTHS_ID[bulan_str]
            
            # Extract total demand
            try:
                # Cek kolom "Total Permintaan (Kantong)" atau index ke-1
                total_demand = float(str(row.iloc[1]).replace('.', '').replace(',', '.'))
            except:
                continue
            
            # Extract blood type breakdowns
            blood_data = {}
            
            # Column indices untuk PRC, TC, FFP, AHF
            # Struktur: [0]=Bulan, [1]=Total, [2-5]=PRC O/A/B/AB, [6-9]=TC, [10-13]=FFP, [14-17]=AHF
            try:
                # PRC
                prc_o = float(str(row.iloc[2]).replace('.', '').replace(',', '.'))
                prc_a = float(str(row.iloc[3]).replace('.', '').replace(',', '.'))
                prc_b = float(str(row.iloc[4]).replace('.', '').replace(',', '.'))
                prc_ab = float(str(row.iloc[5]).replace('.', '').replace(',', '.'))
                
                # TC
                tc_o = float(str(row.iloc[6]).replace('.', '').replace(',', '.'))
                tc_a = float(str(row.iloc[7]).replace('.', '').replace(',', '.'))
                tc_b = float(str(row.iloc[8]).replace('.', '').replace(',', '.'))
                tc_ab = float(str(row.iloc[9]).replace('.', '').replace(',', '.'))
                
                # FFP
                ffp_o = float(str(row.iloc[10]).replace('.', '').replace(',', '.'))
                ffp_a = float(str(row.iloc[11]).replace('.', '').replace(',', '.'))
                ffp_b = float(str(row.iloc[12]).replace('.', '').replace(',', '.'))
                ffp_ab = float(str(row.iloc[13]).replace('.', '').replace(',', '.'))
                
                # AHF
                ahf_o = float(str(row.iloc[14]).replace('.', '').replace(',', '.'))
                ahf_a = float(str(row.iloc[15]).replace('.', '').replace(',', '.'))
                ahf_b = float(str(row.iloc[16]).replace('.', '').replace(',', '.'))
                ahf_ab = float(str(row.iloc[17]).replace('.', '').replace(',', '.'))
                
                # Aggregate per blood type
                blood_data = {
                    'O': prc_o + tc_o + ffp_o + ahf_o,
                    'A': prc_a + tc_a + ffp_a + ahf_a,
                    'B': prc_b + tc_b + ffp_b + ahf_b,
                    'AB': prc_ab + tc_ab + ffp_ab + ahf_ab
                }
                
                months_data.append({
                    'bulan': bulan_num,
                    'total_demand': total_demand,
                    'blood_O': blood_data['O'],
                    'blood_A': blood_data['A'],
                    'blood_B': blood_data['B'],
                    'blood_AB': blood_data['AB']
                })
            
            except Exception as e:
                print(f"    Warning: Could not parse row {idx}: {e}")
                continue
        
        print(f"    ✓ Extracted {len(months_data)} months")
        
        return pd.DataFrame(months_data)
    
    def expand_to_daily(self, monthly_df, year=2024):
        """
        Expand monthly data ke daily data
        
        Assumsi: Demand terdistribusi uniform setiap hari dalam bulan
        Collection diasumsikan slightly lower (collection rate ~80-90%)
        
        Parameters:
        -----------
        monthly_df : pd.DataFrame
            Monthly aggregated data
        year : int
            Tahun untuk data (default 2024)
        
        Returns:
        --------
        pd.DataFrame : Daily data dengan kolom (date, blood_type, demand, collection)
        """
        
        print(f"\n[3] Expanding to daily data...")
        
        daily_records = []
        
        for _, month_row in monthly_df.iterrows():
            bulan = int(month_row['bulan'])
            total_demand = month_row['total_demand']
            
            # Get days in month
            if bulan == 12:
                next_date = datetime(year + 1, 1, 1)
            else:
                next_date = datetime(year, bulan + 1, 1)
            
            current_date = datetime(year, bulan, 1)
            days_in_month = (next_date - current_date).days
            
            # Daily demand per blood type
            for blood_base in ['O', 'A', 'B', 'AB']:
                blood_demand_monthly = month_row[f'blood_{blood_base}']
                daily_demand = blood_demand_monthly / days_in_month
                
                # Add slight variation to each day
                for day in range(days_in_month):
                    date = current_date + timedelta(days=day)
                    
                    # Add random variation (±10%)
                    variation = np.random.uniform(0.9, 1.1)
                    day_demand = int(daily_demand * variation)
                    
                    # Collection rate (80-90%)
                    collection_rate = np.random.uniform(0.8, 0.9)
                    day_collection = int(day_demand * collection_rate)
                    
                    # Add both positive and negative blood types
                    for blood_type in self.BLOOD_TYPE_GROUPS[blood_base]:
                        # Split demand between + and -
                        split = np.random.uniform(0.4, 0.6)
                        demand_split = int(day_demand * split)
                        collection_split = int(day_collection * split)
                        
                        daily_records.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'blood_type': blood_type,
                            'demand': demand_split,
                            'collection': collection_split
                        })
        
        daily_df = pd.DataFrame(daily_records)
        print(f"    ✓ Expanded to {len(daily_df):,} daily records")
        print(f"    Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        
        return daily_df
    
    def convert(self, year=2024, output_file='data/processed/blood_demand_data_converted.csv'):
        """
        Main conversion pipeline
        
        Parameters:
        -----------
        year : int
            Year untuk data (default 2024)
        output_file : str
            Path untuk output file
        """
        
        print(f"\n" + "="*80)
        print("INDONESIAN BLOOD DATA CONVERSION")
        print("="*80)
        
        # Step 1: Load
        if not self.load_and_parse():
            return False
        
        # Step 2: Extract
        monthly_df = self.extract_monthly_data()
        if monthly_df.empty:
            print(f"    ✗ No monthly data extracted")
            return False
        
        # Step 3: Expand
        daily_df = self.expand_to_daily(monthly_df, year=year)
        
        # Step 4: Save
        print(f"\n[4] Saving converted data...")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(output_file, index=False)
        
        print(f"    ✓ Saved to: {output_file}")
        print(f"    Records: {len(daily_df):,}")
        
        # Step 5: Validation
        print(f"\n[5] Validation Summary")
        print(f"    Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        print(f"    Blood types: {', '.join(sorted(daily_df['blood_type'].unique()))}")
        print(f"    Avg daily demand: {daily_df['demand'].mean():.2f}")
        print(f"    Collection rate: {(daily_df['collection'].sum() / daily_df['demand'].sum() * 100):.2f}%")
        
        print(f"\n" + "="*80)
        print("✓ CONVERSION COMPLETED")
        print("="*80)
        
        return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# INDONESIAN BLOOD DATA CONVERTER")
    print("#"*80)
    
    # Check arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    else:
        input_file = 'data/raw/DATA DARAH.csv'
        year = 2024
    
    # Convert
    converter = IndonesianBloodDataConverter(input_file)
    
    if converter.convert(year=year):
        print(f"\nNext step:")
        print(f"  python tools/data_upload.py data/processed/blood_demand_data_converted.csv")
    else:
        print(f"\nConversion failed. Check file format and try again.")
    
    print("\n" + "#"*80 + "\n")
