import pandas as pd
import os
import numpy as np

def parse_time_from_status(status_str: str) -> str:
    """
    Extracts time (HH:MM) from a status string like 'Landed 11:15 AM'.
    Returns None if no time is found.
    """
    if not isinstance(status_str, str):
        return None
    
    # Handle formats like "Landed 1:48 PM" or "Landed 11:15 AM"
    try:
        # Convert to datetime and then format to 24-hour HH:MM
        dt_obj = pd.to_datetime(status_str.replace('Landed ', '').strip(), format='%I:%M %p', errors='coerce')
        if pd.notna(dt_obj):
            return dt_obj.strftime('%H:%M')
    except Exception:
        pass
        
    return None


def preprocess_flight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, transforms, and engineers features for the flight data.

    Args:
        df: The raw flight data DataFrame from load.py.

    Returns:
        A preprocessed DataFrame ready for analysis.
    """
    print("Starting preprocessing...")
    
    # --- 0. Data Integrity Patch ---
    # This patch handles cases where a malformed header in the source data
    # causes the columns to be shifted to the left.
    if 'date' not in df.columns and 'from' in df.columns:
        print("Warning: 'date' column not found. Checking for shifted columns...")
        # Check if the 'from' column likely contains dates by testing the first value
        try:
            first_valid_from = df['from'].dropna().iloc[0]
            pd.to_datetime(first_valid_from)
            is_shifted = True
            print(f"Detected date-like data ('{first_valid_from}') in 'from' column. Applying patch.")
        except (ValueError, TypeError, IndexError):
            is_shifted = False
            print("'from' column does not appear to contain dates. Cannot apply patch.")

        if is_shifted:
            # Get the names of the columns that need to be shifted right
            shifted_col_names = df.columns[df.columns.get_loc('from'):].tolist()
            
            # Create the new 'date' column from the old 'from' column's data
            df['date'] = df['from']
            
            # Shift the data in the remaining columns one position to the right
            for i in range(len(shifted_col_names) - 1):
                df[shifted_col_names[i]] = df[shifted_col_names[i+1]]
            
            # The last column's data is now invalid, so clear it
            df[shifted_col_names[-1]] = np.nan
            print("Successfully patched shifted columns.")


    # --- 1. Data Type Conversion and Cleaning ---
    if 'ata' in df.columns:
        df.rename(columns={'ata': 'status_text'}, inplace=True)
        df['ata'] = df['status_text'].apply(parse_time_from_status)
    else:
        df['ata'] = np.nan

    time_cols = ['std', 'atd', 'sta', 'ata']
    
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['-', 'nan', 'None', ''], np.nan)

    if 'date' not in df.columns:
         raise KeyError("Critical 'date' column is missing after attempting patch. Please check the source data and load.py script.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # --- 2. Combine Date and Time into Timestamps ---
    for col in time_cols:
        if col in df.columns:
            df[f'{col}_ts'] = pd.to_datetime(df['date'].dt.strftime('%Y-%m-%d') + ' ' + df[col], errors='coerce')

    if 'ata_ts' in df.columns and 'atd_ts' in df.columns:
        arrival_earlier = df['ata_ts'] < df['atd_ts']
        df.loc[arrival_earlier, 'ata_ts'] += pd.Timedelta(days=1)
    
    if 'sta_ts' in df.columns and 'std_ts' in df.columns:
        sta_earlier = df['sta_ts'] < df['std_ts']
        df.loc[sta_earlier, 'sta_ts'] += pd.Timedelta(days=1)

    # --- 3. Feature Engineering ---
    if 'atd_ts' in df.columns and 'std_ts' in df.columns:
        df['departure_delay_mins'] = (df['atd_ts'] - df['std_ts']).dt.total_seconds() / 60
    
    if 'ata_ts' in df.columns and 'sta_ts' in df.columns:
        df['arrival_delay_mins'] = (df['ata_ts'] - df['sta_ts']).dt.total_seconds() / 60

    if 'std_ts' in df.columns:
        df['scheduled_hour'] = df['std_ts'].dt.hour
        df['day_of_week'] = df['std_ts'].dt.day_name()

    if 'from' in df.columns:
        df['from_airport'] = df['from'].str.extract(r'\((\w+)\)')
    if 'to' in df.columns:
        df['to_airport'] = df['to'].str.extract(r'\((\w+)\)')
    
    # --- 4. Final Cleanup ---
    final_cols = {
        'date': 'flight_date', 'flight_number': 'flight_number', 'from': 'from_full', 'to': 'to_full',
        'from_airport': 'from_airport', 'to_airport': 'to_airport', 'aircraft': 'aircraft',
        'std_ts': 'scheduled_departure', 'atd_ts': 'actual_departure', 'sta_ts': 'scheduled_arrival',
        'ata_ts': 'actual_arrival', 'departure_delay_mins': 'departure_delay',
        'arrival_delay_mins': 'arrival_delay', 'scheduled_hour': 'scheduled_hour', 'day_of_week': 'day_of_week'
    }
    
    df_processed = df.rename(columns=final_cols)
    
    final_cols_exist = [col for col in final_cols.values() if col in df_processed.columns]
    df_processed = df_processed[final_cols_exist]

    critical_cols = ['flight_date', 'flight_number', 'scheduled_departure', 'actual_departure', 'scheduled_arrival', 'actual_arrival']
    critical_cols_exist = [col for col in critical_cols if col in df_processed.columns]
    if critical_cols_exist:
        df_processed.dropna(subset=critical_cols_exist, inplace=True)

    print("Preprocessing complete.")
    print(f"Total rows after processing: {len(df_processed)}")
    
    return df_processed


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'outputs')
    
    loaded_data_path = os.path.join(output_path, '01_loaded_data_from_excel.csv')
    if not os.path.exists(loaded_data_path):
        print(f"Error: Input file not found at {loaded_data_path}")
        print("Please run `core/load.py` first.")
    else:
        try:
            raw_df = pd.read_csv(loaded_data_path, low_memory=False)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            raw_df = pd.DataFrame()

        if not raw_df.empty:
            processed_df = preprocess_flight_data(raw_df)
            
            print("\nFirst 5 rows of the processed data:")
            print(processed_df.head())
            
            processed_data_path = os.path.join(output_path, '02_processed_data.csv')
            processed_df.to_csv(processed_data_path, index=False)
            print(f"\nSaved processed data to {processed_data_path}")
