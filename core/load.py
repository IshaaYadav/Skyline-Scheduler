import pandas as pd
import glob
import os
import re

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes DataFrame column names.
    - Converts to lowercase
    - Replaces spaces and special characters with underscores
    - Strips leading/trailing whitespace and underscores
    """
    new_columns = {}
    columns = [str(c) for c in df.columns]
    for col in columns:
        new_col = col.lower()
        new_col = re.sub(r'[^a-zA-Z0-9]', '_', new_col)
        new_col = re.sub(r'_+', '_', new_col)
        new_col = re.sub(r'unnamed_\d+', '', new_col)
        new_col = new_col.strip('_')
        new_columns[col] = new_col
    
    rename_map = {str(old_col): new_col for old_col, new_col in zip(df.columns, new_columns.values())}
    df = df.rename(columns=rename_map)
    
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    
    return df

def load_flight_data(data_dir: str) -> pd.DataFrame:
    """
    Loads all flight data Excel files from a directory into a single DataFrame.
    This version is robust to inconsistent headers and file structures.

    Args:
        data_dir: The path to the directory containing the flight data .xlsx files.

    Returns:
        A pandas DataFrame with the combined and initially cleaned flight data.
    """
    print(f"Searching for Excel files in: {data_dir}")
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel (.xlsx) files found in the directory: {data_dir}")

    print(f"Found {len(excel_files)} files: {excel_files}")

    all_dfs = []
    for f in excel_files:
        try:
            df_raw = pd.read_excel(f, header=None)
            
            header_row_index = -1
            for i, row in df_raw.iterrows():
                row_str = ' '.join(str(x) for x in row.dropna())
                if 'From' in row_str and 'To' in row_str and ('STD' in row_str or 'STA' in row_str):
                    header_row_index = i
                    break
            
            if header_row_index == -1:
                print(f"Warning: Could not find a valid header row in {f}. Skipping file.")
                continue

            df = pd.read_excel(f, header=header_row_index)
            df = clean_col_names(df)

            # --- FIX: Intelligently find and rename the date column ---
            cleaned_columns = df.columns.tolist()
            if 'date' not in cleaned_columns:
                print("Warning: 'date' column not found. Attempting to locate it by position.")
                try:
                    from_index = cleaned_columns.index('from')
                    potential_date_col_index = from_index - 1
                    if potential_date_col_index >= 0:
                        col_to_rename = cleaned_columns[potential_date_col_index]
                        print(f"Found potential date column at index {potential_date_col_index} with name '{col_to_rename}'. Renaming to 'date'.")
                        cleaned_columns[potential_date_col_index] = 'date'
                        df.columns = cleaned_columns
                except ValueError:
                    print("Could not find 'from' column to infer 'date' column position.")

            if 'flight_number' in df.columns:
                df['flight_number'] = df['flight_number'].fillna(method='ffill')
                df.dropna(subset=['from', 'to'], inplace=True)
                all_dfs.append(df)
            else:
                 print(f"Warning: 'flight_number' column not found in {f} after cleaning. Skipping file.")

        except Exception as e:
            print(f"Could not read or process file {f}: {e}")
            continue

    if not all_dfs:
        print("No dataframes were created. Please check the Excel files for valid headers and data.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print("Successfully loaded and combined data.")
    print(f"Total rows: {len(combined_df)}")
    print("Columns:", combined_df.columns.tolist())
    
    return combined_df

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data')
    
    flight_df = load_flight_data(data_path)
    
    if not flight_df.empty:
        print("\nFirst 5 rows of the loaded data:")
        print(flight_df.head())
        
        output_path = os.path.join(project_root, 'outputs')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        flight_df.to_csv(os.path.join(output_path, '01_loaded_data_from_excel.csv'), index=False)
        print(f"\nSaved initial loaded data to outputs/01_loaded_data_from_excel.csv")
