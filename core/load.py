import pandas as pd
import datetime

def _parse_flight_time(val):
    """Convert flight time into minutes safely."""
    if pd.isna(val):
        return None
    if isinstance(val, datetime.timedelta):
        return int(val.total_seconds() // 60)
    if isinstance(val, datetime.time):
        return val.hour * 60 + val.minute
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        try:
            td = pd.to_timedelta(val)
            return int(td.total_seconds() // 60)
        except Exception:
            return None
    return None


def load_flights(filepath: str) -> pd.DataFrame:
    """
    Load flight data from Excel or CSV and standardize columns.

    Args:
        filepath (str): Path to flight data file.

    Returns:
        pd.DataFrame: Cleaned flight schedule data.
    """
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Convert flight_time to minutes
    if "flight_time" in df.columns:
        df["flight_time_min"] = df["flight_time"].apply(_parse_flight_time)

    return df
