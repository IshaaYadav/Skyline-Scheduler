import pandas as pd
import os
import numpy as np

def calculate_hourly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates hourly KPIs for airport traffic and delays.

    Args:
        df: The preprocessed flight data DataFrame.

    Returns:
        A DataFrame with hourly statistics:
        - total_departures
        - total_arrivals
        - total_flights
        - average_departure_delay
        - average_arrival_delay
    """
    print("Calculating hourly KPIs...")

    # --- Data Type Correction Patch ---
    # Ensure delay columns are numeric, coercing any errors to NaN.
    # This prevents the TypeError during aggregation.
    df['departure_delay'] = pd.to_numeric(df['departure_delay'], errors='coerce')
    df['arrival_delay'] = pd.to_numeric(df['arrival_delay'], errors='coerce')

    # Ensure timestamp columns are in datetime format
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
    df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])

    # --- Departure KPIs ---
    # Extract hour from scheduled departure
    df['departure_hour'] = df['scheduled_departure'].dt.hour
    
    # Group by hour and calculate departure stats
    departure_kpis = df.groupby('departure_hour').agg(
        total_departures=('flight_number', 'count'),
        average_departure_delay=('departure_delay', 'mean')
    ).reset_index()
    departure_kpis.rename(columns={'departure_hour': 'hour'}, inplace=True)

    # --- Arrival KPIs ---
    # Extract hour from scheduled arrival
    df['arrival_hour'] = df['scheduled_arrival'].dt.hour
    
    # Group by hour and calculate arrival stats
    arrival_kpis = df.groupby('arrival_hour').agg(
        total_arrivals=('flight_number', 'count'),
        average_arrival_delay=('arrival_delay', 'mean')
    ).reset_index()
    arrival_kpis.rename(columns={'arrival_hour': 'hour'}, inplace=True)

    # --- Merge KPIs ---
    # Merge departure and arrival KPIs on the hour
    hourly_summary = pd.merge(departure_kpis, arrival_kpis, on='hour', how='outer')
    
    # Fill NaN values with 0 for counts, as no activity means 0 flights
    hourly_summary['total_departures'] = hourly_summary['total_departures'].fillna(0).astype(int)
    hourly_summary['total_arrivals'] = hourly_summary['total_arrivals'].fillna(0).astype(int)

    # Calculate total flights
    hourly_summary['total_flights'] = hourly_summary['total_departures'] + hourly_summary['total_arrivals']

    # Round delay values for readability
    hourly_summary['average_departure_delay'] = hourly_summary['average_departure_delay'].round(2)
    hourly_summary['average_arrival_delay'] = hourly_summary['average_arrival_delay'].round(2)

    print("Hourly KPIs calculated successfully.")
    return hourly_summary


def find_busiest_slots(hourly_kpis: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Identifies the busiest time slots based on total flight volume.

    Args:
        hourly_kpis: The DataFrame with hourly KPI calculations.
        top_n: The number of busiest slots to return.

    Returns:
        A DataFrame showing the top N busiest hours.
    """
    return hourly_kpis.sort_values(by='total_flights', ascending=False).head(top_n)


def find_best_slots(hourly_kpis: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Identifies the best time slots with the lowest average delays.
    This considers both arrival and departure delays.

    Args:
        hourly_kpis: The DataFrame with hourly KPI calculations.
        top_n: The number of best slots to return.

    Returns:
        A DataFrame showing the top N best hours for operations.
    """
    # Calculate a combined average delay, giving equal weight to departure and arrival
    # We use nanmean to handle cases where an hour might only have departures or arrivals
    hourly_kpis['combined_avg_delay'] = np.nanmean(hourly_kpis[['average_departure_delay', 'average_arrival_delay']], axis=1)
    
    return hourly_kpis.sort_values(by='combined_avg_delay', ascending=True).head(top_n)


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'outputs')
    
    # Load the processed data from the previous step
    processed_data_path = os.path.join(output_path, '02_processed_data.csv')
    if not os.path.exists(processed_data_path):
        print(f"Error: Input file not found at {processed_data_path}")
        print("Please run `core/preprocess.py` first.")
    else:
        processed_df = pd.read_csv(processed_data_path)
        
        # Calculate the hourly KPIs
        hourly_kpis_df = calculate_hourly_kpis(processed_df)
        
        # Save the hourly KPIs
        kpi_data_path = os.path.join(output_path, '03_hourly_kpis.csv')
        hourly_kpis_df.to_csv(kpi_data_path, index=False)
        print(f"\nSaved hourly KPI data to {kpi_data_path}")

        # --- Find and Print Insights ---
        
        # Busiest Slots
        busiest_hours = find_busiest_slots(hourly_kpis_df)
        print("\n--- Busiest Time Slots to Avoid (by total flights) ---")
        print(busiest_hours[['hour', 'total_flights', 'average_departure_delay', 'average_arrival_delay']])

        # Best Slots
        best_hours = find_best_slots(hourly_kpis_df)
        print("\n--- Best Time Slots for Operations (by lowest combined delay) ---")
        print(best_hours[['hour', 'total_flights', 'combined_avg_delay']])
