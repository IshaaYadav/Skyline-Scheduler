import pandas as pd
import os

def calculate_cascade_effects(df: pd.DataFrame, min_turnaround_mins: int = 30) -> pd.DataFrame:
    """
    Analyzes the cascading (knock-on) effect of delays for each aircraft.

    Args:
        df: The preprocessed flight data DataFrame.
        min_turnaround_mins: The minimum expected time for an aircraft to be on the ground.

    Returns:
        A DataFrame with flights ranked by their cascading delay impact.
    """
    print("Analyzing cascading delay effects...")

    # Ensure timestamp columns are in datetime format
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
    df['actual_departure'] = pd.to_datetime(df['actual_departure'])
    df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])
    df['actual_arrival'] = pd.to_datetime(df['actual_arrival'])

    # Sort flights by aircraft and then by scheduled departure time
    df = df.sort_values(by=['aircraft', 'scheduled_departure']).reset_index(drop=True)

    # Group by aircraft to analyze flight chains
    df_grouped = df.groupby('aircraft')

    cascade_data = []

    for aircraft_id, group in df_grouped:
        # Shift the data to compare a flight with its next flight
        group['next_flight_dep'] = group['actual_departure'].shift(-1)
        group['next_flight_delay'] = group['departure_delay'].shift(-1)
        
        # Calculate turnaround time in minutes
        group['turnaround_mins'] = (group['next_flight_dep'] - group['actual_arrival']).dt.total_seconds() / 60
        
        # Identify propagated delays
        # A delay is propagated if:
        # 1. The current flight arrived late.
        # 2. The next flight departed late.
        # 3. The turnaround time was less than the arrival delay + a minimum ground time,
        #    implying no time to 'catch up'.
        
        propagated_delay = (group['arrival_delay'] > 0) & \
                           (group['next_flight_delay'] > 0) & \
                           (group['turnaround_mins'] < (group['arrival_delay'] + min_turnaround_mins))
        
        # The cascade amount is the delay of the next flight, capped at the current flight's arrival delay
        group['cascade_amount'] = group[['arrival_delay', 'next_flight_delay']].min(axis=1).where(propagated_delay, 0)
        
        cascade_data.append(group)

    if not cascade_data:
        print("No cascade data could be generated.")
        return pd.DataFrame()

    # Combine the processed groups back into a single DataFrame
    cascade_df = pd.concat(cascade_data, ignore_index=True)

    # Calculate a total cascade score for each flight
    # For simplicity, we'll use the direct cascade amount as the score
    cascade_df.rename(columns={'cascade_amount': 'cascade_score'}, inplace=True)
    
    # Sort by the highest impact
    cascade_df = cascade_df.sort_values(by='cascade_score', ascending=False)
    
    print("Cascade analysis complete.")
    return cascade_df


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'outputs')
    
    processed_data_path = os.path.join(output_path, '02_processed_data.csv')
    if not os.path.exists(processed_data_path):
        print(f"Error: Input file not found at {processed_data_path}")
        print("Please run `core/preprocess.py` first.")
    else:
        processed_df = pd.read_csv(processed_data_path)
        
        # Calculate cascade effects
        cascade_results_df = calculate_cascade_effects(processed_df)
        
        # Save the results
        cascade_data_path = os.path.join(output_path, '05_cascade_analysis.csv')
        cascade_results_df.to_csv(cascade_data_path, index=False)
        print(f"\nSaved cascade analysis data to {cascade_data_path}")

        # Display the top 10 flights with the biggest cascading impact
        print("\n--- Top 10 Flights with the Biggest Cascading Impact ---")
        top_impact_flights = cascade_results_df.head(10)
        
        # Select relevant columns for display
        display_cols = [
            'flight_date', 'flight_number', 'aircraft', 'from_airport', 'to_airport',
            'arrival_delay', 'turnaround_mins', 'cascade_score'
        ]
        # Filter to columns that actually exist in the dataframe
        display_cols_exist = [col for col in display_cols if col in top_impact_flights.columns]
        
        print(top_impact_flights[display_cols_exist])
