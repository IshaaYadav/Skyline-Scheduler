import pandas as pd
import re

def extract_entities(query: str, df: pd.DataFrame):
    """
    Extracts destination and day of the week from the user query.
    """
    destination_code = None
    day_of_week = None
    
    # Create a mapping of airport names (e.g., 'Chandigarh') to codes ('IXC')
    airport_map = df[['to_full', 'to_airport']].drop_duplicates().dropna()
    airport_name_map = {re.search(r'^(.*?)\s*\(', name).group(1).lower(): code for name, code in zip(airport_map['to_full'], airport_map['to_airport']) if re.search(r'^(.*?)\s*\(', name)}

    # Extract destination
    for name, code in airport_name_map.items():
        if name in query.lower():
            destination_code = code
            break
            
    # Extract day of the week
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days:
        if day in query.lower():
            day_of_week = day.capitalize()
            break
            
    return destination_code, day_of_week

def calculate_filtered_kpis(df: pd.DataFrame):
    """
    Calculates hourly KPIs on a pre-filtered DataFrame.
    """
    if df.empty:
        return pd.DataFrame()
        
    # Simplified KPI calculation from core/kpis.py
    df['departure_hour'] = pd.to_datetime(df['scheduled_departure']).dt.hour
    kpis = df.groupby('departure_hour').agg(
        total_flights=('flight_number', 'count'),
        average_departure_delay=('departure_delay', 'mean')
    ).reset_index()
    kpis.rename(columns={'departure_hour': 'hour'}, inplace=True)
    return kpis


def parse_query(query: str, processed_df: pd.DataFrame, cascade_df: pd.DataFrame):
    """
    Parses a natural language query with entity extraction to retrieve insights.
    """
    query_lower = query.lower().strip()
    destination, day = extract_entities(query, processed_df)
    
    filtered_df = processed_df.copy()
    title_suffix = ""

    # Apply filters if entities were found
    if destination:
        filtered_df = filtered_df[filtered_df['to_airport'] == destination]
        dest_name = processed_df[processed_df['to_airport'] == destination]['to_full'].iloc[0]
        title_suffix += f" to {dest_name}"
    if day:
        filtered_df = filtered_df[filtered_df['day_of_week'] == day]
        title_suffix += f" on a {day}"

    if filtered_df.empty and (destination or day):
        return "No Flights Found", f"I couldn't find any flights matching your criteria ({title_suffix.strip()})."

    # --- Intent: Busiest Times ---
    if any(keyword in query_lower for keyword in ["busiest", "most flights", "most traffic"]):
        title = f"Busiest Hours{title_suffix}"
        kpis = calculate_filtered_kpis(filtered_df)
        result_df = kpis.sort_values('total_flights', ascending=False).head(5)
        return title, result_df

    # --- Intent: Best Times (Lowest Delays) ---
    if any(keyword in query_lower for keyword in ["best time", "least busy", "lowest delay"]):
        title = f"Best Hours (Lowest Delay){title_suffix}"
        kpis = calculate_filtered_kpis(filtered_df)
        result_df = kpis.sort_values('average_departure_delay').head(5)
        return title, result_df

    # --- Intent: Cascade/Impactful Flights (not affected by filters) ---
    if any(keyword in query_lower for keyword in ["cascade", "impact", "knock-on", "disruptive"]):
        title = "Top Flights Causing Cascading Delays"
        display_cols = ['flight_date', 'flight_number', 'aircraft', 'from_airport', 'to_airport', 'arrival_delay', 'cascade_score']
        display_cols_exist = [col for col in display_cols if col in cascade_df.columns]
        return title, cascade_df.head(5)[display_cols_exist]

    return "Query Not Understood", "I can answer questions about 'busiest hours' and 'best times' for specific destinations and days. Please try again."
