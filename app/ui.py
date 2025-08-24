import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="Flight Scheduling Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---

@st.cache_data
def load_data(file_path):
    """Loads data from a CSV file with caching."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_resource
def load_model_artifacts(model_dir):
    """Loads the trained model, encoder, and columns."""
    model_path = os.path.join(model_dir, 'delay_predictor.joblib')
    encoder_path = os.path.join(model_dir, 'encoder.joblib')
    columns_path = os.path.join(model_dir, 'model_columns.joblib')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(columns_path):
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        model_cols = joblib.load(columns_path)
        return model, encoder, model_cols
    return None, None, None

def predict_delay(model, encoder, model_cols, hour, day, airport):
    """Predicts delay for a given flight scenario."""
    # Create a DataFrame for the input
    input_df = pd.DataFrame([[hour, day, airport]], columns=['scheduled_hour', 'day_of_week', 'to_airport'])
    
    # Encode categorical features
    encoded_features = encoder.transform(input_df[['day_of_week', 'to_airport']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['day_of_week', 'to_airport']))
    
    # Combine with numerical features
    final_input = pd.concat([input_df[['scheduled_hour']], encoded_df], axis=1)
    
    # Ensure all model columns are present
    final_input = final_input.reindex(columns=model_cols, fill_value=0)
    
    # Make prediction
    prediction = model.predict(final_input)[0]
    return prediction

# --- Data and Model Loading ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_PATH, 'models')

kpi_df = load_data(os.path.join(OUTPUT_PATH, '03_hourly_kpis.csv'))
cascade_df = load_data(os.path.join(OUTPUT_PATH, '05_cascade_analysis.csv'))
processed_df = load_data(os.path.join(OUTPUT_PATH, '02_processed_data.csv'))
model, encoder, model_cols = load_model_artifacts(MODEL_DIR)


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Airport Overview", "Delay Prediction", "Cascade Analysis"])

# --- Main App ---

if page == "Airport Overview":
    st.title("✈️ Airport Performance Overview")
    st.markdown("Analyzing flight traffic and delays by hour at Mumbai Airport (BOM).")

    if kpi_df is not None:
        # Display KPIs
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Busiest Hours")
            busiest = kpi_df.sort_values('total_flights', ascending=False).head(5)
            st.dataframe(busiest[['hour', 'total_flights', 'average_departure_delay', 'average_arrival_delay']])
        
        with col2:
            st.subheader("Best Hours (Lowest Delays)")
            kpi_df['combined_avg_delay'] = kpi_df[['average_departure_delay', 'average_arrival_delay']].mean(axis=1)
            best = kpi_df.sort_values('combined_avg_delay').head(5)
            st.dataframe(best[['hour', 'total_flights', 'combined_avg_delay']])

        # Display Plots
        st.header("Visual Analysis")
        
        # Flight Volume Plot
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=kpi_df['hour'], y=kpi_df['total_departures'], name='Departures', marker_color='indianred'))
        fig_vol.add_trace(go.Bar(x=kpi_df['hour'], y=kpi_df['total_arrivals'], name='Arrivals', marker_color='lightsalmon'))
        fig_vol.update_layout(barmode='stack', title_text='<b>Total Flight Volume by Hour</b>', xaxis_title='Hour', yaxis_title='Number of Flights', template='plotly_white')
        st.plotly_chart(fig_vol, use_container_width=True)

        # Delay vs Volume Plot
        fig_delay = make_subplots(specs=[[{"secondary_y": True}]])
        fig_delay.add_trace(go.Bar(x=kpi_df['hour'], y=kpi_df['total_flights'], name='Total Flights', marker_color='lightblue'), secondary_y=False)
        fig_delay.add_trace(go.Scatter(x=kpi_df['hour'], y=kpi_df['average_departure_delay'], name='Avg. Departure Delay', mode='lines+markers', line=dict(color='firebrick')), secondary_y=True)
        fig_delay.add_trace(go.Scatter(x=kpi_df['hour'], y=kpi_df['average_arrival_delay'], name='Avg. Arrival Delay', mode='lines+markers', line=dict(color='seagreen')), secondary_y=True)
        fig_delay.update_layout(title_text='<b>Average Delays vs. Flight Volume by Hour</b>', template='plotly_white')
        fig_delay.update_yaxes(title_text="Number of Flights", secondary_y=False)
        fig_delay.update_yaxes(title_text="Average Delay (Minutes)", secondary_y=True)
        st.plotly_chart(fig_delay, use_container_width=True)

    else:
        st.error("KPI data not found. Please run `core/kpis.py` first.")


elif page == "Delay Prediction":
    st.title("🔮 What-If: Predict Departure Delay")
    st.markdown("Tune the schedule for a flight and see the predicted impact on its departure delay.")

    if model is not None and processed_df is not None:
        # Create a mapping from full airport name to airport code for the dropdown
        airport_map_df = processed_df[['to_full', 'to_airport']].drop_duplicates().dropna()
        airport_map_df = airport_map_df.sort_values('to_full')
        
        # Create a dictionary for easy lookup: { 'Chennai (MAA)': 'MAA', ... }
        airport_name_to_code_map = pd.Series(airport_map_df.to_airport.values, index=airport_map_df.to_full).to_dict()

        col1, col2, col3 = st.columns(3)
        with col1:
            hour = st.slider("Scheduled Hour (24H)", 0, 23, 10)
        with col2:
            day = st.selectbox("Day of the Week", processed_df['day_of_week'].dropna().unique())
        with col3:
            # Use the full names for the dropdown options
            selected_airport_name = st.selectbox("Destination Airport", options=airport_name_to_code_map.keys())

        if st.button("Predict Delay"):
            # Get the corresponding airport code from the selected name
            airport_code = airport_name_to_code_map[selected_airport_name]
            
            # Use the code for prediction
            predicted_delay = predict_delay(model, encoder, model_cols, hour, day, airport_code)
            
            st.metric(label="Predicted Departure Delay", value=f"{predicted_delay:.2f} minutes")
            if predicted_delay > 20:
                st.warning("High delay predicted. Consider rescheduling.")
            else:
                st.success("Predicted delay is within an acceptable range.")
    else:
        st.error("Model or processed data not found. Please run `core/delay_model.py` and `core/preprocess.py` first.")


elif page == "Cascade Analysis":
    st.title("🔗 Cascade Effect Analysis")
    st.markdown("Identifying flights that cause the largest knock-on delays to subsequent flights.")

    if cascade_df is not None:
        st.subheader("Top 10 Most Impactful Flights")
        
        # Filter to essential columns for display
        display_cols = [
            'flight_date', 'flight_number', 'aircraft', 'from_airport', 'to_airport',
            'arrival_delay', 'turnaround_mins', 'cascade_score'
        ]
        display_cols_exist = [col for col in display_cols if col in cascade_df.columns]
        
        st.dataframe(cascade_df.head(10)[display_cols_exist])
        
        st.info("A high 'cascade_score' indicates that this flight's arrival delay significantly contributed to the delay of the next flight operated by the same aircraft.")
    else:
        st.error("Cascade analysis data not found. Please run `core/cascade.py` first.")
