import streamlit as st
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nlp import parse_query

# --- Page Configuration ---
st.set_page_config(
    page_title="Skyline Scheduler",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #F0F2F6; /* Light Gray */
    }

    /* Sidebar container */
    section[data-testid="stSidebar"] {
        background-color: #0E1117 !important; /* Dark Sidebar */
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #D1D5DB !important; /* Light gray text */
    }

    /* Sidebar radio buttons */
    section[data-testid="stSidebar"] .stRadio > label {
        display: block;
        padding: 8px 12px;
        border-radius: 8px;
        transition: background-color 0.3s ease, color 0.3s ease;
    }


    /* Card-like containers */
    .st-emotion-cache-z5fcl4 {
        border-radius: 10px;
        padding: 25px !important;
        background-color: #FFFFFF;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .st-emotion-cache-z5fcl4:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }

    /* Main titles */
    h1 {
        color: #1E3A8A; /* Dark Blue */
        font-weight: bold;
    }

    /* Subheaders */
    h2, h3 {
        color: #3B82F6; /* Medium Blue */
    }

    /* Metric label styling */
    .st-emotion-cache-1g6go79 {
        color: #4B5563; /* Gray */
        font-size: 1rem;
    }

    /* Metric value styling */
    .st-emotion-cache-1wivapv {
        font-size: 2.5rem !important;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions (Caching) ---
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_resource
def load_model_artifacts(model_dir):
    model_path = os.path.join(model_dir, 'delay_predictor.joblib')
    encoder_path = os.path.join(model_dir, 'encoder.joblib')
    columns_path = os.path.join(model_dir, 'model_columns.joblib')
    if all(os.path.exists(p) for p in [model_path, encoder_path, columns_path]):
        return joblib.load(model_path), joblib.load(encoder_path), joblib.load(columns_path)
    return None, None, None

def predict_delay(model, encoder, model_cols, hour, day, airport):
    input_df = pd.DataFrame([[hour, day, airport]], columns=['scheduled_hour', 'day_of_week', 'to_airport'])
    encoded_features = encoder.transform(input_df[['day_of_week', 'to_airport']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['day_of_week', 'to_airport']))
    final_input = pd.concat([input_df[['scheduled_hour']], encoded_df], axis=1).reindex(columns=model_cols, fill_value=0)
    return model.predict(final_input)[0]

# --- Data Loading ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_PATH, 'models')
IMAGE_PATH = os.path.join(PROJECT_ROOT, 'data', 'SkylineScheduler.jpg')

kpi_df = load_data(os.path.join(OUTPUT_PATH, '03_hourly_kpis.csv'))
cascade_df = load_data(os.path.join(OUTPUT_PATH, '05_cascade_analysis.csv'))
processed_df = load_data(os.path.join(OUTPUT_PATH, '02_processed_data.csv'))
model, encoder, model_cols = load_model_artifacts(MODEL_DIR)

# --- Sidebar ---
with st.sidebar:
    page = st.radio("Navigation", ["NLP Query", "Airport Overview", "Delay Prediction", "Cascade Analysis"], label_visibility="hidden")
    st.markdown("---")

# --- Main App Header ---
col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, width=100)
with col2:
    st.title("Skyline Scheduler")

# --- Main App Pages ---
if page == "NLP Query":
    st.header("💬 Insight Finder")
    st.markdown("Ask your query for assistance.")
    with st.container():
        user_query = st.text_input("Your question:", placeholder="e.g., What is the best time to fly to Delhi on a Saturday?")
        if user_query:
            if processed_df is not None and cascade_df is not None:
                title, result = parse_query(user_query, processed_df, cascade_df)
                st.subheader(title)
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                else:
                    st.warning(result)
            else:
                st.error("Data files not found. Please run the core pipeline first.")

elif page == "Airport Overview":
    st.header("✈️ Airport Performance Overview")
    st.markdown("A high-level look at flight traffic and delays by hour at Mumbai Airport (BOM).")
    if kpi_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                st.subheader("Busiest Hours")
                busiest = kpi_df.sort_values('total_flights', ascending=False).head(5)
                st.dataframe(busiest[['hour', 'total_flights', 'average_departure_delay', 'average_arrival_delay']], use_container_width=True)
        with col2:
            with st.container():
                st.subheader("Best Hours (Lowest Delays)")
                kpi_df['combined_avg_delay'] = kpi_df[['average_departure_delay', 'average_arrival_delay']].mean(axis=1)
                best = kpi_df.sort_values('combined_avg_delay').head(5)
                st.dataframe(best[['hour', 'total_flights', 'combined_avg_delay']], use_container_width=True)
        st.markdown("---")
        with st.container():
            st.subheader("Visual Analysis")
            fig_delay = make_subplots(specs=[[{"secondary_y": True}]])
            fig_delay.add_trace(go.Bar(x=kpi_df['hour'], y=kpi_df['total_flights'], name='Total Flights', marker_color='#3B82F6'), secondary_y=False)
            fig_delay.add_trace(go.Scatter(x=kpi_df['hour'], y=kpi_df['average_departure_delay'], name='Avg. Departure Delay', mode='lines+markers', line=dict(color='#EF4444')), secondary_y=True)
            fig_delay.add_trace(go.Scatter(x=kpi_df['hour'], y=kpi_df['average_arrival_delay'], name='Avg. Arrival Delay', mode='lines+markers', line=dict(color='#10B981')), secondary_y=True)
            fig_delay.update_layout(title_text='<b>Average Delays vs. Flight Volume by Hour</b>', template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_delay, use_container_width=True)
    else:
        st.error("KPI data not found. Please run `core/kpis.py` first.")

elif page == "Delay Prediction":
    st.header("🔮 What-If: Delay Predictor")
    st.markdown("Tune the schedule for a hypothetical flight to see the predicted impact on its departure delay.")
    if model and processed_df is not None:
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                airport_map_df = processed_df[['to_full', 'to_airport']].drop_duplicates().dropna().sort_values('to_full')
                airport_map = pd.Series(airport_map_df.to_airport.values, index=airport_map_df.to_full).to_dict()
                c1, c2, c3 = st.columns(3)
                with c1:
                    hour = st.slider("Scheduled Hour (24H)", 0, 23, 10)
                with c2:
                    day = st.selectbox("Day of Week", processed_df['day_of_week'].dropna().unique())
                with c3:
                    selected_name = st.selectbox("Destination", options=airport_map.keys())
                if st.button("Predict Delay", use_container_width=True, type="primary"):
                    airport_code = airport_map[selected_name]
                    predicted_delay = predict_delay(model, encoder, model_cols, hour, day, airport_code)
                    st.session_state.predicted_delay = predicted_delay
            with col2:
                if 'predicted_delay' in st.session_state:
                    delay = st.session_state.predicted_delay
                    st.metric(label="Predicted Departure Delay", value=f"{delay:.1f} min")
                    if delay > 20:
                        st.error("High delay risk.")
                    elif delay > 10:
                        st.warning("Moderate delay risk.")
                    else:
                        st.success("Low delay risk.")
    else:
        st.error("Model or data not found. Please run `core/delay_model.py` and `core/preprocess.py` first.")

elif page == "Cascade Analysis":
    st.header("🔗 Cascade Effect Analysis")
    st.markdown("Identifying flights that cause the largest knock-on delays to subsequent flights.")
    if cascade_df is not None:
        with st.container():
            st.subheader("Top 10 Most Disruptive Flights")
            display_cols = ['flight_number', 'aircraft', 'to_airport', 'arrival_delay', 'turnaround_mins', 'cascade_score']
            display_cols_exist = [col for col in display_cols if col in cascade_df.columns]
            st.dataframe(cascade_df.head(10)[display_cols_exist], use_container_width=True)
            st.info("A high 'cascade_score' indicates that a flight's arrival delay significantly contributed to the delay of the next flight by the same aircraft.")
    else:
        st.error("Cascade analysis data not found. Please run `core/cascade.py` first.")
