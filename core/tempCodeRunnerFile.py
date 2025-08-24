import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

def train_delay_prediction_model(data_path: str, model_output_dir: str):
    """
    Trains a machine learning model to predict departure delays.

    Args:
        data_path: Path to the processed flight data CSV.
        model_output_dir: Directory to save the trained model and artifacts.
    """
    print("Starting model training process...")

    # --- 1. Load and Prepare Data ---
    df = pd.read_csv(data_path)

    # Drop rows with missing target or key features
    df.dropna(subset=['departure_delay', 'scheduled_hour', 'day_of_week', 'to_airport'], inplace=True)
    
    # For simplicity, we'll cap delays at a reasonable threshold (e.g., 4 hours) to handle extreme outliers
    df = df[df['departure_delay'] < 240]
    df = df[df['departure_delay'] >= 0] # Focus on predicting delays, not early departures

    # --- FIX: Add a check to ensure data exists after filtering ---
    if df.empty:
        print("\nError: No data available for model training after filtering.")
        print("This can happen if all flights in the dataset were early, had extreme delays, or had missing values.")
        print("Please check the contents of '02_processed_data.csv'.")
        return # Exit the function gracefully

    # --- 2. Feature Selection and Engineering ---
    features = ['scheduled_hour', 'day_of_week', 'to_airport']
    target = 'departure_delay'

    X = df[features]
    y = df[target]

    # --- 3. Preprocessing (One-Hot Encoding for Categorical Features) ---
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit and transform the categorical features
    X_encoded = encoder.fit_transform(X[['day_of_week', 'to_airport']])
    
    # Create a DataFrame with the encoded features
    encoded_cols = encoder.get_feature_names_out(['day_of_week', 'to_airport'])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_cols, index=X.index)

    # Combine numerical and encoded categorical features
    X_final = pd.concat([X[['scheduled_hour']], X_encoded_df], axis=1)

    # --- 4. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- 5. Model Training ---
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)

    # --- 6. Model Evaluation ---
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nModel Evaluation - Mean Absolute Error on Test Set: {mae:.2f} minutes")
    
    # --- 7. Save Model and Artifacts ---
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    model_path = os.path.join(model_output_dir, 'delay_predictor.joblib')
    encoder_path = os.path.join(model_output_dir, 'encoder.joblib')
    columns_path = os.path.join(model_output_dir, 'model_columns.joblib')

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(X_final.columns.tolist(), columns_path)

    print(f"Model saved to {model_path}")
    print(f"Encoder saved to {encoder_path}")
    print(f"Model columns saved to {columns_path}")


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, 'outputs')
    
    processed_data_path = os.path.join(output_path, '02_processed_data.csv')
    model_dir = os.path.join(output_path, 'models')

    if not os.path.exists(processed_data_path):
        print(f"Error: Input file not found at {processed_data_path}")
        print("Please run `core/preprocess.py` first.")
    else:
        train_delay_prediction_model(processed_data_path, model_dir)
