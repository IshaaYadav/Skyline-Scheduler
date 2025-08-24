import os
import subprocess
import sys

def run_streamlit():
    """
    Finds and runs the ui.py script using Streamlit.
    """
    # Get the directory where this server.py script is located
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to ui.py
    ui_path = os.path.join(app_dir, "ui.py")

    if not os.path.exists(ui_path):
        print(f"Error: ui.py not found at {ui_path}")
        sys.exit(1)

    print(f"Launching Streamlit app from: {ui_path}")
    
    # Use subprocess to run the streamlit command
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path], check=True)
    except FileNotFoundError:
        print("Error: 'streamlit' command not found.")
        print("Please make sure Streamlit is installed correctly ('pip install streamlit').")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the Streamlit app: {e}")

if __name__ == "__main__":
    run_streamlit()
