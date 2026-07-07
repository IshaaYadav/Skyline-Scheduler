# ✈️ Skyline Scheduler

**Live app:** [skylinescheduler.streamlit.app](https://skylinescheduler.streamlit.app/)

Skyline Scheduler turns raw airport flight logs into actionable scheduling insights. It cleans messy Excel export data, computes hourly traffic and delay KPIs, models how delays cascade from one flight to the next on the same aircraft, and trains a model to predict departure delays for a hypothetical flight — all wrapped in an interactive Streamlit dashboard with a natural-language query box.

---

## Features

- **💬 NLP Query** — Ask questions in plain English (e.g. *"What is the best time to fly to Delhi on a Saturday?"*) and get back the busiest hours, best hours, or the most disruptive flights, filtered by destination and/or day of week.
- **📊 Airport Overview** — Hourly breakdown of total flight volume alongside average departure and arrival delays, with an interactive Plotly chart.
- **🔮 Delay Prediction** — A "what-if" tool: pick an hour, day of week, and destination, and a trained RandomForest model predicts the expected departure delay in minutes.
- **🔗 Cascade Analysis** — Surfaces the flights whose delays had the biggest knock-on effect on the next flight flown by the same aircraft (tight turnaround + late arrival = propagated delay).

## How it works

The project is a linear data pipeline (`core/`) that feeds a Streamlit UI (`app/`):

```
data/Flight_Data.xlsx
        │
        ▼
core/load.py          →  outputs/01_loaded_data_from_excel.csv
        │  (finds header row, merges multi-sheet exports, forward-fills flight numbers)
        ▼
core/preprocess.py    →  outputs/02_processed_data.csv
        │  (parses times, builds timestamps, computes departure/arrival delay in minutes,
        │   extracts scheduled_hour, day_of_week, airport codes)
        ▼
core/kpis.py           →  outputs/03_hourly_kpis.csv
core/visualize.py      →  outputs/plots/*.html
core/delay_model.py    →  outputs/models/*.joblib (RandomForestRegressor + OneHotEncoder)
core/cascade.py        →  outputs/05_cascade_analysis.csv
        │
        ▼
app/ui.py (Streamlit)  ←  app/nlp.py (lightweight rule-based query parser)
```

Each `core/*.py` module can be run standalone (`python core/load.py`, etc.) and reads/writes CSVs under `outputs/` so the pipeline can be re-run step by step, or you can just use the pre-generated `outputs/` files already included in this repo to launch the app immediately.

## Tech stack

| Layer            | Tools |
|------------------|-------|
| Data processing  | pandas, numpy |
| Machine learning | scikit-learn (RandomForestRegressor, OneHotEncoder) |
| NLP              | sentence-transformers, faiss-cpu (for future embedding-based query matching) |
| Visualization    | Plotly, Matplotlib, Seaborn |
| Web app          | Streamlit |
| Config/validation| PyYAML, Pydantic |

## Project structure

```
Skyline-Scheduler/
├── app/
│   ├── server.py         # Launches the Streamlit app
│   ├── ui.py             # Streamlit dashboard (all 4 pages)
│   └── nlp.py            # Rule-based query parser for the NLP Query page
├── core/
│   ├── load.py           # Reads & merges raw Excel flight logs
│   ├── preprocess.py     # Cleans data, engineers delay/time features
│   ├── kpis.py            # Hourly traffic & delay aggregations
│   ├── cascade.py        # Knock-on delay analysis per aircraft
│   ├── delay_model.py    # Trains the delay prediction model
│   └── visualize.py      # Generates standalone Plotly HTML charts
├── data/
│   └── Flight_Data.xlsx  # Source flight log data
├── outputs/               # Generated CSVs, trained model artifacts, plots
├── requirements.txt
└── LICENSE
```

## Getting started

### 1. Clone and install dependencies

```bash
git clone https://github.com/IshaaYadav/Skyline-Scheduler.git
cd Skyline-Scheduler
pip install -r requirements.txt
```

### 2. (Optional) Rebuild the data pipeline from scratch

The repo already ships with pre-generated files in `outputs/`, so this step is optional. To regenerate everything from `data/Flight_Data.xlsx`:

```bash
python core/load.py
python core/preprocess.py
python core/kpis.py
python core/cascade.py
python core/delay_model.py
python core/visualize.py
```

### 3. Launch the dashboard

```bash
python app/server.py
# or directly:
streamlit run app/ui.py
```

The app will open at `http://localhost:8501`.

## Data

`data/Flight_Data.xlsx` contains raw scheduled/actual departure and arrival records (flight number, aircraft, origin/destination, STD/ATD/STA/ATA) for flights out of Mumbai (BOM). The loader is written to tolerate the quirks of these exports — inconsistent header rows, merged cells, and forward-filled flight numbers — so it should generalize to similarly structured exports from other airports.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
