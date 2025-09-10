# Advanced Forecasting and Anomaly Detection Pipeline

This project implements a robust forecasting and alerting pipeline for multi-site manufacturing operations.

## Project Structure

/advanced_forecasting
│
├── app/ # CLI application entrypoint
├── src/ # Source modules (data loading, features, models, anomaly)
├── models/ # Pretrained ML models (XGBoost)
├── logic_horizon_datasets # Input datasets
├── notebooks/ # Exploratory Data Analysis and model development notebooks
├── outputs/ # Generated forecasts and anomaly alerts CSVs
├── requirements.txt # Python dependencies
└── README.md # This file

## Setup
. **Clone the repository**

git clone <repo_url>
cd advanced_forecasting

text

2. **Create and activate virtual environment**

python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data files are in `logic_leap_horizon_datasets/` directory.**

## Usage

### CLI Application

Run forecasts:
```bash
python app/main.py forecast --site S1 --start 2025-01-01 --end 2025-01-10 --target units_produced
```

Get anomalies:
```bash
python app/main.py anomalies --site S1
```

### Generate Outputs

Run the script to generate CSV outputs:
```bash
python generate_outputs.py
```

This will create:
- `outputs/forecast_units.csv`
- `outputs/forecast_power.csv`
- `outputs/alerts.csv`

### Notebooks

Explore the notebooks in `notebooks/` for EDA, feature engineering, modeling, and anomaly detection.

## Project Structure

- `src/`: Core modules (loader, features, models, anomaly)
- `app/`: CLI application
- `notebooks/`: Jupyter notebooks
- `outputs/`: Generated CSV files
- `requirements.txt`: Dependencies
- `README.md`: This file

## Methodology

- Data loading and merging from multiple temporal datasets
- Feature engineering: temporal, lag, rolling features
- Forecasting: Prophet and XGBoost models
- Anomaly detection: Z-score based on downtime
- Evaluation: MAE, MAPE

## Executive Brief

See `executive_brief.md` for business insights.

python -m app.app forecast --site S1 --start 2025-09-11 --end 2025-09-30
