# Advanced Forecasting and Anomaly Detection Pipeline

This repository delivers a robust analytics pipeline for multi-site industrial operations. The pipeline forecasts daily production units and power consumption for each site and automatically detects operational downtime anomalies.

- **Automated 14-day forecasting** for both `units_produced` and `power_kwh` using advanced ML models (XGBoost) and baseline models for comparison.
- **Interpretable anomaly detection** using rolling z-score for early alerts on downtime.
- **CLI tool** for easy, flexible generation of forecasts and alerts by site and date range.
- **Reproducible and modular**: Clean code structure, notebooks for EDA/modeling, and clear documentation ensure ease of use and extension.
- **Business-ready outputs**: Forecast and alert CSVs ready for integration with operations.


## Setup
1. **Clone the repository**

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
- Forecasting: XGBoost models
- Anomaly detection: Z-score based on downtime
- Evaluation: MAE, MAPE

## Executive Brief

See `executive_brief.pdf` for business insights.
