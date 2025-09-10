import typer
import pandas as pd
import os
from src.loader import load_operations_data, load_site_meta, preprocess_operations
from src.features import add_time_features, add_lag_features, add_rolling_features
from src.anomaly import calculate_rolling_stats, detect_zscore_anomalies
import joblib

app = typer.Typer()

DATA_PATH = r'D:\kowsi\project_works\Advanced_Forecasting_Anomaly_Detection\logic_leap_horizon_datasets'
OUTPUT_DIR = 'outputs'

EXPECTED_REGION_COLS = ['region_East', 'region_North', 'region_South', 'region_West']

@app.command()
def forecast(
    site: str = typer.Option(..., help="Site ID"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
):
    """
    Generate units and power forecasts for a site and date range.
    Saves outputs as forecast_units.csv and forecast_power.csv.
    """
    # Load datasets
    ops = load_operations_data(os.path.join(DATA_PATH, 'operations_daily_365d.csv'))
    meta = load_site_meta(os.path.join(DATA_PATH, 'site_meta.csv'))

    # Preprocess
    df = preprocess_operations(ops)
    df = df.merge(meta, on='site_id', how='left')

    # Add features
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Filter site and date range
    df_site = df[(df['site_id'] == site) & (df['date'] >= start) & (df['date'] <= end)].copy()
    if df_site.empty:
        typer.echo(f"No data found for site {site} between {start} and {end}.")
        raise typer.Exit()

    # One-hot encode region
    df_site = pd.get_dummies(df_site, columns=['region'])
    for col in EXPECTED_REGION_COLS:
        if col not in df_site.columns:
            df_site[col] = 0

    # Features list
    features = [
        'units_produced_lag1', 'power_kwh_lag1', 'units_produced_roll7', 'power_kwh_roll7',
        'day_of_week', 'month', 'week_of_year', 'commissioned_year', 'shift_hours_per_day'
    ] + EXPECTED_REGION_COLS

    X = df_site[features]

    # Load models
    units_model = joblib.load('models/xgb_units_model.pkl')
    power_model = joblib.load('models/xgb_power_model.pkl')

    # Forecast
    df_site['forecast_units'] = units_model.predict(X)
    df_site['forecast_power'] = power_model.predict(X)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save forecasts separately
    units_file = os.path.join(OUTPUT_DIR, 'forecast_units.csv')
    power_file = os.path.join(OUTPUT_DIR, 'forecast_power.csv')

    df_site[['site_id', 'date', 'forecast_units']].to_csv(units_file, index=False)
    df_site[['site_id', 'date', 'forecast_power']].to_csv(power_file, index=False)

    typer.echo(f"Forecasts saved to {OUTPUT_DIR}/forecast_units.csv and {OUTPUT_DIR}/forecast_power.csv")


@app.command()
def alert(
    site: str = typer.Option(..., help="Site ID"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
):
    """
    Generate downtime anomaly alerts for a site and date range.
    Saves output as alerts.csv.
    """
    ops = load_operations_data(os.path.join(DATA_PATH, 'operations_daily_365d.csv'))
    df = ops[(ops['site_id'] == site) & (ops['date'] >= start) & (ops['date'] <= end)].copy()
    df['date'] = pd.to_datetime(df['date'])

    df = calculate_rolling_stats(df)
    df = detect_zscore_anomalies(df)

    alerts = df[df['anomaly']][['site_id', 'date', 'downtime_minutes', 'downtime_zscore']]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    alerts_file = os.path.join(OUTPUT_DIR, 'alerts.csv')
    alerts.to_csv(alerts_file, index=False)

    typer.echo(f"Alerts saved to {OUTPUT_DIR}/alerts.csv")


if __name__ == "__main__":
    app()
