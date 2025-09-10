import pandas as pd

def load_operations_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_site_meta(filepath):
    return pd.read_csv(filepath)

def preprocess_operations(df):
    # Remove zero production/power rows
    df_clean = df[(df['units_produced'] > 0) & (df['power_kwh'] > 0)].copy()
    return df_clean
