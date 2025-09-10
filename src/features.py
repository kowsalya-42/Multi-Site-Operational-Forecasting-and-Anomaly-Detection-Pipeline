def add_time_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df

def add_lag_features(df, group_col='site_id', lag_cols=['units_produced', 'power_kwh'], lags=[1]):
    df = df.sort_values([group_col, 'date'])
    for col in lag_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df

def add_rolling_features(df, group_col='site_id', roll_cols=['units_produced', 'power_kwh'], window=7):
    df = df.sort_values([group_col, 'date'])
    for col in roll_cols:
        df[f"{col}_roll{window}"] = df.groupby(group_col)[col].rolling(window).mean().reset_index(0, drop=True)
    return df
