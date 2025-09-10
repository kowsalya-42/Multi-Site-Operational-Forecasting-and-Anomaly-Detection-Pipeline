def calculate_rolling_stats(df, col='downtime_minutes', window=7, group_col='site_id'):
    df = df.sort_values([group_col, 'date'])
    df[f'{col}_roll_mean'] = df.groupby(group_col)[col].transform(lambda x: x.rolling(window, 1).mean())
    df[f'{col}_roll_std'] = df.groupby(group_col)[col].transform(lambda x: x.rolling(window, 1).std())
    return df

def detect_zscore_anomalies(df, col='downtime_minutes', zscore_col='downtime_zscore', threshold=3):
    df[zscore_col] = (df[col] - df[f'{col}_roll_mean']) / df[f'{col}_roll_std']
    df['anomaly'] = df[zscore_col].abs() > threshold
    return df
