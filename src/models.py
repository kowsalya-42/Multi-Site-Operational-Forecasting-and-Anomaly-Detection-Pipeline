import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def train_xgboost_regressor(X_train, y_train, random_state=42):
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mape
