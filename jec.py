# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Load your sales data
# Replace 'sales_data.csv' with your dataset's file path
data = pd.read_csv('sales_data.csv')
# Assuming you have a date column, parse it as a datetime object
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Check for stationarity
def test_stationarity(timeseries):
    # Calculate rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, label='Original')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std')
    plt.legend()
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] <= 0.05:
        print('Stationary (Reject Null Hypothesis)')
    else:
        print('Non-Stationary (Fail to Reject Null Hypothesis)')

# Make the time series stationary
data_diff = data['sales'].diff().dropna()
test_stationarity(data_diff)

# Plot ACF and PACF to determine p and q values for ARIMA
plot_acf(data_diff, lags=30)
plot_pacf(data_diff, lags=30)
plt.show()

# Fit ARIMA model
p = 1  # Replace with the appropriate lag value from PACF
d = 1  # Differencing order
q = 1  # Replace with the appropriate lag value from ACF

model = ARIMA(data['sales'], order=(p, d, q))
model_fit = model.fit(disp=0)

# Make predictions
forecast_period = 12  # Replace with the number of periods to forecast
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_period)

# Create a date range for future predictions
future_dates = pd.date_range(start=data.index[-1], periods=forecast_period + 1, closed='right')

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({'forecast': forecast}, index=future_dates[1:])

# Plot original data and forecast
plt.figure(figsize=(12, 6))
plt.plot(data['sales'], label='Original Data')
plt.plot(forecast_df, label='Forecast')
plt.legend()
plt.title('Sales Forecast')
plt.show()

# Evaluate the model (optional)
# Split your data into training and testing sets and use them to evaluate the model's accuracy.
# Calculate Mean Squared Error (MSE) and other relevant metrics.
# You may also want to perform cross-validation for a more robust evaluation.

# Save the model (optional)
# If you're satisfied with your ARIMA model, you can save it for future use:
# model_fit.save('sales_arima_model.pkl')

# Load the model (for future predictions)
# To load the model for future sales predictions without training it again:
# loaded_model = ARIMAResults.load('sales_arima_model.pkl')
