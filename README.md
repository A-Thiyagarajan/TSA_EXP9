### Name: Thiyagarajan A
### Register No: 212222240110
### Date:
# EX.NO.09        A  project on Time series analysis on Google stock prediction using the ARIMA model


### AIM:
The aim of this project is to forecast Google stock prices using the ARIMA model and evaluate its accuracy through visualization and statistical metrics.
### ALGORITHM:
1. Load the stock dataset from a CSV file and set the 'Date' column as the index.
2. Visualize the time series and check for stationarity using ACF, PACF plots, and the ADF test.
3. Apply differencing to transform the series into a stationary one if needed.
4. Determine ARIMA parameters (p, d, q) using ACF/PACF plots or the auto_arima function.
5. Fit the ARIMA model to the stationary stock data with the identified parameters.
6. Make predictions for future stock prices using the fitted model.
7. Evaluate and visualize predictions against actual values using metrics like MAE and RMSE, and plot for comparison.
### PROGRAM:
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('Google_Stock_Price_Train.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.set_index('Date', inplace=True)

# Filter data from 2010 onward
data = data[data.index >= '2010-01-01']

# Convert 'Close' column to numeric and remove missing values
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Close'].fillna(method='ffill', inplace=True)

# Plot the Close price to inspect for trends
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Google Close Price')
plt.title('Time Series of Google Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Check stationarity with ADF test
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, apply differencing
data['Close_diff'] = data['Close'].diff().dropna()
result_diff = adfuller(data['Close_diff'].dropna())
print('Differenced ADF Statistic:', result_diff[0])
print('Differenced p-value:', result_diff[1])

# Plot ACF and PACF for differenced data
plot_acf(data['Close_diff'].dropna())
plt.title('ACF of Differenced Close Price')
plt.show()
plot_pacf(data['Close_diff'].dropna())
plt.title('PACF of Differenced Close Price')
plt.show()

# Plot Differenced Representation
plt.figure(figsize=(10, 5))
plt.plot(data['Close_diff'], label='Differenced Close Price', color='red')
plt.title('Differenced Representation of Google Close Price')
plt.xlabel('Date')
plt.ylabel('Differenced Close Price')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.legend()
plt.show()

# Use auto_arima to find the optimal (p, d, q) parameters
stepwise_model = auto_arima(data['Close'], start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=False, trace=True)
p, d, q = stepwise_model.order
print(stepwise_model.summary())

# Fit the ARIMA model using the optimal parameters
model = sm.tsa.ARIMA(data['Close'], order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast the next 30 days
forecast = fitted_model.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('ARIMA Forecast of Google Stock Price')
plt.legend()
plt.show()

# Evaluate the model with MAE and RMSE
predictions = fitted_model.predict(start=0, end=len(data['Close']) - 1)
mae = mean_absolute_error(data['Close'], predictions)
rmse = np.sqrt(mean_squared_error(data['Close'], predictions))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/c67e2852-b5bc-4389-ba86-e455052a6355)

![image](https://github.com/user-attachments/assets/656afb12-b16a-4996-afcb-2e84f0fa8123)

![image](https://github.com/user-attachments/assets/f2d12e67-d6ac-4dfd-8e36-a79147f45ea6)

![image](https://github.com/user-attachments/assets/f01f4528-e10f-4129-b7d4-ff1f662421cf)

![image](https://github.com/user-attachments/assets/4419cf72-9015-4e59-b702-59e3c2eb6121)

![image](https://github.com/user-attachments/assets/999d375f-22f0-404f-b05d-47c1cca54de0)


![image](https://github.com/user-attachments/assets/7e60300e-3ec5-40b8-b789-dfd0f6de9cad)

![image](https://github.com/user-attachments/assets/7fd91b71-d59e-43b9-bd3b-036f4744d711)

![image](https://github.com/user-attachments/assets/b135129a-eb50-4326-b899-a7d43ea8d476)


### RESULT:
Thus the Time series analysis on Google stock prediction using the ARIMA model completed successfully.
