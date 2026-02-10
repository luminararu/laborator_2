import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)
consumption = pd.read_csv("household_hourly_dataset.csv")
power = np.array(consumption["Global_active_power"])
exog_columns = ["Global_reactive_power", "Voltage", "Global_intensity",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
exogenous = consumption[exog_columns]

split = int(power.shape[0] * 0.7)
n_forecast = len(power) - split
train_p = power[:split]
train_ex = exogenous.iloc[:split, :]
test_p = power[split:]

forecasted_exog_data = {}
for col in train_ex.columns:
    series = train_ex[col]
    exog_model = ARIMA(series, order=order, seasonal_order=seasonal_order)
    exog_fit = exog_model.fit()
    forecast = exog_fit.forecast(steps=n_forecast)
    forecasted_exog_data[col] = forecast

future_exog = pd.DataFrame(forecasted_exog_data)
model = ARIMA(endog=train_p, exog=train_ex, order=order, seasonal_order=seasonal_order)
model_fit = model.fit()
predictions = model_fit.forecast(steps=n_forecast, exog=future_exog)
x_test_indices = np.arange(split, split + n_forecast)

print(f"MSE: {np.sum((test_p[:100] - predictions[:100]) ** 2)}")

plt.title("ARMA Results")
plt.plot(x_test_indices[:100], test_p[:100], label="Ground Truth")
plt.plot(x_test_indices[:100], predictions[:100], label="Predictions", color="red", linestyle=":")
plt.show()
