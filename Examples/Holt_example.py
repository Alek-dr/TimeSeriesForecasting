import pandas as pd
import matplotlib.pyplot as plt
from time_series_forecasting.holt_models import HoltModel

#Dataset example from https://www.otexts.org/fpp/7/2

data = pd.read_csv('datasets/AirPassengers.csv', parse_dates=["Year"])
data.set_index("Year",inplace=True)

N = 6
#Get last date
last_data = data.index.values[data.shape[0]-1]
forecast_data = pd.date_range(last_data, periods=N, freq='Y')[1:]

#n is steps to forecast
n = len(forecast_data)
#Create an object of rolling window

holt = HoltModel(data['y'].values, 0.8, 0.2)
#Forecast by n steps
f = holt.forecast(n=n)

#Make data frame with forecast data
df = pd.DataFrame(columns=['Forecast'])
df['Forecast'] = f
#Index allows to plot as time series
df.index = forecast_data

#Plot result
df['Forecast'].iloc[:].plot()
data['y'].iloc[:].plot()

plt.grid(True)
plt.show()