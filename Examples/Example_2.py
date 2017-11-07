import pandas as pd
import matplotlib.pyplot as plt
from TimeSeriesForecasting.LinearModel import LinearModel

data = pd.read_excel('single-family-home-sales.xlsx', parse_dates=[0])
data.set_index("Month",inplace=True)

#Get last date
last_data = data.index.values[data.shape[0]-1]
#Generate data from last date to date "12/1/1995 (month day year) with steps = 1 month
forecast_data = pd.date_range(last_data,"12/1/1998", freq='MS')[1:]

#n is steps to forecast
n = len(forecast_data)
#Create an object of rolling window
lm = LinearModel(data['Home Sales'].values, n=20)
#Forecast by n steps
f = lm.forecast(n,3)

#Make data frame with forecast data
df = pd.DataFrame(columns=['Forecast'])
df['Forecast'] = f
#Index allows to plot as time series
df.index = forecast_data
df['Forecast'].iloc[:].plot()

#Show plot with existing data
data['Home Sales'].iloc[:].plot()
#Show forecast
data['Forecast'].iloc[:].plot()

plt.grid(True)
plt.show()