import pandas as pd
import matplotlib.pyplot as plt
from time_series_forecasting.rolling_window import RollingWindow

#Dataset from
#http://video.udacity-data.com.s3.amazonaws.com/topher/2016/September/57e47a41_single-family-home-sales/single-family-home-sales.xlsx

data = pd.read_excel('datasets/single-family-home-sales.xlsx', parse_dates=[0])
data.set_index("Month",inplace=True)

N = 30
#Get last date
last_data = data.index.values[data.shape[0]-1]
#Generate data from last date to date "12/1/1995 (month day year) with steps = 1 month
forecast_data = pd.date_range(last_data, periods=N, freq='MS')[1:]

#n is steps to forecast
n = len(forecast_data)
#Create an object of rolling window
rw = RollingWindow(data['Home Sales'].values)
#Forecast by n steps
f = rw.forecast(alpha=0.3, n=n)

#Make data frame with forecast data
df = pd.DataFrame(columns=['Forecast'])
df['Forecast'] = f
#Index allows to plot as time series
df.index = forecast_data

#Plot result
df['Forecast'].iloc[:].plot()
data['Home Sales'].iloc[:].plot()

plt.grid(True)
plt.show()