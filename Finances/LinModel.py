import pandas as pd
import matplotlib.pyplot as plt
from TimeSeriesForecasting.LinearModel import LinearModel

data = pd.read_csv('All_data.csv',parse_dates=True)
data.set_index("Date",inplace=True)

N = 10

#Get N last datas
start_data = data.index.values[data.shape[0]-N-1]
last_data = data.index.values[data.shape[0]-1]
#Generate data from last date to date "12/1/1995 (month day year) with steps = 1 month
forecast_data = pd.date_range(start_data,last_data, freq='D')[1:]

#n is steps to forecast
n = len(forecast_data)

#Create an object of rolling window
lin = LinearModel(data['V'].values)

f = lin.forecast(n)

data['Forecast'] = 0
data.loc[-n:,'Forecast'] = f

data['V'][-n:].plot(legend='V')
data['Forecast'][-n:].plot(legend='Forecast', marker='o',color='red')

plt.grid(True)
plt.show()