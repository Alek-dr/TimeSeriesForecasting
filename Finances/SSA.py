import pandas as pd
import matplotlib.pyplot as plt
from TimeSeriesForecasting.SSA import SSA

data = pd.read_csv('All_data.csv',parse_dates=True)
data.set_index("Date",inplace=True)

N = 6

#Get N last datas
start_data = data.index.values[data.shape[0]-N-1]
last_data = data.index.values[data.shape[0]-1]
#Generate data from last date to date "12/1/1995 (month day year) with steps = 1 month
forecast_data = pd.date_range('10/31/2017','11/6/2017', freq='D')[1:]

print(forecast_data)

#n is steps to forecast
n = len(forecast_data)

#Create an object of rolling window
val = data['V'].values['04/1/2011','12/31/2016']

# ssa1 = SSA(data['V'].values[:-N],L=16)
# comp = range(0,14)
# f1 = ssa1.forecast_by_component(N,comp)
#
# data['Forecast V'] = 0
# data.loc[-N:,'Forecast V'] = f1
#
# ssa2 = SSA(data['U'].values[:-N],L=16)
# comp = range(0,14)
# f2 = ssa2.forecast_by_component(N,comp)
#
# data['Forecast U'] = 0
# data.loc[-N:,'Forecast U'] = f2
#
# df = pd.DataFrame(columns=['Date','Forecast V','Forecast U'])
# df['Forecast V'] = f1
# df['Forecast U'] = f2
# df['Date'] = forecast_data
# df.set_index("Date",inplace=True)
# df = df.round(3)
# data['V'][-N:].plot(legend='V')
# data['Forecast V'][-N:].plot(legend='Forecast', marker='o',color='red')
#


# df['Forecast V'].plot()

# plt.grid(True)
# plt.show()

# df.to_csv('Forecast.csv',header=True,index=True)