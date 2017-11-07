import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('All_data.csv',parse_dates=True)
data.set_index("Date",inplace=True)

b = data['B'].values
b /= 100
data['B'] = b

data['V'].plot(legend=True)
data['B'].plot(legend=True)
data['U'].plot(legend=True)
data['G'].plot(legend=True)
plt.grid(True)
plt.show()


