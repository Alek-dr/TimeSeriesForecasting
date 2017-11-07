import pandas as pd

#Resample data to each day, merje into one file

index_date = pd.date_range("01/4/2011","10/31/2017", freq='D')[1:]

B = pd.read_csv('csv/B.csv',parse_dates=[0])
B.set_index("Date",inplace=True)
B = B.resample('D', fill_method='ffill')

G = pd.read_csv('csv/G.csv',parse_dates=[0])
G.set_index("Date",inplace=True)
G = G.resample('D', fill_method='ffill')

U = pd.read_csv('csv/U.csv',parse_dates=[0])
U.set_index("Date",inplace=True)
U = U.resample('D', fill_method='ffill')

V = pd.read_csv('csv/V.csv',parse_dates=[0])
V.set_index("Date",inplace=True)
V = V.resample('D', fill_method='ffill')

frames = [B,G,U,V]
res = pd.concat(frames, axis=1, join_axes=[V.index])

res = res.round(3)

res.to_csv('All_data.csv',index=True,header=True)