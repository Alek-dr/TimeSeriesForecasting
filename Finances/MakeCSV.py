import pandas as pd

V = pd.read_excel('V Daily (2011 01 04 - 2017 10 31).xlsx')
V.dropna(inplace=True)
V.drop(V.index[[0]], inplace=True)
V.columns = ['Date','V']
V.reset_index(drop=True, inplace=True)
V.set_index("Date",inplace=True)

V.to_csv('V.csv',index=True)
