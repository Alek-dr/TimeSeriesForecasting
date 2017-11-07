from sklearn.linear_model import ElasticNet
import pandas as pd
from numpy import arange, zeros, reshape
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

data = pd.read_csv('All_data.csv',parse_dates=[0])
data.set_index("Date",inplace=True)

n = 30

y = data['V'].values
y = reshape(y,(y.shape[0],1))
X = arange(0,len(y),1)
X = reshape(X,(X.shape[0],1))

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(X)

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_poly, y)

y_ = []
x_ = []
for x in X:
    x_.append(x)
    x = reshape(x,(1,1))
    x = poly_features.fit_transform(x)
    p = elastic_net.predict(x)
    y_.append(p)

for x in range(len(y),len(y)+n):
    x_.append(x)
    x = reshape(x,(1,1))
    x = poly_features.fit_transform(x)
    p = elastic_net.predict(x)
    y_.append(p)

plt.plot(X,y)
plt.plot(x_,y_)
plt.show()

