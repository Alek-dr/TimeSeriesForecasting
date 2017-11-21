import numpy as np
import pandas as pd
from numpy.linalg.linalg import dot, inv, eigvals, qr
from numpy import transpose
from scipy.linalg import hankel
from sklearn.metrics import mean_squared_error
from time_series_forecasting.predict_model import PredictModel

class SSA(PredictModel):

    def __init__(self, ts, L=3):
        PredictModel.__init__(self, ts)
        self.L = L
        self.pred_ts = ts
        self.K = len(self.ts) - L + 1
        self.X = self.observe_matrix(ts, self.K)
        self.V, self.R = self.get_v(self.X)
        self.tsh = 1e-2
        #V - Orthonormality system of vectors
        #R - counts of non-zero eigen values
        self.F = self.get_f(self.R)
        # F - solve of system equation

    def describ_ts(self,r=None):
        #return description of time-series
        #based on R components
        if r==None:
            r = self.R
        else:
            if r>self.R:
                r = self.R
        Z = self.get_z(r)
        t = self.ts_from_z(Z)
        return np.array(t)

    def ts_from_z(self, Z):
        row = Z.shape[0]
        col = Z.shape[1]
        t = []
        for i in range(row):
            sq = Z[0:i + 1, 0:i + 1]
            t.append(self.diag_sum(sq))
        for j in range(col - row):
            sq = Z[:, 1 + j:row + j + 1]
            t.append(self.diag_sum(sq))
        for _ in range(row - 1):
            sq = sq[1:, 1:]
            t.append(self.diag_sum(sq))
        return t

    def diag_sum(self, sq):
        sum = []
        j = len(sq) - 1
        for i in range(len(sq)):
            sum.append(sq[i, j])
            j -= 1
        sum = np.array(sum)
        return sum.sum() / len(sum)

    def observe_matrix(self, ts, K):
        h = hankel(ts)
        h = h[0:self.L, 0:K]
        return h

    def get_f(self, r):
        if r>self.R:
            r = self.R
        V_ = self.V[0:self.V.shape[0] - 1, 0:r]
        v = self.V[self.V.shape[0] - 1, 0:r]
        g = inv(dot(transpose(V_), V_))
        a = dot(v, g)
        P = dot(a, transpose(V_))
        return P

    def get_f_by_comp(self,V):
        V_ = V[0:V.shape[0] - 1, :]
        v = V[V.shape[0] - 1,:]
        g = inv(dot(transpose(V_), V_))
        a = dot(v, g)
        P = dot(a, transpose(V_))
        return P

    def get_v(self, X):
        C = dot((1 / (self.K)), dot(X, transpose(X)))
        V, _ = qr(C)
        eigV = eigvals(C)
        eigV = np.sort(eigV)[::-1]
        ind = np.where(eigV > 0)[0]
        r = len(ind)
        return V, r

    def get_z(self,r):
        V = self.V[:, 0:r]
        Vt = transpose(V)
        U = dot(Vt, self.X)
        Z = dot(V, U)

        return Z

    def describ_by_component(self,comp_ind):
        V = self.V[:, comp_ind]
        Vt = transpose(V)
        U = dot(Vt, self.X)
        Z = dot(V, U)
        t = self.ts_from_z(Z)
        return t

    def forecast(self, n):
        N = len(self.pred_ts)
        for k in range(n):
            q = self.pred_ts[N - self.L + 1 + k:N + k]
            q = np.reshape(q, (len(q), 1))
            f = dot(self.F, q)
            self.pred_ts = np.concatenate((self.pred_ts, f))
        return self.pred_ts[N:]

    def forecast_by_component(self,n,comp_ind):
        V = self.V[:, comp_ind]
        F = self.get_f_by_comp(V)
        N = len(self.pred_ts)
        for k in range(n):
            q = self.pred_ts[N - self.L + 1 + k:N + k]
            q = np.reshape(q, (len(q), 1))
            f = dot(F, q)
            self.pred_ts = np.concatenate((self.pred_ts, f))
        pred = self.pred_ts[N:]
        self.pred_ts = self.ts
        return pred

    def optimize(self, ts):
        mses = []
        _L = []
        _N = []
        L = self.L
        R = self.R
        stop = False
        for i in range(2, len(self.ts)):
            if stop:
                break
            for j in range(1, R):
                self.__init__(self.ts, i)
                try:
                    self.F = self.get_f(j)
                    pred_ts = self.forecast(len(ts))
                    mse = mean_squared_error(ts, pred_ts)
                    mses.append(mse)
                    _L.append(i)
                    _N.append(j)
                    if mse <= self.tsh:
                        stop=True
                        break
                except:
                    continue
        self.__init__(self.ts,L)
        mses = np.array(mses)
        _L = np.array(_L)
        _N = np.array(_N)
        data = {'MSE': mses, 'L': _L, 'R': _N}
        data = pd.DataFrame(data)
        return data

    def forecast_with_param(self, n, L, r):
        #r - nbmber of components
        self.__init__(self.ts, L)
        self.F = self.get_f(r)
        ts = self.forecast(n)
        return ts