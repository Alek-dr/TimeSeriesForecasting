from time_series_forecasting.predict_model import PredictModel
from numpy import concatenate, zeros, flipud, fliplr, dot, arange, transpose
from scipy.linalg import hankel, inv

class LinearModel(PredictModel):

    #Simple autotegressive model

    def __init__(self, ts, n=None):
        PredictModel.__init__(self, ts)
        self.pred_ts = self.ts
        if n==None:
            n = len(ts)//2
            self.n = n
            self.l = len(ts)-n+1
            self.observ_matr = self.get_observ_matr(ts,n)
        else:
            self.n = n
            self.l = len(ts) - n + 1
            self.observ_matr = self.get_observ_matr(ts, n)

    def get_observ_matr(self, ts, n):
        p1 = hankel(ts[::-1])
        p2 = flipud(hankel(ts))
        r = arange(p2.shape[0])
        p2[r, r] = 0
        p2 = fliplr(p2)
        res = p1+p2
        t = len(ts)
        l = t-n+1
        return res[:l,:n]

    def forecast_one_step(self):
        y = self.ts[-self.l:]
        obs_t = transpose(self.observ_matr)
        a = inv(dot(obs_t,self.observ_matr))
        b = dot(a,obs_t)
        W = dot(b,y)
        y_ = transpose(y[-self.n:])
        f = dot(W,y_)
        return f

    def forecast_one_step(self, obs_matr, ts):
        l = len(ts) - self.n + 1
        y = ts[-l:]
        obs_t = transpose(obs_matr)
        a = inv(dot(obs_t,obs_matr))
        b = dot(a,obs_t)
        W = dot(b,y)
        y_ = transpose(y[-self.n:])
        f = dot(W,y_)
        p = zeros(1)
        p[0] = f
        return p

    def forecast(self, n):
        obs_matr = self.observ_matr
        for i in range(n):
            f = self.forecast_one_step(obs_matr,self.pred_ts)
            self.pred_ts = concatenate((self.pred_ts, f))
            obs_matr = self.get_observ_matr(self.pred_ts,n=self.n)
        prediction = self.pred_ts[-n:]
        self.pred_ts = self.ts
        return prediction








