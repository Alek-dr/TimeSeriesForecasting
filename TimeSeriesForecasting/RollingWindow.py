from TimeSeriesForecasting.PredictModel import PredictModel
from numpy import power, concatenate, array, asarray, round, zeros

class RollingWindow(PredictModel):

    def __init__(self, ts):
        PredictModel.__init__(self, ts)
        self.pred_ts = self.ts

    def expl_smoothing(self, alpha, span=None):
        f = 0
        if span==None:
            span=len(self.pred_ts)
        total = len(self.pred_ts)
        for i, j in zip(range(span),reversed(range(total))):
            f+=alpha*power((1-alpha),i)*self.pred_ts[j]
        f = round(f,5)
        p = zeros(1)
        p[0] = f
        return p

    def forecast_exp_smooth(self, n, alpha, span=None):
        for i in range(n):
            f = self.expl_smoothing(alpha,span)
            self.pred_ts = concatenate((self.pred_ts, f))
        prediction = self.pred_ts[-n:]
        self.pred_ts = self.ts
        return prediction


