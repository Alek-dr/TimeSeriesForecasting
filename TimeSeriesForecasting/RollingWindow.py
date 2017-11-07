from TimeSeriesForecasting.PredictModel import PredictModel
from numpy import power, concatenate, array, asarray, round, zeros, dot, rint


class RollingWindow(PredictModel):

    #Little modification of simple exp smoothing

    def __init__(self, ts):
        PredictModel.__init__(self, ts)
        self.pred_ts = self.ts

    def expl_smoothing(self, weights,s):
        ts_ = self.ts[-len(weights) + 1:][::-1]
        f = dot(ts_,weights[1:])
        p = zeros(1)
        p[0] = weights[0]*s +f
        return p

    def get_weight(self,alpha,iteration,last_weights):
        weights = zeros(iteration+2)
        weights[0] = power((1-alpha),iteration+1)
        weights[2:] = last_weights[1:]
        weights[1] = 1-weights.sum()
        return weights

    def forecast(self,alpha,n=1):
        weights = zeros(2)
        weights[0] = 1-alpha
        weights[1] = alpha
        #Initial series estimation
        l = int(rint(len(self.ts)*0.75))
        s = sum(self.ts[l:])/(len(self.ts)-l)
        for i in range(n):
            f = self.expl_smoothing(weights,s)
            weights = self.get_weight(alpha=alpha,iteration=i+1,last_weights=weights)
            self.pred_ts = concatenate((self.pred_ts, f))
        prediction = self.pred_ts[-n:]
        self.pred_ts = self.ts
        return prediction


