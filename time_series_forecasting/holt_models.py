from time_series_forecasting.predict_model import PredictModel
from numpy import concatenate, round, zeros


class HoltModel(PredictModel):

    def __init__(self, ts, alpha,beta):
        PredictModel.__init__(self, ts)
        self.pred_ts = self.ts
        self.alpha = alpha
        self.beta = beta

    def get_exp_smooth(self):
        a = []
        b = []
        a.append(self.ts[0])
        b.append(self.ts[1]-self.ts[0])
        a = [round(elem,2) for elem in a]
        b = [round(elem, 2) for elem in b]
        for i in range(0,len(self.ts)):
            a_ = round(self.alpha*self.ts[i]+(1-self.alpha)*(a[i]+b[i]),2)
            b_ = round(self.beta*(a_-a[i])+(1-self.beta)*b[i],2)
            a.append(a_)
            b.append(b_)
        a = [round(elem, 2) for elem in a]
        b = [round(elem, 2) for elem in b]
        return a,b

    def forecast(self, n):
        a, b = self.get_exp_smooth()
        for i in range(n):
            y = a[-1] + b[-1]*(i+1)
            p = zeros(1)
            p[0] = y
            self.pred_ts = concatenate((self.pred_ts, p))
        prediction = self.pred_ts[-n:]
        return prediction
