import pandas as pd
import numpy as np
import copy
import random as rnd

class Data:

    def __array2oneHot__(self, H, set):
        Y = np.zeros(set).ravel()
        H = np.array(H).astype(int)
        for n in H:
            Y[n - 1] = 1
        return Y

    def __init__(self, file, n, set, col):
        self.file = file
        self.n = n
        self.set = set
        data = pd.read_csv(file)
        data = pd.DataFrame(data)
        data = data[col]
        #data = data.drop(data.columns[[0, 1]], axis=1)
        self.count = len(data.axes[0])
        data = data.values
        data = np.array(data).ravel()
        self.X = np.array(np.zeros((self.count, n)))
        j = 0
        for i in range(self.count):
            for z in range(n):
                self.X[i][z] = data[j]
                j = j + 1

    X = []
    count = 0
    curr_count = 0

    def make_sum_batches(self, steps):
        sum_seq = []
        batchX = []
        batchY = []
        for i in range(self.X.__len__()):
            sum_seq.append(np.sum(self.X[i]))
        z = 0
        n = (sum_seq.__len__()-1)//2
        X = np.zeros((n, steps))
        Y = np.zeros((n, steps))
        for i in range(n):
            x = []
            y = []
            for j in range(steps):
                # X[i,j] = np.array(sum_seq[i+z])
                # Y[i,j] = np.array(sum_seq[i+z+1])
                _x = []
                _y = []
                _x.append(np.array(sum_seq[i+z]))
                x.append(_x)
                _y.append(np.array(sum_seq[i+z+1]))
                y.append(_y)
                z+=1
            batchX.append(x)
            batchY.append(y)
            z = 0
        return np.array(batchX), np.array(batchY), sum_seq

    def make_batches(self, steps, one_hot=True, norm=True):
        batchX = []
        batchY = []
        for j in range(self.X.__len__() - steps):
            x = copy.copy(self.X[j:j + steps])
            y = copy.copy(self.X[j + 1:j + 1 + steps])
            if one_hot:
                _x = np.array(np.zeros((steps, self.set)))
                _y = np.array(np.zeros((steps, self.set)))
                for i in range(steps):
                    _x[i] = self.__array2oneHot__(x[i], self.set)
                    _y[i] = self.__array2oneHot__(y[i], self.set)
                batchX.append(_x)
                batchY.append(_y)
            else:
                if norm:
                    x[::1] /= self.set
                    y[::1] /= self.set
                batchX.append(x)
                batchY.append(y)
        return np.array(batchX), np.array(batchY)

    def next_batch(self, size, steps, one_hot=True, norm=True):
        batchX = []
        batchY = []
        for _ in range(size):
            rnd.seed(3)
            start = rnd.randint(0, self.X.__len__() - steps -1)
            x = copy.copy(self.X[start:start+steps])
            y = copy.copy(self.X[start+1:start+1 + steps])
            if one_hot:
                _x = np.array(np.zeros((steps, self.set)))
                _y = np.array(np.zeros((steps, self.set)))
                for i in range(steps):
                    _x[i] = self.__array2oneHot__(x[i],self.set)
                    _y[i] = self.__array2oneHot__(y[i], self.set)
                batchX.append(_x)
                batchY.append(_y)
            else:
                if norm:
                    x[::1] /= self.set
                    y[::1] /= self.set
                batchX.append(x)
                batchY.append(y)
        return np.array(batchX), np.array(batchY)

    def get_last(self, steps, one_hot=True, norm=True):
        batchX = []
        start = self.X.__len__() - steps
        x = copy.copy(self.X[start:])
        if one_hot:
            _x = np.array(np.zeros((steps, self.set)))
            _y = np.array(np.zeros((steps, self.set)))
            for i in range(steps):
                _x[i] = self.__array2oneHot__(x[i],self.set)
                #_y[i] = self.__array2oneHot__(y[i], self.set)
            batchX.append(_x)
        else:
            if norm:
                x[::1] /= self.set
            batchX.append(x)
        return np.array(batchX)

