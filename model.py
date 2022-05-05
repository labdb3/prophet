from pyexpat import model
from tkinter import Y
import numpy as np
import pandas as pd
from sklearn.exceptions import NonBLASDotWarning
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_squared_error
import time

from torch import ge

np.random.seed(int(time.time()))


class MetaModel:
    def __init__(self) -> None:
        pass

    def pred(self, X: np.ndarray):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def save(self, path: str) -> bool:
        pass

    def load(self, path: str) -> bool:
        pass


class PolynomialModel(MetaModel):
    def __init__(self, power: list = None) -> None:
        self.power = power
        self.model = None

    def pred(self, X: np.ndarray):
        _X = self.preprocess(X)
        preditions = self.model.predict(_X)
        return preditions

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = LinearRegression(fit_intercept = False)
        _X = self.preprocess(X)
        self.model.fit(_X, y)


    def save(self, path: str):
        pickle.dump([self.model, self.power], open(path, "wb"))


    def load(self, path: str):
        self.model, self.power = pickle.load(open(path, "rb"))


    def preprocess(self, X: np.ndarray):
        data = X.tolist()
        res = []
        for line in data:
            tmp = []
            for idx, item in enumerate(line):
                for i in range(self.power[idx]):
                    tmp.append(pow(item, i+1))

            res.append(tmp)

        return np.array(res)


def gen_data():
    data = np.random.random([5, 6])
    pow = [3, 2, 3, 3, 3, 3]
    y = np.zeros(5)
    for i in range(data.shape[1]):
        for j in range(pow[i]):
            y += np.power(data[:, i], j+1)

    return data, y[:, np.newaxis]


def gen_test_data():
    data = np.random.random([5, 6])
    pow = [3, 2, 3, 3, 3, 3]
    y = np.zeros(5)
    for i in range(data.shape[1]):
        for j in range(pow[i]):
            y += np.power(data[:, i], j+1)

    return data, y[:, np.newaxis]


if __name__ == '__main__':
    model = PolynomialModel([3,2,3,3,3,3])
    data, y = gen_data()
    model.fit(data,y)
    _y = model.pred(data)
    print(mean_squared_error(_y,y))
    model.save("model.pkl")

    model = PolynomialModel()
    model.load("model.pkl")
    data, y = gen_data()
    _y = model.pred(data)
    print(mean_squared_error(_y, y))
