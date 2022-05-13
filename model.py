import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math


def cal_metrics(label, pred):
    metrics = {
        'MSE': mean_squared_error(label, pred),
        'R2': r2_score(label, pred)
    }
    return metrics


class MetaModel:
    def __init__(self) -> None:
        pass

    def pred(self, X: np.ndarray):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
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
        try:
            assert self.model != None
            assert self.power != None
            assert X.shape[1] == len(self.power)

            _X = self.preprocess(X)
            preditions = self.model.predict(_X)
            return preditions
        except Exception as e:
            return e

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            assert self.power != None
            assert X.shape[1] == len(self.power)

            self.model = LinearRegression(fit_intercept=False)
            _X = self.preprocess(X)
            self.model.fit(_X, y)
        except Exception as e:
            return e

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


class GreyModel11(MetaModel):
    def __init__(self) -> None:
        self.a = None
        self.b = None

    def pred(self, X: np.ndarray):
        const = self.b / self.a
        X0 = X
        F = [X0[0]]
        k = 1
        while k < len(X0) + 10:
            F.append((X0[0] - const) * math.exp(-self.a * k) + const)
            k += 1

        # get the predicted sequence

        x_hat = [X0[0]]
        g = 1
        while g < len(X0) + 10:
            print(g)
            x_hat.append(F[g] - F[g - 1])
            g += 1
        X0 = np.array(X0)
        x_hat = np.array(x_hat)
        return x_hat

    def fit(self, X: np.ndarray) -> bool:
        X0 = X.squeeze()
        X1 = np.cumsum(X0)
        M = (X1[1:]+X1[:-1])/2
        # least square method
        Y = np.mat(X0[1:]).reshape(-1, 1)
        B = np.mat(-M).reshape(-1, 1)
        B = np.hstack((B, np.ones((len(B), 1))))

        # parameters
        beta = np.linalg.inv(B.T.dot(B)).dot(B.T).dot(Y)
        self.a, self.b = beta

        # predict model


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


def RegisteredModel(key):
    models_dict = {
        '多项式模型': PolynomialModel,
        '单元灰色模型': GreyModel11,
    }
    return models_dict[key]


if __name__ == '__main__':
    model = PolynomialModel([3, 2, 3, 3, 3, 3])
    data, y = gen_data()
    model.fit(data, y)
    _y = model.pred(data)
    print(mean_squared_error(_y, y))
    model.save("model.pkl")

    model = PolynomialModel()
    model.load("model.pkl")
    data, y = gen_data()
    _y = model.pred(data)
    print(mean_squared_error(_y, y))
