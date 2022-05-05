import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
        '多项式模型': PolynomialModel
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
