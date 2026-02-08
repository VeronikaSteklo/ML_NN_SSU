import numpy as np


class MyLinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class MyPolynomialRegression(MyLinearRegression):
    def __init__(self, degree=2, lr=0.01, n_iters=1000):
        self.degree = degree
        super().__init__(lr, n_iters)

    def _transform(self, X):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X, y):
        X_transformed = self._transform(X)
        self.mean = np.mean(X_transformed, axis=0)
        self.std = np.std(X_transformed, axis=0)
        self.std[self.std == 0] = 1

        X_scaled = (X_transformed - self.mean) / self.std
        super().fit(X_scaled, y)

    def predict(self, X):
        X_transformed = self._transform(X)
        X_scaled = (X_transformed - self.mean) / self.std
        return super().predict(X_scaled)
