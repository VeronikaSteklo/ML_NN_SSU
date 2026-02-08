import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone


class SimpleAveragingRegressor:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)


class MyBaggingRegressor:
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            model = clone(self.base_model)
            model.fit(X[indices], y[indices])
            self.models.append(model)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


class MyRandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=5, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feat_indices = []

    def fit(self, X, y):
        self.trees = []
        self.feat_indices = []
        n_samples, n_features = X.shape

        if self.max_features == 'sqrt':
            n_sub_features = int(np.sqrt(n_features))
        else:
            n_sub_features = n_features

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            features = np.random.choice(n_features, n_sub_features, replace=False)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_boot[:, features], y_boot)

            self.trees.append(tree)
            self.feat_indices.append(features)

    def predict(self, X):
        predictions = np.array([tree.predict(X[:, feats])
                                for tree, feats in zip(self.trees, self.feat_indices)])
        return np.mean(predictions, axis=0)


class MyStackingRegressor:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)

        meta_features = np.column_stack([m.predict(X) for m in self.base_models])
        self.meta_model.fit(meta_features, y)

    def predict(self, X):
        meta_features = np.column_stack([m.predict(X) for m in self.base_models])
        return self.meta_model.predict(meta_features)


class MySimpleBoosting:
    def __init__(self, n_estimators=10, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.models = []

    def fit(self, X, y):
        f_prev = np.zeros(len(y))

        for _ in range(self.n_estimators):
            residuals = y - f_prev

            model = DecisionTreeRegressor(max_depth=3)
            model.fit(X, residuals)

            f_prev += self.lr * model.predict(X)
            self.models.append(model)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for model in self.models:
            y_pred += self.lr * model.predict(X)
        return y_pred
