from collections import Counter
import numpy as np


class MyKNN:
    """Ручной knn"""

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", eps: float = 1e-6):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.eps = eps

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, X, metric: str = "euclidean"):
        if metric == "euclidean":
            return np.sqrt(np.sum((X - self.X_train) ** 2, axis=1))
        elif metric == "manhattan":
            return np.sum(np.abs(X - self.X_train), axis=1)

    def predict_proba(self, X, metric: str = "euclidean"):
        distances = self.distance(X, metric="euclidean")
        k_indices = np.argsort(distances)[:self.n_neighbors]

        if self.weights == "distance":
            k_distances = distances[k_indices]
            k_labels = [self.y_train[x] for x in k_indices]

            weights = 1 / (k_distances + self.eps)
            class_weights = {}
            for label, weight in zip(k_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + weight

            return max(class_weights, key=class_weights.get)

        k_nearest_labels = [self.y_train[x] for x in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

    def predict(self, X, metric: str = "euclidean"):
        X = np.array(X)
        prediction = [self.predict_proba(x, metric) for x in X]
        return np.array(prediction)
