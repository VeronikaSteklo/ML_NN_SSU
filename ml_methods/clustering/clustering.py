import numpy as np


class MyKMeans:
    def __init__(self, k=3, max_iters=100):
        self.labels_ = None
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels
        return self.labels_

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


class MyDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1: continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

        return self.labels_

    def _expand_cluster(self, X, root_idx, neighbors, cluster_id):
        self.labels_[root_idx] = cluster_id
        queue = list(neighbors)

        while queue:
            curr_idx = queue.pop(0)

            if self.labels_[curr_idx] == -1:
                self.labels_[curr_idx] = cluster_id

            elif self.labels_[curr_idx] != -1:
                continue

            self.labels_[curr_idx] = cluster_id

            curr_neighbors = self._get_neighbors(X, curr_idx)

            if len(curr_neighbors) >= self.min_samples:
                queue.extend(curr_neighbors)

    def _get_neighbors(self, X, idx):
        diff = X - X[idx]
        dist = np.linalg.norm(diff, axis=1)
        return np.where(dist <= self.eps)[0]
