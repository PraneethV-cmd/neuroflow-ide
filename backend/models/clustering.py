import numpy as np
from ..utils.distances import (
    euclidean_distance, 
    manhattan_distance, 
    minkowski_distance, 
    chebyshev_distance, 
    cosine_similarity_distance
)

class KMeans:
    """K-Means clustering with multiple distance metrics"""

    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4,
                 distance_metric='euclidean', minkowski_p=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.X_mean = None
        self.X_std = None

        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'minkowski': lambda x1, x2: minkowski_distance(x1, x2, p=self.minkowski_p),
            'chebyshev': chebyshev_distance,
            'cosine': cosine_similarity_distance
        }

    def fit(self, X):
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Feature scaling
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)

        X_scaled = (X - self.X_mean) / self.X_std

        np.random.seed(self.random_state)

        # Initialize centroids randomly from data points
        random_indices = np.random.choice(len(X_scaled), self.n_clusters, replace=False)
        self.centroids = X_scaled[random_indices]

        distance_func = self.distance_functions.get(
            self.distance_metric, euclidean_distance
        )

        for iteration in range(self.max_iters):
            # Step 1: Assign clusters
            labels = []
            for x in X_scaled:
                distances = [distance_func(x, c) for c in self.centroids]
                labels.append(np.argmin(distances))

            labels = np.array(labels)

            # Step 2: Update centroids
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = X_scaled[labels == i]
                if len(cluster_points) == 0:
                    # Reinitialize empty cluster
                    new_centroids.append(
                        X_scaled[np.random.randint(0, len(X_scaled))]
                    )
                else:
                    new_centroids.append(np.mean(cluster_points, axis=0))

            new_centroids = np.array(new_centroids)

            # Step 3: Check convergence
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.labels = labels
        self.n_iters_ = iteration + 1
        self.inertia_ = self._compute_inertia(X_scaled, labels)

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = (X - self.X_mean) / self.X_std

        distance_func = self.distance_functions.get(
            self.distance_metric, euclidean_distance
        )

        labels = []
        for x in X_scaled:
            distances = [distance_func(x, c) for c in self.centroids]
            labels.append(np.argmin(distances))

        return np.array(labels)

    def _compute_inertia(self, X, labels):
        inertia = 0.0
        for i, x in enumerate(X):
            inertia += euclidean_distance(x, self.centroids[labels[i]]) ** 2
        return inertia


class HierarchicalClustering:
    """Agglomerative Hierarchical Clustering with multiple linkages"""

    def __init__(self, n_clusters=2, linkage='single',
                 distance_metric='euclidean', minkowski_p=3):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p

        self.labels = None
        self.X_mean = None
        self.X_std = None
        self.merge_history = []

        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'minkowski': lambda x1, x2: minkowski_distance(x1, x2, p=self.minkowski_p),
            'chebyshev': chebyshev_distance,
            'cosine': cosine_similarity_distance
        }

    def fit(self, X):
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Feature scaling
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)

        X_scaled = (X - self.X_mean) / self.X_std

        n_samples = len(X_scaled)

        # Initially each point is its own cluster
        clusters = [[i] for i in range(n_samples)]

        distance_func = self.distance_functions.get(
            self.distance_metric, euclidean_distance
        )

        def cluster_distance(c1, c2):
            distances = []
            for i in c1:
                for j in c2:
                    distances.append(distance_func(X_scaled[i], X_scaled[j]))

            if self.linkage == 'single':
                return np.min(distances)
            elif self.linkage == 'complete':
                return np.max(distances)
            elif self.linkage == 'average':
                return np.mean(distances)
            else:
                raise ValueError("Invalid linkage method")

        # Agglomerative merging
        while len(clusters) > self.n_clusters:
            min_dist = float("inf")
            merge_idx = None

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = cluster_distance(clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_idx = (i, j)

            i, j = merge_idx
            new_cluster = clusters[i] + clusters[j]

            self.merge_history.append({
                "cluster_1": clusters[i],
                "cluster_2": clusters[j],
                "distance": min_dist
            })

            clusters.pop(j)
            clusters.pop(i)
            clusters.append(new_cluster)

        # Assign labels
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cluster_id

        self.labels = labels
        self.n_iters_ = n_samples - self.n_clusters

        return self

    def predict(self):
        """Hierarchical clustering does not support predict for new points"""
        return self.labels


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise"""

    def __init__(self, eps=0.5, min_samples=5,
                 distance_metric='euclidean', minkowski_p=3):
        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p

        self.labels = None
        self.X_mean = None
        self.X_std = None

        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'minkowski': lambda x1, x2: minkowski_distance(x1, x2, p=self.minkowski_p),
            'chebyshev': chebyshev_distance,
            'cosine': cosine_similarity_distance
        }

    def fit(self, X):
        X = np.array(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Feature scaling
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)

        X_scaled = (X - self.X_mean) / self.X_std

        n_samples = len(X_scaled)
        labels = np.full(n_samples, -1)  # -1 means noise
        visited = np.zeros(n_samples, dtype=bool)

        distance_func = self.distance_functions.get(
            self.distance_metric, euclidean_distance
        )

        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(X_scaled, i, distance_func)

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # noise
            else:
                self._expand_cluster(
                    X_scaled, labels, visited,
                    i, neighbors, cluster_id, distance_func
                )
                cluster_id += 1

        self.labels = labels
        self.n_clusters_ = cluster_id
        self.n_noise_ = int(np.sum(labels == -1))
        
        # Identify core samples for prediction/evaluation
        self.core_sample_indices_ = []
        for i in range(n_samples):
             # Re-check core point condition to store indices
             neighbors = self._region_query(X_scaled, i, distance_func)
             if len(neighbors) >= self.min_samples:
                 self.core_sample_indices_.append(i)

        return self

    def _region_query(self, X, idx, distance_func):
        neighbors = []
        for j in range(len(X)):
            if distance_func(X[idx], X[j]) <= self.eps:
                neighbors.append(j)
        return neighbors

    def _expand_cluster(self, X, labels, visited,
                        point_idx, neighbors, cluster_id, distance_func):
        labels[point_idx] = cluster_id
        i = 0

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(X, neighbor_idx, distance_func)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(
                        n for n in new_neighbors if n not in neighbors
                    )

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            i += 1

    def predict(self):
        """DBSCAN does not support prediction on new points"""
        return self.labels
