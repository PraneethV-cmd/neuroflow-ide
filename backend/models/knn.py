"""
K-Nearest Neighbors Models
"""
import numpy as np
from ..utils.distances import (
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    chebyshev_distance,
    cosine_similarity_distance
)


class KNNRegressor:
    """K-Nearest Neighbors Regressor with multiple distance metrics"""
    
    def __init__(self, k=5, distance_metric='euclidean', minkowski_p=3):
        self.k = k
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p
        self.X_train = None
        self.y_train = None
        self.X_mean = None
        self.X_std = None
        
        # Map metric names to functions
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'minkowski': lambda x1, x2: minkowski_distance(x1, x2, p=self.minkowski_p),
            'chebyshev': chebyshev_distance,
            'cosine': cosine_similarity_distance
        }
    
    def fit(self, X, y):
        """Store training data and compute scaling parameters"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Feature scaling for better distance calculations
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)
        
        # Scale features
        self.X_train = (X - self.X_mean) / self.X_std
        self.y_train = y
        
        return self
    
    def predict(self, X):
        """Predict using KNN"""
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Scale features using training statistics
        X_scaled = (X - self.X_mean) / self.X_std
        
        predictions = []
        distance_func = self.distance_functions.get(self.distance_metric, euclidean_distance)
        
        for x in X_scaled:
            # Calculate distances to all training points
            distances = []
            for x_train in self.X_train:
                dist = distance_func(x, x_train)
                distances.append(dist)
            
            distances = np.array(distances)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get corresponding y values
            k_nearest_labels = self.y_train[k_indices]
            
            # Predict as mean of k nearest neighbors
            prediction = np.mean(k_nearest_labels)
            predictions.append(prediction)
        
        return np.array(predictions)


class KNNClassifier:
    """K-Nearest Neighbors Classifier with multiple distance metrics"""
    
    def __init__(self, k=5, distance_metric='euclidean', minkowski_p=3):
        self.k = k
        self.distance_metric = distance_metric
        self.minkowski_p = minkowski_p
        self.X_train = None
        self.y_train = None
        self.X_mean = None
        self.X_std = None
        self.classes = None
        
        # Map metric names to functions
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'minkowski': lambda x1, x2: minkowski_distance(x1, x2, p=self.minkowski_p),
            'chebyshev': chebyshev_distance,
            'cosine': cosine_similarity_distance
        }
    
    def fit(self, X, y):
        """Store training data and compute scaling parameters"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Store unique classes
        self.classes = np.unique(y)
        
        # Feature scaling for better distance calculations
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)
        
        # Scale features
        self.X_train = (X - self.X_mean) / self.X_std
        self.y_train = y
        
        return self
    
    def predict(self, X):
        """Predict using KNN classification (majority vote)"""
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Scale features using training statistics
        X_scaled = (X - self.X_mean) / self.X_std
        
        predictions = []
        distance_func = self.distance_functions.get(self.distance_metric, euclidean_distance)
        
        for x in X_scaled:
            # Calculate distances to all training points
            distances = []
            for x_train in self.X_train:
                dist = distance_func(x, x_train)
                distances.append(dist)
            
            distances = np.array(distances)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get corresponding y values (classes)
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote - find most common class
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            prediction = unique[np.argmax(counts)]
            predictions.append(prediction)
        
        return np.array(predictions)
