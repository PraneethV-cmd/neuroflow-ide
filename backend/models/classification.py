import numpy as np
from ..utils.preprocessing import add_intercept
from ..utils.distances import (
    euclidean_distance, 
    manhattan_distance, 
    minkowski_distance, 
    chebyshev_distance, 
    cosine_similarity_distance
)

def sigmoid(x):
    """Numerically stable sigmoid"""
    x = np.clip(x, -500, 500)  # Prevent overflow
    return np.where(x >= 0, 
                   1 / (1 + np.exp(-x)), 
                   np.exp(x) / (1 + np.exp(x)))

class LogisticRegression:
    """Logistic Regression implementation from scratch"""
    
    def __init__(self, learning_rate=0.1, n_iterations=10000, C=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C  # Regularization parameter
        self.weights = None
        self.X_mean = None
        self.X_std = None
    
    def fit(self, X, y):
        """Train using gradient descent with feature scaling"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Add intercept and handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_original = X.copy()
        X = add_intercept(X)
        
        n_samples, n_features = X.shape
        
        # Feature scaling (crucial for logistic regression stability)
        if n_features > 1:
            self.X_mean = np.mean(X_original, axis=0)
            self.X_std = np.std(X_original, axis=0)
            self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)
            X[:, 1:] = (X[:, 1:] - self.X_mean) / self.X_std
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            linear_model = X @ self.weights
            y_pred = sigmoid(linear_model)
            
            # Compute gradients with L2 regularization
            # Remove regularization from intercept
            regularization = np.zeros_like(self.weights)
            if len(self.weights) > 1:
                regularization[1:] = (1 / self.C) * self.weights[1:]
            
            dw = (1 / n_samples) * (X.T @ (y_pred - y) + regularization)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            
            # Early stopping if gradients are very small
            if np.linalg.norm(dw) < 1e-8:
                break
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_original = X.copy()
        X = add_intercept(X)
        
        # Apply same scaling if needed
        if self.X_mean is not None and self.X_std is not None:
            X[:, 1:] = (X[:, 1:] - self.X_mean) / self.X_std
        
        linear_model = X @ self.weights
        probabilities = sigmoid(linear_model)
        # Ensure no NaN values
        probabilities = np.nan_to_num(probabilities, nan=0.5)
        return np.clip(probabilities, 0.0, 1.0)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions
    
    @property
    def coef_(self):
        if self.weights is None:
            return np.array([])
        
        coef = self.weights[1:].copy()
        # Adjust coefficients back to original scale
        if self.X_std is not None:
            coef = coef / self.X_std
        return coef
    
    @property
    def intercept_(self):
        if self.weights is None:
            return 0.0
        
        intercept = self.weights[0]
        # Adjust intercept back to original scale
        if self.X_mean is not None and self.X_std is not None:
            intercept = intercept - np.sum(self.weights[1:] * self.X_mean / self.X_std)
        return intercept


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


class GaussianNaiveBayes:
    """
    Custom Gaussian Naive Bayes Classifier implementation from scratch
    
    Assumes features follow a Gaussian (normal) distribution within each class.
    Uses Maximum Likelihood Estimation to calculate class priors, means, and variances.
    """
    
    def __init__(self, var_smoothing=1e-9):
        """
        Initialize Gaussian Naive Bayes classifier
        
        Parameters:
        - var_smoothing: Small value added to variances for numerical stability
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # Class means
        self.var_ = None    # Class variances
        self.epsilon_ = None  # Smoothing value added to variance
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes classifier
        
        Parameters:
        - X: Training features (n_samples, n_features)
        - y: Training labels (n_samples,)
        
        Returns:
        - self
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize arrays for storing statistics
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        
        # Calculate statistics for each class
        for idx, cls in enumerate(self.classes_):
            # Get samples belonging to this class
            X_cls = X[y == cls]
            
            # Calculate class prior (P(class))
            self.class_prior_[idx] = X_cls.shape[0] / n_samples
            
            # Calculate mean for each feature (μ)
            self.theta_[idx, :] = np.mean(X_cls, axis=0)
            
            # Calculate variance for each feature (σ²)
            self.var_[idx, :] = np.var(X_cls, axis=0)
        
        # Add smoothing to variance for numerical stability
        # This prevents division by zero and handles constant features
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ += self.epsilon_
        
        self.is_fitted = True
        return self
    
    def _calculate_log_likelihood(self, X):
        """
        Calculate log likelihood for each class
        
        Uses the Gaussian probability density function:
        P(x|class) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        
        Log likelihood:
        log P(x|class) = -0.5 * log(2πσ²) - (x-μ)²/(2σ²)
        
        Parameters:
        - X: Input features (n_samples, n_features)
        
        Returns:
        - log_likelihood: Log likelihood for each class (n_samples, n_classes)
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # Get mean and variance for this class
            mean = self.theta_[idx, :]
            var = self.var_[idx, :]
            
            # Calculate log of Gaussian PDF for each feature
            # log P(x|class) = -0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]
            log_prior = np.log(self.class_prior_[idx])
            log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_prob -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            
            # Add log prior: log P(class|x) ∝ log P(x|class) + log P(class)
            log_likelihood[:, idx] = log_prior + log_prob
        
        return log_likelihood
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        - X: Input features (n_samples, n_features)
        
        Returns:
        - predictions: Predicted class labels (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        log_likelihood = self._calculate_log_likelihood(X)
        
        # Return class with highest log likelihood
        return self.classes_[np.argmax(log_likelihood, axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        - X: Input features (n_samples, n_features)
        
        Returns:
        - probabilities: Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        log_likelihood = self._calculate_log_likelihood(X)
        
        # Convert log likelihoods to probabilities using softmax (stable version)
        # Shift allow for stability
        log_likelihood_stable = log_likelihood - np.max(log_likelihood, axis=1, keepdims=True)
        likelihood = np.exp(log_likelihood_stable)
        probabilities = likelihood / np.sum(likelihood, axis=1, keepdims=True)
        
        return probabilities
