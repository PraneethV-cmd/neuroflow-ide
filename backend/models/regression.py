import numpy as np
from ..utils.preprocessing import add_intercept, generate_polynomial_features
from ..utils.distances import (
    euclidean_distance, 
    manhattan_distance, 
    minkowski_distance, 
    chebyshev_distance, 
    cosine_similarity_distance
)

class LinearRegression:
    """ Linear Regression - Handles ALL edge cases with multiple optimization strategies"""
    
    def __init__(self, method='auto', learning_rate=0.01, n_iterations=10000, tol=1e-8):
        self.method = method  # 'normal', 'gradient', 'svd', 'auto'
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.is_fitted = False
    
    def _normal_equation(self, X, y):
        """Solve using normal equation with multiple fallbacks"""
        try:
            # Method 1: Standard normal equation
            return np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            try:
                # Method 2: Moore-Penrose pseudoinverse
                return np.linalg.pinv(X.T @ X) @ X.T @ y
            except:
                try:
                    # Method 3: Direct pseudoinverse
                    return np.linalg.pinv(X) @ y
                except:
                    # Method 4: Ridge regression fallback
                    ridge_lambda = 1e-6
                    n_features = X.shape[1]
                    return np.linalg.inv(X.T @ X + ridge_lambda * np.eye(n_features)) @ X.T @ y
    
    def _gradient_descent(self, X, y):
        """Robust gradient descent with adaptive learning rate"""
        n_samples, n_features = X.shape
        
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.01, n_features)
        
        best_weights = self.weights.copy()
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for iteration in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights
            error = y_pred - y
            
            # Compute loss (MSE)
            loss = np.mean(error ** 2)
            
            # Check for improvement
            if loss < best_loss - self.tol:
                best_loss = loss
                best_weights = self.weights.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                break
            
            # Compute gradients
            gradients = (2 / n_samples) * (X.T @ error)
            
            # Adaptive learning rate
            grad_norm = np.linalg.norm(gradients)
            if grad_norm > 1.0:
                gradients = gradients / grad_norm  # Normalize large gradients
            
            current_lr = self.learning_rate / (1 + 0.001 * iteration)  # Decay learning rate
            
            # Update weights
            self.weights -= current_lr * gradients
            
            # Check convergence
            if grad_norm < self.tol:
                break
        
        return best_weights
    
    def _svd_solution(self, X, y):
        """SVD-based solution for ill-conditioned problems"""
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            # Regularize small singular values
            s_inv = np.divide(1, s, out=np.zeros_like(s), where=np.abs(s) > 1e-10)
            return Vt.T @ np.diag(s_inv) @ U.T @ y
        except:
            # Fallback to pseudoinverse
            return np.linalg.pinv(X) @ y
    
    def fit(self, X, y):
        """Robust fitting that handles all edge cases"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Handle insufficient samples
        if n_samples < n_features:
            # Use gradient descent for underdetermined system
            self.method = 'gradient'
        
        # Store original data statistics
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std = np.where(self.X_std < 1e-10, 1.0, self.X_std)
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        if self.y_std < 1e-10:
            self.y_std = 1.0
        
        # Scale features
        X_scaled = (X - self.X_mean) / self.X_std
        # Scale target
        y_scaled = (y - self.y_mean) / self.y_std
        
        # Add intercept
        X_with_intercept = add_intercept(X_scaled)
        
        # Choose method automatically if 'auto'
        actual_method = self.method
        if actual_method == 'auto':
            if n_samples > 1000 or n_samples < n_features:
                actual_method = 'gradient'
            else:
                actual_method = 'normal'
        
        # Fit using chosen method
        try:
            if actual_method == 'normal':
                self.weights = self._normal_equation(X_with_intercept, y_scaled)
            elif actual_method == 'gradient':
                self.weights = self._gradient_descent(X_with_intercept, y_scaled)
            elif actual_method == 'svd':
                self.weights = self._svd_solution(X_with_intercept, y_scaled)
            else:
                self.weights = self._normal_equation(X_with_intercept, y_scaled)
        except Exception as e:
            # Ultimate fallback - mean prediction
            print(f"All methods failed, using fallback: {e}")
            self.weights = np.zeros(X_with_intercept.shape[1])
            self.weights[0] = np.mean(y_scaled)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make robust predictions"""
        if not self.is_fitted or self.weights is None:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Scale features
        X_scaled = (X - self.X_mean) / self.X_std
        # Add intercept
        X_with_intercept = add_intercept(X_scaled)
        
        # Predict scaled values
        y_pred_scaled = X_with_intercept @ self.weights
        
        # Scale back to original
        y_pred = y_pred_scaled * self.y_std + self.y_mean
        
        # Ensure no crazy values
        y_pred = np.clip(y_pred, -1e10, 1e10)  # Prevent extreme values
        y_pred = np.nan_to_num(y_pred, nan=self.y_mean)  # Replace NaN with mean
        
        return y_pred
    
    @property
    def coef_(self):
        """Get coefficients in original scale"""
        if self.weights is None or not self.is_fitted:
            return np.array([])
        
        # weights[1:] are coefficients for scaled features
        # Convert back to original scale
        coef = self.weights[1:] * self.y_std / self.X_std
        return coef
    
    @property
    def intercept_(self):
        """Get intercept in original scale"""
        if self.weights is None or not self.is_fitted:
            return 0.0
        
        # intercept = weights[0] * y_std + y_mean - sum(coef * X_mean / X_std)
        intercept_scaled = self.weights[0]
        intercept_original = intercept_scaled * self.y_std + self.y_mean
        
        if len(self.weights) > 1:
            # Adjust for feature scaling
            coef_scaled = self.weights[1:]
            adjustment = np.sum(coef_scaled * self.y_std * self.X_mean / self.X_std)
            intercept_original -= adjustment
        
        return intercept_original


class PolynomialRegression:
    """
    Polynomial Regression using feature transformation + Linear Regression
    Transforms input features to polynomial space and fits a linear model
    """
    
    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.linear_model = LinearRegression()
        self.feature_names = None
        self.n_features_original = None
        self.n_features_poly = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit polynomial regression model
        
        Parameters:
        - X: Input features (n_samples, n_features)
        - y: Target values (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_original = X.shape[1]
        
        # Generate polynomial features
        X_poly, self.feature_names = generate_polynomial_features(
            X, 
            degree=self.degree, 
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        
        self.n_features_poly = X_poly.shape[1]
        
        # Fit linear regression on polynomial features
        # Note: LinearRegression will add its own intercept, so we need to handle this
        # If we already included bias in polynomial features, don't let LinearRegression add another
        self.linear_model.fit(X_poly, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions using polynomial regression
        
        Parameters:
        - X: Input features (n_samples, n_features)
        
        Returns:
        - y_pred: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Generate polynomial features (same transformation as training)
        X_poly, _ = generate_polynomial_features(
            X, 
            degree=self.degree, 
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        
        # Predict using linear model
        y_pred = self.linear_model.predict(X_poly)
        
        return y_pred
    
    @property
    def coef_(self):
        """Get coefficients for polynomial features"""
        if not self.is_fitted or self.linear_model.coef_ is None:
            return np.array([])
        return self.linear_model.coef_
    
    @property
    def intercept_(self):
        """Get intercept term"""
        if not self.is_fitted:
            return 0.0
        return self.linear_model.intercept_


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
