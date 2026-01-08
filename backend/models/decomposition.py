import numpy as np

class PCA:
    """
    Principal Component Analysis implementation from scratch
    Supports both explicit component count and variance-based selection
    """
    
    def __init__(self, n_components=None, variance_threshold=None, standardize=True):
        """
        Initialize PCA
        
        Parameters:
        - n_components: Explicit number of components (1 to N)
        - variance_threshold: Variance retention threshold (0.0 to 1.0)
        - standardize: Whether to standardize data before PCA
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.standardize = standardize
        
        # Fitted attributes
        self.mean_ = None
        self.std_ = None
        self.components_ = None  # Principal components (eigenvectors)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None  # Actual number of components used
        self.is_fitted = False
    
    def fit(self, X):
        """
        Fit PCA on data X
        
        Parameters:
        - X: Input data (n_samples, n_features)
        
        Returns:
        - self
        """
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Standardize data if requested
        if self.standardize:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            # Handle constant features
            self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)
            X_centered = (X - self.mean_) / self.std_
        else:
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            self.std_ = np.ones(n_features)
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Ensure eigenvalues are non-negative (numerical stability)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Calculate explained variance
        total_variance = np.sum(eigenvalues)
        if total_variance < 1e-10:
            # All features are constant
            self.explained_variance_ratio_ = np.zeros(n_features)
        else:
            self.explained_variance_ratio_ = eigenvalues / total_variance
        
        self.explained_variance_ = eigenvalues
        
        # Determine number of components
        if self.variance_threshold is not None:
            # Use variance threshold
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            self.n_components_ = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            # Ensure at least 1 component
            self.n_components_ = max(1, self.n_components_)
        elif self.n_components is not None:
            # Use explicit component count
            self.n_components_ = min(self.n_components, n_features)
        else:
            # Default: use all components
            self.n_components_ = n_features
        
        # Store principal components (eigenvectors)
        self.components_ = eigenvectors[:, :self.n_components_]
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform data using fitted PCA
        
        Parameters:
        - X: Input data (n_samples, n_features)
        
        Returns:
        - X_transformed: Transformed data (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Standardize using fitted parameters
        if self.standardize:
            X_centered = (X - self.mean_) / self.std_
        else:
            X_centered = X - self.mean_
        
        # Project onto principal components
        X_transformed = X_centered @ self.components_
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data in one step
        
        Parameters:
        - X: Input data (n_samples, n_features)
        
        Returns:
        - X_transformed: Transformed data (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def get_loadings(self):
        """
        Get component loadings (correlation between original features and PCs)
        
        Returns:
        - loadings: Component loadings matrix (n_features, n_components)
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        # Loadings are the eigenvectors scaled by sqrt of eigenvalues
        loadings = self.components_ * np.sqrt(self.explained_variance_[:self.n_components_])
        return loadings

class SVD:
    """
    Singular Value Decomposition implementation using numpy
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.is_fitted = False
        
    def fit_transform(self, X):
        """
        Fit SVD and return transformed data (U * S)
        """
        X = np.array(X, dtype=float)
        
        # Handle 1D case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Compute SVD
        # full_matrices=False gives U: (M, K), S: (K,), Vt: (K, N) where K = min(M, N)
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            
            self.U = U
            self.S = S
            self.Vt = Vt
            
            # Calculate explained variance (squared singular values / (n_samples - 1))
            n_samples = X.shape[0]
            if n_samples > 1:
                self.explained_variance_ = (S ** 2) / (n_samples - 1)
                total_var = np.sum(self.explained_variance_)
                if total_var > 0:
                    self.explained_variance_ratio_ = self.explained_variance_ / total_var
                else:
                    self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)
            else:
                self.explained_variance_ = np.zeros_like(S)
                self.explained_variance_ratio_ = np.zeros_like(S)

            # Limit components if requested
            if self.n_components is not None:
                k = min(self.n_components, len(S))
                self.U = self.U[:, :k]
                self.S = self.S[:k]
                self.Vt = self.Vt[:k, :]
                self.explained_variance_ = self.explained_variance_[:k]
                self.explained_variance_ratio_ = self.explained_variance_ratio_[:k]
            
            self.is_fitted = True
            
            # Transform data: Projected data = U * S
            X_transformed = self.U * self.S
            
            return X_transformed
            
        except np.linalg.LinAlgError:
            raise ValueError("SVD computation failed (LinAlgError)")
