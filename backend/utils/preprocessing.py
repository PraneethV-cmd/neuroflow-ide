import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Robust train-test split implementation"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = max(1, int(n_samples * test_size))  # Ensure at least 1 test sample
    n_train = n_samples - n_test
    
    if n_train < 1:
        raise ValueError("Not enough samples for training")
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def add_intercept(X):
    """Add intercept term to features"""
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])

def safe_float_conversion(arr):
    """Convert numpy array to list with NaN/Inf checking"""
    if arr is None:
        return []
    arr = np.array(arr)
    # Replace NaN and Inf with 0 or appropriate values
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
    # Convert to Python float types
    return [float(x) for x in arr]

def robust_feature_scaling(X):
    """Robust feature scaling that handles constant features"""
    if X.size == 0:
        return X, np.array([]), np.array([])
    
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Handle constant features
    std = np.where(std < 1e-10, 1.0, std)
    
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def robust_target_scaling(y):
    """Scale target variable for better convergence"""
    y = np.array(y, dtype=float)
    y_mean = np.mean(y)
    y_std = np.std(y)
    
    if y_std < 1e-10:
        return y, 0.0, 1.0  # Constant target
    
    y_scaled = (y - y_mean) / y_std
    return y_scaled, y_mean, y_std

def generate_polynomial_features(X, degree=2, include_bias=True, interaction_only=False):
    """
    Generate polynomial features from input data
    
    Parameters:
    - X: Input features (numpy array), shape (n_samples, n_features)
    - degree: Polynomial degree (1-5)
    - include_bias: Whether to include bias term (constant 1)
    - interaction_only: If True, only generate interaction terms (no powers)
    
    Returns:
    - X_poly: Transformed feature matrix
    - feature_names: List of feature names for reference
    """
    X = np.array(X, dtype=float)
    
    # Handle 1D case
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Start with bias if requested
    features = []
    feature_names = []
    
    if include_bias:
        features.append(np.ones(n_samples))
        feature_names.append('1')
    
    # Add original features (degree 1)
    for i in range(n_features):
        features.append(X[:, i])
        feature_names.append(f'x{i}')
    
    # Generate higher degree features
    if degree > 1:
        # For each degree from 2 to specified degree
        for d in range(2, degree + 1):
            # Generate all combinations of features with total degree = d
            from itertools import combinations_with_replacement
            
            for combo in combinations_with_replacement(range(n_features), d):
                # Check if this is an interaction term or a power term
                is_interaction = len(set(combo)) > 1
                is_power = len(set(combo)) == 1
                
                # Skip power terms if interaction_only is True
                if interaction_only and is_power:
                    continue
                
                # Compute the feature
                feature = np.ones(n_samples)
                name_parts = []
                for idx in combo:
                    feature *= X[:, idx]
                    name_parts.append(f'x{idx}')
                
                features.append(feature)
                
                # Create readable name
                if is_power:
                    feature_names.append(f'x{combo[0]}^{d}')
                else:
                    feature_names.append('*'.join(name_parts))
    
    # Stack all features
    X_poly = np.column_stack(features)
    
    return X_poly, feature_names
