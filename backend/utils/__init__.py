"""
Utility functions for the backend
"""
from .metrics import regression_metrics, classification_metrics
from .preprocessing import (
    train_test_split,
    add_intercept,
    safe_float_conversion,
    robust_feature_scaling,
    robust_target_scaling,
    generate_polynomial_features
)
from .distances import (
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    chebyshev_distance,
    cosine_similarity_distance
)

__all__ = [
    'regression_metrics',
    'classification_metrics',
    'train_test_split',
    'add_intercept',
    'safe_float_conversion',
    'robust_feature_scaling',
    'robust_target_scaling',
    'generate_polynomial_features',
    'euclidean_distance',
    'manhattan_distance',
    'minkowski_distance',
    'chebyshev_distance',
    'cosine_similarity_distance'
]
