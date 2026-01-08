import numpy as np

def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    """Calculate Manhattan distance between two points"""
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p=3):
    """Calculate Minkowski distance between two points"""
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)

def chebyshev_distance(x1, x2):
    """Calculate Chebyshev distance between two points"""
    return np.max(np.abs(x1 - x2))

def cosine_similarity_distance(x1, x2):
    """Calculate Cosine Similarity distance between two points"""
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    
    if norm_x1 == 0 or norm_x2 == 0:
        return 1.0  # Maximum distance if either vector is zero
    
    cosine_sim = dot_product / (norm_x1 * norm_x2)
    # Convert similarity to distance (1 - similarity)
    return 1 - cosine_sim
