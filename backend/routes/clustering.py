"""
Clustering API Routes
"""
from flask import Blueprint, request, jsonify
import numpy as np
from backend.models.clustering import KMeans
from backend.models.decomposition import PCA
from backend.utils.preprocessing import safe_float_conversion

clustering_bp = Blueprint('clustering', __name__)


@clustering_bp.route("/api/kmeans", methods=["POST"])
def api_kmeans():
    """
    K-Means Clustering endpoint
    
    Request body:
    - X: Feature matrix (2D array) - selected numeric columns only
    - n_clusters: Number of clusters (2-20)
    - max_iters: Maximum iterations (default: 300)
    - tol: Convergence tolerance (default: 1e-4)
    - random_state: Random seed for reproducibility (default: 42)
    - distance_metric: Distance metric to use (default: 'euclidean')
    - minkowski_p: P value for Minkowski distance (default: 3)
    """
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        
        # Get parameters
        n_clusters = data.get("n_clusters", 3)
        max_iters = data.get("max_iters", 300)
        tol = data.get("tol", 1e-4)
        random_state = data.get("random_state", 42)
        distance_metric = data.get("distance_metric", "euclidean")
        minkowski_p = data.get("minkowski_p", 3)
        
        # Validate n_clusters
        if n_clusters < 2 or n_clusters > 20:
            return jsonify({"error": "Number of clusters must be between 2 and 20"}), 400
        
        # Validate we have enough samples
        n_samples = len(X)
        if n_clusters >= n_samples:
            return jsonify({
                "error": f"Number of clusters ({n_clusters}) must be less than number of samples ({n_samples})"
            }), 400
        
        # Validate distance metric
        valid_metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine']
        if distance_metric not in valid_metrics:
            return jsonify({
                "error": f"Invalid distance metric. Must be one of: {valid_metrics}"
            }), 400
        
        # Create and fit K-Means model
        model = KMeans(
            n_clusters=n_clusters,
            max_iters=max_iters,
            tol=tol,
            distance_metric=distance_metric,
            minkowski_p=minkowski_p,
            random_state=random_state
        )
        
        model.fit(X)
        
        # Get cluster labels
        cluster_labels = model.labels
        
        # Get cluster centers (need to unscale them back to original space)
        # Centroids are in scaled space, convert back to original
        centroids_scaled = model.centroids
        centroids_original = centroids_scaled * model.X_std + model.X_mean
        
        # Calculate cluster statistics
        cluster_sizes = []
        for i in range(n_clusters):
            cluster_sizes.append(int(np.sum(cluster_labels == i)))
        
        # Calculate inertia (sum of squared distances to nearest centroid)
        inertia = float(model.inertia_)
        
        # --- PCA for Visualization ---
        try:

            
            # Use PCA to project data to 2D for visualization
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(X)
            
            # Also project centroids
            # Centroids are in original space in `centroids_original`
            centroids_pca = pca.transform(centroids_original)
            
            pca_explained_variance = float(np.sum(pca.explained_variance_ratio_))
            
            pca_data = {
                 "coords": safe_float_conversion(pca_coords),
                 "centroids": safe_float_conversion(centroids_pca),
                 "variance_ratio": pca_explained_variance
            }
        except Exception as pca_error:
            print(f"PCA Error: {pca_error}")
            pca_data = None

        return jsonify({
            "success": True,
            "n_clusters": n_clusters,
            "cluster_labels": [int(label) for label in cluster_labels],
            "cluster_centers": centroids_original.tolist(),
            "cluster_sizes": cluster_sizes,
            "inertia": inertia,
            "n_iterations": int(model.n_iters_),
            "distance_metric": distance_metric,
            "converged": model.n_iters_ < max_iters,
            # Store scaling parameters for Model Evaluator
            "X_mean": safe_float_conversion(model.X_mean),
            "X_std": safe_float_conversion(model.X_std),
            "pca_data": pca_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@clustering_bp.route("/api/hierarchical-clustering", methods=["POST"])
def api_hierarchical_clustering():
    """
    Hierarchical Clustering endpoint
    
    Request body:
    - X: Feature matrix (2D array) - selected numeric columns only
    - linkage_method: Linkage method ('single', 'complete', 'average', 'ward')
    - distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine')
    - n_clusters: Optional number of clusters to form (cuts dendrogram)
    - distance_threshold: Optional distance threshold to cut dendrogram
    """
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        from scipy.spatial.distance import pdist
        
        data = request.json
        X = np.array(data["X"], dtype=float)
        
        # Get parameters
        linkage_method = data.get("linkage_method", "ward")
        distance_metric = data.get("distance_metric", "euclidean")
        n_clusters = data.get("n_clusters", None)
        distance_threshold = data.get("distance_threshold", None)
        
        # Validate linkage method
        valid_linkage = ['single', 'complete', 'average', 'ward']
        if linkage_method not in valid_linkage:
            return jsonify({
                "error": f"Invalid linkage method. Must be one of: {valid_linkage}"
            }), 400
        
        # Validate distance metric
        valid_metrics = ['euclidean', 'manhattan', 'cosine']
        if distance_metric not in valid_metrics:
            return jsonify({
                "error": f"Invalid distance metric. Must be one of: {valid_metrics}"
            }), 400
        
        # Ward linkage only supports Euclidean distance
        if linkage_method == 'ward' and distance_metric != 'euclidean':
            return jsonify({
                "error": "Ward linkage only supports Euclidean distance"
            }), 400
        
        # Validate that both n_clusters and distance_threshold are not set
        if n_clusters is not None and distance_threshold is not None:
            return jsonify({
                "error": "Cannot specify both n_clusters and distance_threshold simultaneously"
            }), 400
        
        # Validate we have enough samples
        n_samples = len(X)
        if n_samples < 2:
            return jsonify({
                "error": "Need at least 2 samples for hierarchical clustering"
            }), 400
        
        # Compute linkage matrix
        if linkage_method == 'ward':
            # Ward uses euclidean distance by default
            linkage_matrix = linkage(X, method='ward')
        else:
            # For other methods, compute distance matrix first
            distance_matrix = pdist(X, metric=distance_metric)
            linkage_matrix = linkage(distance_matrix, method=linkage_method)
        
        # Generate dendrogram data
        dend = dendrogram(linkage_matrix, no_plot=True)
        
        # Determine cluster labels
        if n_clusters is not None:
            # Cut dendrogram by number of clusters
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            cut_criterion = "n_clusters"
            cut_value = n_clusters
        elif distance_threshold is not None:
            # Cut dendrogram by distance threshold
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
            cut_criterion = "distance_threshold"
            cut_value = distance_threshold
            n_clusters = len(np.unique(cluster_labels))
        else:
            # Default: cut to form 3 clusters
            cluster_labels = fcluster(linkage_matrix, 3, criterion='maxclust')
            cut_criterion = "n_clusters"
            cut_value = 3
            n_clusters = 3
        
        # Convert to 0-indexed labels
        cluster_labels = cluster_labels - 1
        
        # Calculate cluster statistics
        unique_labels = np.unique(cluster_labels)
        cluster_sizes = []
        cluster_representatives = []
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_points = X[cluster_mask]
            
            # Cluster size
            cluster_sizes.append(int(np.sum(cluster_mask)))
            
            # Cluster representative (mean)
            cluster_mean = np.mean(cluster_points, axis=0)
            cluster_representatives.append(cluster_mean.tolist())
        
        # Calculate cut height for dendrogram visualization
        if cut_criterion == "distance_threshold":
            cut_height = cut_value
        else:
            # Find the height at which we get n_clusters
            # This is the (n_samples - n_clusters)th merge distance
            merge_idx = n_samples - n_clusters - 1
            if merge_idx >= 0 and merge_idx < len(linkage_matrix):
                cut_height = float(linkage_matrix[merge_idx, 2])
            else:
                cut_height = 0.0
        
        return jsonify({
            "success": True,
            "n_clusters": int(n_clusters),
            "cluster_labels": [int(label) for label in cluster_labels],
            "cluster_sizes": cluster_sizes,
            "cluster_representatives": cluster_representatives,
            "linkage_method": linkage_method,
            "distance_metric": distance_metric,
            "cut_criterion": cut_criterion,
            "cut_value": float(cut_value),
            "cut_height": cut_height,
            "dendrogram": {
                "icoord": dend['icoord'],
                "dcoord": dend['dcoord'],
                "ivl": dend['ivl'],
                "leaves": dend['leaves'],
                "color_list": dend['color_list']
            },
            "linkage_matrix": linkage_matrix.tolist()
        })
        
    except Exception as e:
        import traceback
        print(f"Hierarchical Clustering Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@clustering_bp.route("/api/dbscan", methods=["POST"])
def api_dbscan():
    """
    DBSCAN Clustering endpoint
    
    Request body:
    - X: Feature matrix (2D array) - selected numeric columns only
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    - distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
    """
    try:
        from backend.models.clustering import DBSCAN
        
        data = request.json
        X = np.array(data["X"], dtype=float)
        
        # Get parameters
        eps = float(data.get("eps", 0.5))
        min_samples = int(data.get("min_samples", 5))
        distance_metric = data.get("distance_metric", "euclidean")
        minkowski_p = float(data.get("minkowski_p", 3))
        
        # Validate parameters
        if eps <= 0:
            return jsonify({"error": "Epsilon must be positive"}), 400
        if min_samples < 1:
            return jsonify({"error": "Minimum samples must be at least 1"}), 400
            
        valid_metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine']
        if distance_metric not in valid_metrics:
            return jsonify({
                "error": f"Invalid distance metric. Must be one of: {valid_metrics}"
            }), 400
            
        # Create and fit DBSCAN model
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            distance_metric=distance_metric,
            minkowski_p=minkowski_p
        )
        
        model.fit(X)
        
        # Get clustering results
        cluster_labels = model.labels
        n_clusters = model.n_clusters_
        n_noise = model.n_noise_
        
        # Calculate cluster statistics and representatives
        cluster_sizes = []
        cluster_representatives = []
        unique_labels = sorted(list(set(cluster_labels)))
        
        # X is in original scale
        
        for label in unique_labels:
            if label == -1:
                continue # Skip noise for representatives
                
            mask = cluster_labels == label
            cluster_points = X[mask]
            
            cluster_sizes.append(int(np.sum(mask)))
            
            # Use mean as representative for visualization
            cluster_rep = np.mean(cluster_points, axis=0)
            cluster_representatives.append(cluster_rep.tolist())
            
        # Get core samples for Model Evaluator
        # We need to return core samples in ORIGINAL scale
        # The model uses scaled data internally
        core_indices = model.core_sample_indices_
        core_samples = X[core_indices]
        core_sample_labels = cluster_labels[core_indices]
        
        # Limit core samples if there are too many (to prevent payload issues)
        # 1000 core samples should be enough for a reasonable frontend approximation
        max_core_samples = 1000
        if len(core_indices) > max_core_samples:
            # Subsample core points
            indices = np.random.choice(len(core_indices), max_core_samples, replace=False)
            core_samples = core_samples[indices]
            core_sample_labels = core_sample_labels[indices]

        # --- PCA for Visualization ---
        try:

            
            # Use PCA to project data to 2D for visualization
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(X)
            
            pca_explained_variance = float(np.sum(pca.explained_variance_ratio_))
            
            pca_data = {
                 "coords": safe_float_conversion(pca_coords),
                 "variance_ratio": pca_explained_variance
            }
        except Exception as pca_error:
            # print(f"PCA Error: {pca_error}")
            pca_data = None

        return jsonify({
            "success": True,
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "cluster_labels": [int(l) for l in cluster_labels],
            "cluster_sizes": cluster_sizes,
            "cluster_representatives": cluster_representatives,
            "core_samples": safe_float_conversion(core_samples),
            "core_sample_labels": [int(l) for l in core_sample_labels],
            "distance_metric": distance_metric,
            "eps": eps,
            "min_samples": min_samples,
            # Pass original data statistics for normalization in frontend prediction
            "X_mean": safe_float_conversion(model.X_mean),
            "X_std": safe_float_conversion(model.X_std),
            "pca_data": pca_data
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
