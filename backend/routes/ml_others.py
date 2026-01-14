from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.preprocessing import safe_float_conversion
from backend.models.decomposition import PCA, SVD

ml_others_bp = Blueprint('ml_others', __name__)

@ml_others_bp.route("/api/pca", methods=["POST"])
def api_pca():
    """
    PCA endpoint for dimensionality reduction
    
    Request body:
    - data: 2D array of numeric features (selected columns only)
    - headers: Column names for selected features
    - full_rows: Optional full row data including unselected columns
    - all_headers: Optional all column headers
    - selected_indices: Optional indices of selected columns in full data
    - n_components: Optional explicit component count
    - variance_threshold: Optional variance retention (0.0-1.0)
    - standardize: Whether to standardize data (default: true)
    - return_loadings: Whether to return component loadings (default: false)
    - return_explained_variance: Whether to return variance details (default: true)
    """
    try:
        request_data = request.json
        
        # Extract data and configuration
        data = np.array(request_data["data"], dtype=float)
        headers = request_data.get("headers", [])
        full_rows = request_data.get("full_rows", None)
        all_headers = request_data.get("all_headers", None)
        selected_indices = request_data.get("selected_indices", None)
        n_components = request_data.get("n_components", None)
        variance_threshold = request_data.get("variance_threshold", None)
        standardize = request_data.get("standardize", True)
        return_loadings = request_data.get("return_loadings", False)
        return_explained_variance = request_data.get("return_explained_variance", True)
        
        # Validate inputs
        if data.size == 0:
            return jsonify({"error": "No data provided"}), 400
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        
        # Validate component count
        if n_components is not None:
            if n_components < 1 or n_components > n_features:
                return jsonify({
                    "error": f"n_components must be between 1 and {n_features}"
                }), 400
        
        # Validate variance threshold
        if variance_threshold is not None:
            if variance_threshold < 0.0 or variance_threshold > 1.0:
                return jsonify({
                    "error": "variance_threshold must be between 0.0 and 1.0"
                }), 400
        
        # Create and fit PCA
        pca = PCA(
            n_components=n_components,
            variance_threshold=variance_threshold,
            standardize=standardize
        )
        
        # Transform data
        transformed_data = pca.fit_transform(data)
        
        # Generate component headers (PC1, PC2, PC3, ...)
        component_headers = [f"PC{i+1}" for i in range(pca.n_components_)]
        
        # Combine transformed data with unselected columns if provided
        if full_rows is not None and all_headers is not None and selected_indices is not None:
            full_rows_array = np.array(full_rows)
            selected_indices_set = set(selected_indices)
            
            # Get unselected column indices
            unselected_indices = [i for i in range(len(all_headers)) if i not in selected_indices_set]
            
            if len(unselected_indices) > 0:
                # Extract unselected columns
                unselected_data = full_rows_array[:, unselected_indices]
                unselected_headers = [all_headers[i] for i in unselected_indices]
                
                # Combine: unselected columns first, then PCA components
                combined_data = np.column_stack([unselected_data, transformed_data])
                combined_headers = unselected_headers + component_headers
            else:
                # No unselected columns, just use transformed data
                combined_data = transformed_data
                combined_headers = component_headers
        else:
            # No full row data provided, just use transformed data
            combined_data = transformed_data
            combined_headers = component_headers
        
        # Prepare response
        response = {
            "success": True,
            "transformed_data": combined_data.tolist(),
            "component_headers": combined_headers,
            "n_components_used": int(pca.n_components_),
            "n_features_original": int(n_features),
            "standardized": standardize
        }
        
        # Add explained variance if requested
        if return_explained_variance:
            explained_var = pca.explained_variance_ratio_[:pca.n_components_]
            cumulative_var = np.cumsum(explained_var)
            
            response["explained_variance"] = safe_float_conversion(explained_var)
            response["cumulative_variance"] = safe_float_conversion(cumulative_var)
            response["total_variance_explained"] = float(cumulative_var[-1]) if len(cumulative_var) > 0 else 0.0
        
        # Add component loadings if requested
        if return_loadings:
            loadings = pca.get_loadings()
            response["loadings"] = loadings.tolist()
            response["original_features"] = headers if headers else [f"Feature_{i+1}" for i in range(n_features)]
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ml_others_bp.route("/api/svd", methods=["POST"])
def api_svd():
    """
    SVD endpoint
    """
    try:
        request_data = request.json
        
        # Extract data and configuration
        data = np.array(request_data["data"], dtype=float)
        headers = request_data.get("headers", [])
        n_components = request_data.get("n_components", None)
        
        # Handle optional full rows for passing through unselected data
        full_rows = request_data.get("full_rows", None)
        all_headers = request_data.get("all_headers", None)
        selected_indices = request_data.get("selected_indices", None)
        
        if data.size == 0:
            return jsonify({"error": "No data provided"}), 400
            
        n_samples, n_features = data.shape
        if n_components is not None:
             if n_components < 1 or n_components > min(n_samples, n_features):
                return jsonify({
                    "error": f"n_components must be between 1 and min(n_samples, n_features)={min(n_samples, n_features)}"
                }), 400
        
        svd = SVD(n_components=n_components)
        transformed_data = svd.fit_transform(data)
        
        # Component headers
        k = transformed_data.shape[1]
        component_headers = [f"Component_{i+1}" for i in range(k)]
        
        # Combine with unselected columns if provided (Copied logic from PCA)
        if full_rows is not None and all_headers is not None and selected_indices is not None:
            full_rows_array = np.array(full_rows)
            selected_indices_set = set(selected_indices)
            unselected_indices = [i for i in range(len(all_headers)) if i not in selected_indices_set]
            
            if len(unselected_indices) > 0:
                unselected_data = full_rows_array[:, unselected_indices]
                unselected_headers = [all_headers[i] for i in unselected_indices]
                combined_data = np.column_stack([unselected_data, transformed_data])
                combined_headers = unselected_headers + component_headers
            else:
                combined_data = transformed_data
                combined_headers = component_headers
        else:
            combined_data = transformed_data
            combined_headers = component_headers

        response = {
            "success": True,
            "transformed_data": combined_data.tolist(),
            "component_headers": combined_headers,
            "singular_values": safe_float_conversion(svd.S),
            "explained_variance_ratio": safe_float_conversion(svd.explained_variance_ratio_),
            "n_components": k
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
