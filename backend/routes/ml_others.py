from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.preprocessing import safe_float_conversion
from backend.models.decomposition import PCA, SVD

ml_others_bp = Blueprint('ml_others', __name__)

@ml_others_bp.route("/api/pca", methods=["POST"])
def api_pca():
    try:
        req = request.json
        X = np.array(req["data"], dtype=float)
        headers = req.get("headers", [])
        
        standardize = req.get("standardize", True)
        fixed_components = req.get("fixed_components")
        variance_threshold = req.get("variance_threshold", 0.95)
        
        n_components = fixed_components if fixed_components else None
        
        model = PCA(n_components=n_components, variance_threshold=variance_threshold, standardize=standardize)
        X_transformed = model.fit_transform(X)
        
        response = {
            "success": True,
            "transformed_data": safe_float_conversion(X_transformed),
            "n_components": int(model.n_components_),
            "explained_variance_ratio": safe_float_conversion(model.explained_variance_ratio_),
            "cumulative_variance": safe_float_conversion(np.cumsum(model.explained_variance_ratio_))
        }
        
        if req.get("return_loadings", False) and headers:
            loadings = model.get_loadings()
            loadings_data = []
            for i, feature_name in enumerate(headers):
                row = {"feature": feature_name}
                for j in range(model.n_components_):
                    row[f"PC{j+1}"] = float(loadings[i, j])
                loadings_data.append(row)
            response["loadings"] = loadings_data
            
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ml_others_bp.route("/api/svd", methods=["POST"])
def api_svd():
    try:
        req = request.json
        X = np.array(req["data"], dtype=float)
        n_components = req.get("n_components", 2)
        
        model = SVD(n_components=n_components)
        X_transformed = model.fit_transform(X)
        
        return jsonify({
            "success": True,
            "transformed_data": safe_float_conversion(X_transformed),
            "explained_variance_ratio": safe_float_conversion(model.explained_variance_ratio_),
            "cumulative_variance": safe_float_conversion(np.cumsum(model.explained_variance_ratio_))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
