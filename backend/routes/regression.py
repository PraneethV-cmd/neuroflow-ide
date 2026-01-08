from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.preprocessing import train_test_split, safe_float_conversion
from backend.utils.metrics import regression_metrics
from backend.models.regression import LinearRegression, PolynomialRegression, KNNRegressor

regression_bp = Blueprint('regression', __name__)

@regression_bp.route("/api/linear-regression", methods=["POST"])
def api_linear_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        learning_rate = data.get("learning_rate", 0.01)
        n_iterations = data.get("n_iterations", 1000)

        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = LinearRegression(method='gradient', learning_rate=learning_rate, n_iterations=n_iterations)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "slope": float(model.coef_[0]) if len(model.coef_) > 0 else 0.0,
            "intercept": float(model.intercept_),
            "train_metrics": regression_metrics(y_train, y_train_pred),
            "test_metrics": regression_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@regression_bp.route("/api/multi-linear-regression", methods=["POST"])
def api_multi_linear_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        learning_rate = data.get("learning_rate", 0.01)
        n_iterations = data.get("n_iterations", 1000)

        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = LinearRegression(method='gradient', learning_rate=learning_rate, n_iterations=n_iterations)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "coefficients": safe_float_conversion(model.coef_),
            "intercept": float(model.intercept_),
            "train_metrics": regression_metrics(y_train, y_train_pred),
            "test_metrics": regression_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@regression_bp.route("/api/polynomial-regression", methods=["POST"])
def api_polynomial_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        degree = data.get("degree", 2)
        include_bias = data.get("include_bias", True)
        interaction_only = data.get("interaction_only", False)
        
        # Validate degree
        if degree < 1 or degree > 5:
            return jsonify({"error": "Degree must be between 1 and 5"}), 400

        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = PolynomialRegression(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "coefficients": safe_float_conversion(model.coef_),
            "intercept": float(model.intercept_),
            "degree": degree,
            "include_bias": include_bias,
            "interaction_only": interaction_only,
            "n_features_original": model.n_features_original,
            "n_features_poly": model.n_features_poly,
            "feature_names": model.feature_names,
            "train_metrics": regression_metrics(y_train, y_train_pred),
            "test_metrics": regression_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@regression_bp.route("/api/knn-regression", methods=["POST"])
def api_knn_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        k = data.get("k", 5)
        distance_metric = data.get("distance_metric", "euclidean")
        minkowski_p = data.get("minkowski_p", 3)
        
        # Validate k
        if k < 1:
            return jsonify({"error": "k must be at least 1"}), 400
        
        valid_metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine']
        if distance_metric not in valid_metrics:
            return jsonify({"error": f"Invalid distance metric. Must be one of: {valid_metrics}"}), 400

        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        k_actual = min(k, len(X_train))

        model = KNNRegressor(k=k_actual, distance_metric=distance_metric, minkowski_p=minkowski_p)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "k": k_actual,
            "distance_metric": distance_metric,
            "train_metrics": regression_metrics(y_train, y_train_pred),
            "test_metrics": regression_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
