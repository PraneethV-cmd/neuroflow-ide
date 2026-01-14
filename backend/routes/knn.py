from flask import Blueprint, request, jsonify
import numpy as np
from backend.models.knn import KNNRegressor, KNNClassifier
from backend.utils.preprocessing import safe_float_conversion, train_test_split
from backend.utils.metrics import regression_metrics, classification_metrics

knn_bp = Blueprint('knn', __name__)



@knn_bp.route("/api/knn-regression", methods=["POST"])
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
            
        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Ensure k is not larger than training set
        k_actual = min(k, len(X_train))
        
        # Configure model
        model = KNNRegressor(
            k=k_actual,
            distance_metric=distance_metric,
            minkowski_p=minkowski_p
        )
        
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = regression_metrics(y_train, y_train_pred)
        test_metrics = regression_metrics(y_test, y_test_pred)

        return jsonify({
            "success": True,
            "k": k_actual,
            "distance_metric": distance_metric,
            "train_metrics": {
                "MSE": train_metrics["mse"],
                "RMSE": train_metrics["rmse"],
                "R2": train_metrics["r2_score"]
            },
            "test_metrics": {
                "MSE": test_metrics["mse"],
                "RMSE": test_metrics["rmse"],
                "R2": test_metrics["r2_score"]
            },
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@knn_bp.route("/api/knn-classification", methods=["POST"])
def api_knn_classification():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)  # Assuming encoded labels 0, 1, 2...

        train_percent = data.get("train_percent", 80)
        k = data.get("k", 5)
        distance_metric = data.get("distance_metric", "euclidean")
        minkowski_p = data.get("minkowski_p", 3)
        
        # Validate k
        if k < 1:
            return jsonify({"error": "k must be at least 1"}), 400
            
        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Ensure k is not larger than training set
        k_actual = min(k, len(X_train))

        # Configure model
        model = KNNClassifier(
            k=k_actual,
            distance_metric=distance_metric,
            minkowski_p=minkowski_p
        )
        
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics


        return jsonify({
            "success": True,
            "k": k_actual,
            "distance_metric": distance_metric,
            "classes": safe_float_conversion(model.classes),
            "train_metrics": {
                "Accuracy": classification_metrics(y_train, y_train_pred)["accuracy"],
                "Precision": classification_metrics(y_train, y_train_pred)["precision"],
                "Recall": classification_metrics(y_train, y_train_pred)["recall"],
                "F1": classification_metrics(y_train, y_train_pred)["f1_score"]
            },
            "test_metrics": {
                "Accuracy": classification_metrics(y_test, y_test_pred)["accuracy"],
                "Precision": classification_metrics(y_test, y_test_pred)["precision"],
                "Recall": classification_metrics(y_test, y_test_pred)["recall"],
                "F1": classification_metrics(y_test, y_test_pred)["f1_score"]
            },
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
