from flask import Blueprint, request, jsonify
import numpy as np
from backend.utils.preprocessing import train_test_split, safe_float_conversion
from backend.utils.metrics import classification_metrics
from backend.models.classification import LogisticRegression, KNNClassifier, GaussianNaiveBayes

classification_bp = Blueprint('classification', __name__)

@classification_bp.route('/api/logistic-regression', methods=['POST'])
def api_logistic_regression():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"])

        train_percent = data.get("train_percent", 80)
        test_size = 1 - train_percent/100

        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            return jsonify({"error": "Logistic regression requires binary labels"}), 400

        if not np.array_equal(unique_labels, [0, 1]):
            low = unique_labels.min()
            y = (y != low).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        learning_rate = data.get("learning_rate", 0.1)
        n_iterations = data.get("n_iterations", 10000)
        
        model = LogisticRegression(
            C=data.get("C", 1.0),
            learning_rate=learning_rate,
            n_iterations=n_iterations
        )
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        return jsonify({
            "success": True,
            "coefficients": safe_float_conversion(model.coef_),
            "intercept": float(model.intercept_),
            "train_metrics": classification_metrics(y_train, y_train_pred),
            "test_metrics": classification_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "test_probabilities": safe_float_conversion(y_test_proba),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@classification_bp.route("/api/knn-classification", methods=["POST"])
def api_knn_classification():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        k = data.get("k", 5)
        distance_metric = data.get("distance_metric", "euclidean")
        minkowski_p = data.get("minkowski_p", 3)
        
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

        model = KNNClassifier(k=k_actual, distance_metric=distance_metric, minkowski_p=minkowski_p)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return jsonify({
            "success": True,
            "k": k_actual,
            "distance_metric": distance_metric,
            "classes": [int(c) for c in model.classes],
            "train_metrics": classification_metrics(y_train, y_train_pred),
            "test_metrics": classification_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@classification_bp.route("/api/naive-bayes", methods=["POST"])
def api_naive_bayes():
    try:
        data = request.json
        X = np.array(data["X"], dtype=float)
        y = np.array(data["y"], dtype=float)

        train_percent = data.get("train_percent", 80)
        var_smoothing = data.get("var_smoothing", 1e-9)

        test_size = max(0.1, min(0.5, 1 - train_percent/100))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = GaussianNaiveBayes(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        return jsonify({
            "success": True,
            "classes": safe_float_conversion(model.classes_),
            "train_metrics": classification_metrics(y_train, y_train_pred),
            "test_metrics": classification_metrics(y_test, y_test_pred),
            "train_predictions": safe_float_conversion(y_train_pred),
            "test_predictions": safe_float_conversion(y_test_pred),
            "test_probabilities": safe_float_conversion(y_test_proba),
            "train_size": len(X_train),
            "test_size": len(X_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
