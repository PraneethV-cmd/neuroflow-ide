import numpy as np

def regression_metrics(y_true, y_pred):
    """Calculate regression metrics without sklearn"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Replace any NaN values
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # Handle edge cases
    if len(y_true) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2_score": 0.0, "mape": 0.0}
    
    # MSE
    mse = float(np.mean((y_true - y_pred) ** 2))
    
    # RMSE
    rmse = float(np.sqrt(max(0, mse)))  # Ensure non-negative
    
    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # R² Score
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    
    if abs(ss_tot) < 1e-10:
        r2 = 1.0 if abs(ss_res) < 1e-10 else 0.0
    else:
        r2 = max(-1.0, min(1.0, 1 - (ss_res / ss_tot)))  # Clamp between -1 and 1
    
    # MAPE
    denom = np.where(np.abs(y_true) < 1e-10, 1e-10, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mape": mape
    }

def classification_metrics(y_true, y_pred):
    """Calculate classification metrics without sklearn"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Replace any NaN values
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # Ensure binary labels
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
