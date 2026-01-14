"""
Metrics for regression and classification
"""
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
    """Calculate classification metrics (Accuracy and Macro-averaged Precision/Recall/F1)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Replace any NaN values
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Accuracy
    accuracy = float(np.mean(y_true == y_pred))
    
    # Calculate macro-averaged metrics
    precisions = []
    recalls = []
    
    # Confusion matrix accumulator
    conf_matrix_custom = {}
    
    # Binary case detection for simple confusion matrix return
    is_binary = len(classes) <= 2 and set(classes).issubset({0, 1})
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0
    
    for cls in classes:
        # Binary arrays for this class vs All
        y_true_cls = (y_true == cls)
        y_pred_cls = (y_pred == cls)
        
        tp = np.sum(y_true_cls & y_pred_cls)
        fp = np.sum((~y_true_cls) & y_pred_cls)
        fn = np.sum(y_true_cls & (~y_pred_cls))
        
        # Precision for this class
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(p)
        
        # Recall for this class
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(r)
        
        # Keep track if binary for legacy return format
        if is_binary and cls == 1:
            tp_total, fp_total, fn_total = tp, fp, fn
            tn_total = np.sum((~y_true_cls) & (~y_pred_cls))

    # Macro Averages
    precision_macro = float(np.mean(precisions)) if precisions else 0.0
    recall_macro = float(np.mean(recalls)) if recalls else 0.0
    f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro) if (precision_macro + recall_macro) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1_score": f1_macro,
        "confusion_matrix": {
            # For multi-class, this simple structure is ambiguous, but keeping for binary compat
            "true_negatives": int(tn_total) if is_binary else 0,
            "false_positives": int(fp_total) if is_binary else 0,
            "false_negatives": int(fn_total) if is_binary else 0,
            "true_positives": int(tp_total) if is_binary else 0
        }
    }
