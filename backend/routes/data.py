from flask import Blueprint, request, jsonify
import numpy as np
from datetime import datetime

data_bp = Blueprint('data', __name__)

@data_bp.route("/api/describe", methods=["POST"])
def api_describe():
    try:
        request_data = request.json
        rows = request_data.get("data", [])
        headers = request_data.get("headers", [])
        selected_columns = request_data.get("selected_columns", [])

        if not rows:
            return jsonify({"error": "No data provided"}), 400

        try:
           data_arr = np.array(rows, dtype=object)
        except Exception as e:
             return jsonify({"error": f"Failed to parse data: {str(e)}"}), 400
             
        n_rows, n_cols = data_arr.shape
        
        if not headers:
            headers = [f"Col_{i}" for i in range(n_cols)]
            
        stats = []
        
        for i in range(n_cols):
            col_name = headers[i] if i < len(headers) else f"Column_{i}"
            
            if selected_columns and col_name not in selected_columns:
                continue
                
            col_data = data_arr[:, i]
            
            is_numeric = False
            numeric_values = []
            
            try:
                non_empty = [x for x in col_data if x is not None and str(x).strip() != '']
                if not non_empty:
                     is_numeric = False
                else:
                    valid_float_count = 0
                    temp_numeric_values = []
                    
                    for x in non_empty:
                        try:
                            val = float(x)
                            temp_numeric_values.append(val)
                            valid_float_count += 1
                        except ValueError:
                            continue
                    
                    if len(non_empty) > 0:
                        ratio = valid_float_count / len(non_empty)
                        if ratio >= 0.7: 
                            is_numeric = True
                            numeric_values = temp_numeric_values
                        else:
                            is_numeric = False
                    else:
                        is_numeric = False
            except Exception as e:
                print(f"DEBUG: Exception in column {col_name}: {e}")
                is_numeric = False
            
            col_stats = {
                "name": col_name,
                "type": "numeric" if is_numeric else "object"
            }
            
            if is_numeric:
                vals = np.array(numeric_values)
                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    variance_val = np.var(vals)
                    
                    if std_val > 1e-10:
                        z_scores = (vals - mean_val) / std_val
                        skewness = np.mean(z_scores ** 3)
                        kurtosis = np.mean(z_scores ** 4) - 3
                    else:
                        skewness = 0.0
                        kurtosis = 0.0
                    
                    unique_vals, counts = np.unique(vals, return_counts=True)
                    top_idx = np.argmax(counts)
                    num_unique = int(len(unique_vals))
                    top_val = float(unique_vals[top_idx])
                    freq_val = int(counts[top_idx])

                    col_stats.update({
                        "count": int(len(vals)),
                        "mean": float(mean_val),
                        "std": float(std_val),
                        "min": float(np.min(vals)),
                        "25%": float(np.percentile(vals, 25)),
                        "50%": float(np.median(vals)),
                        "75%": float(np.percentile(vals, 75)),
                        "max": float(np.max(vals)),
                        "variance": float(variance_val),
                        "skewness": float(skewness),
                        "kurtosis": float(kurtosis),
                        "unique": num_unique,
                        "top": top_val,
                        "freq": freq_val
                    })
                else:
                    col_stats.update({"count": 0, "note": "Empty numeric column"})
            else:
                non_empty = [str(x) for x in col_data if x is not None and str(x).strip() != '']
                col_stats.update({
                    "count": len(non_empty),
                    "unique": 0,
                    "top": "-",
                    "freq": 0
                })
                
                if len(non_empty) > 0:
                    unique_vals, counts = np.unique(non_empty, return_counts=True)
                    top_idx = np.argmax(counts)
                    col_stats["unique"] = int(len(unique_vals))
                    col_stats["top"] = str(unique_vals[top_idx])
                    col_stats["freq"] = int(counts[top_idx])
            
            stats.append(col_stats)
            
        return jsonify({
            "success": True,
            "statistics": stats
        })

    except Exception as e:
        print(f"Error in /api/describe: {e}")
        return jsonify({"error": str(e)}), 500


@data_bp.route("/api/convert_dtypes", methods=["POST"])
def api_convert_dtypes():
    try:
        request_data = request.json
        rows = request_data.get("data", [])
        headers = request_data.get("headers", [])
        conversions = request_data.get("conversions", {})
        date_formats = request_data.get("date_formats", {})

        if not rows:
            return jsonify({"error": "No data provided"}), 400

        if not conversions:
            return jsonify({
                "success": True,
                "data": rows,
                "headers": headers,
                "conversion_metadata": {}
            })
        
        n_rows = len(rows)
        n_cols = len(headers)
        col_indices = {h: i for i, h in enumerate(headers)}
        
        ops_by_index = {}
        for col_name, target_type in conversions.items():
            if col_name in col_indices:
                idx = col_indices[col_name]
                ops_by_index[idx] = {
                    "type": target_type,
                    "format": date_formats.get(col_name)
                }
        
        conversion_stats = {col: {"success": 0, "failed": 0} for col in conversions}
        converted_rows = []

        for row in rows:
            new_row = list(row)
            for idx, op in ops_by_index.items():
                val = new_row[idx]
                target_type = op["type"]
                fmt = op.get("format")
                
                try:
                    converted_val = val
                    if val is None or (isinstance(val, str) and val.strip() == ''):
                        converted_val = None
                    elif target_type == 'integer':
                        try:
                            f_val = float(val)
                            converted_val = int(f_val)
                        except:
                            converted_val = None
                    elif target_type == 'float':
                        converted_val = float(val)
                    elif target_type == 'string':
                        converted_val = str(val)
                    elif target_type == 'boolean':
                        s_val = str(val).lower().strip()
                        if s_val in ['true', '1', 'yes', 'y', 'on']:
                            converted_val = True
                        elif s_val in ['false', '0', 'no', 'n', 'off']:
                            converted_val = False
                        else:
                             converted_val = bool(val) if isinstance(val, (bool, int, float)) else None
                    elif target_type == 'datetime':
                        s_val = str(val).strip()
                        if fmt and fmt != 'auto':
                            py_fmt = fmt.replace('YYYY', '%Y').replace('MM', '%m').replace('DD', '%d')
                            dt = datetime.strptime(s_val, py_fmt)
                        else:
                            for f in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d']:
                                try:
                                    dt = datetime.strptime(s_val, f)
                                    break
                                except:
                                    continue
                            else:
                                raise ValueError("Could not parse date")
                        converted_val = dt.isoformat()
                    
                    new_row[idx] = converted_val
                    conversion_stats[headers[idx]]["success"] += 1
                    
                except Exception:
                    new_row[idx] = None
                    conversion_stats[headers[idx]]["failed"] += 1
            converted_rows.append(new_row)

        return jsonify({
            "success": True,
            "data": converted_rows,
            "headers": headers,
            "conversion_metadata": conversion_stats
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
