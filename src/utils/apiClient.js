// API client for backend communication
const API_BASE_URL = 'http://localhost:5000/api';

export async function trainLogisticRegression(X, y, trainPercent, featureNames = [], targetName = 'target', options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}/logistic-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        learning_rate: options.learningRate,
        n_iterations: options.maxIterations,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainLinearRegression(X, y, trainPercent, featureName = 'X', targetName = 'y', options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}/linear-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        learning_rate: options.learningRate,
        n_iterations: options.maxIterations,
        feature_name: featureName,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainMultiLinearRegression(X, y, trainPercent, featureNames = [], targetName = 'y', options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}/multi-linear-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        learning_rate: options.learningRate,
        n_iterations: options.maxIterations,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainKNNRegression(X, y, trainPercent, k, distanceMetric, featureNames = [], targetName = 'y', minkowskiP = 3) {
  try {
    const response = await fetch(`${API_BASE_URL}/knn-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        k,
        distance_metric: distanceMetric,
        minkowski_p: minkowskiP,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainKNNClassification(X, y, trainPercent, k, distanceMetric, featureNames = [], targetName = 'y', minkowskiP = 3) {
  try {
    const response = await fetch(`${API_BASE_URL}/knn-classification`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        k,
        distance_metric: distanceMetric,
        minkowski_p: minkowskiP,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainNaiveBayes(X, y, trainPercent, alpha = 1.0, featureNames = [], targetName = 'y') {
  try {
    const response = await fetch(`${API_BASE_URL}/naive-bayes`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        alpha,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainPolynomialRegression(X, y, trainPercent, degree, includeBias, interactionOnly, featureNames = [], targetName = 'y') {
  try {
    const response = await fetch(`${API_BASE_URL}/polynomial-regression`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        X,
        y,
        train_percent: trainPercent,
        degree,
        include_bias: includeBias,
        interaction_only: interactionOnly,
        feature_names: featureNames,
        target_name: targetName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function applyPCA(data, headers, config, fullRows = null, allHeaders = null, selectedIndices = null) {
  try {
    const requestBody = {
      data,
      headers,
      n_components: config.n_components,
      variance_threshold: config.variance_threshold,
      standardize: config.standardize !== undefined ? config.standardize : true,
      return_loadings: config.return_loadings || false,
      return_explained_variance: config.return_explained_variance !== undefined ? config.return_explained_variance : true
    };

    // Add optional full row data for propagating unselected columns
    if (fullRows !== null && allHeaders !== null && selectedIndices !== null) {
      requestBody.full_rows = fullRows;
      requestBody.all_headers = allHeaders;
      requestBody.selected_indices = selectedIndices;
    }

    const response = await fetch(`${API_BASE_URL}/pca`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function applySVD(data, headers, config, fullRows = null, allHeaders = null, selectedIndices = null) {
  try {
    const requestBody = {
      data,
      headers,
      n_components: config.n_components
    };

    // Add optional full row data for propagating unselected columns
    if (fullRows !== null && allHeaders !== null && selectedIndices !== null) {
      requestBody.full_rows = fullRows;
      requestBody.all_headers = allHeaders;
      requestBody.selected_indices = selectedIndices;
    }

    const response = await fetch(`${API_BASE_URL}/svd`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}


// Database Reader API functions
export async function testDatabaseConnection(params) {
  try {
    const response = await fetch(`${API_BASE_URL}/database/test-connection`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      const errorData = await response.json();
      return { success: false, error: errorData.error || `HTTP error! status: ${response.status}` };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    return { success: false, error: `API call failed: ${error.message}` };
  }
}

export async function fetchDatabaseTables(params) {
  try {
    const response = await fetch(`${API_BASE_URL}/database/fetch-tables`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      const errorData = await response.json();
      return { success: false, error: errorData.error || `HTTP error! status: ${response.status}` };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    return { success: false, error: `API call failed: ${error.message}` };
  }
}

export async function previewDatabaseData(params) {
  try {
    const response = await fetch(`${API_BASE_URL}/database/preview-data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      const errorData = await response.json();
      return { success: false, error: errorData.error || `HTTP error! status: ${response.status}` };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    return { success: false, error: `API call failed: ${error.message}` };
  }
}

export async function loadDatabaseData(params) {
  try {
    const response = await fetch(`${API_BASE_URL}/database/load-data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      const errorData = await response.json();
      return { success: false, error: errorData.error || `HTTP error! status: ${response.status}` };
    }

    const data = await response.json();
    return data;
  } catch (error) {
    return { success: false, error: `API call failed: ${error.message}` };
  }
}

export async function describeData(data, headers, selectedColumns = []) {
  try {
    const response = await fetch(`${API_BASE_URL}/describe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        data,
        headers,
        selected_columns: selectedColumns
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}


export async function convertDataTypes(data, headers, conversions, dateFormats = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}/convert_dtypes`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        data,
        headers,
        conversions,
        date_formats: dateFormats
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    throw new Error(`API call failed: ${error.message}`);
  }
}

export async function trainKMeans(X, nClusters, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}/kmeans`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        X,
        n_clusters: nClusters,
        ...options
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(`K-Means API call failed: ${error.message}`);
  }
}

export async function trainHierarchicalClustering(X, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}/hierarchical-clustering`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        X,
        ...options
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(`Hierarchical Clustering API call failed: ${error.message}`);
  }
}

export async function trainDBSCAN(X, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}/dbscan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        X,
        ...options
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(`DBSCAN API call failed: ${error.message}`);
  }
}

export async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      return false;
    }
    const data = await response.json();
    return data.status === 'ok';
  } catch (error) {
    return false;
  }
}
