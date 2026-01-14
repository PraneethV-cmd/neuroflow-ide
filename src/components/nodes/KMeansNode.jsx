import React, { useMemo, useState, useEffect } from 'react';
import { Handle, Position, useStore, useReactFlow } from 'reactflow';
import './LinearRegressionNode.css';
import { FaProjectDiagram } from 'react-icons/fa';
import { parseFullTabularFile } from '../../utils/parseTabularFile';
import { trainKMeans } from '../../utils/apiClient';
import InfoButton from '../ui/InfoButton';

const KMeansNode = ({ id, data, isConnectable }) => {
    const { setNodes } = useReactFlow();

    // State management
    const [selectedFeatures, setSelectedFeatures] = useState([]);
    const [nClusters, setNClusters] = useState(3);
    const [nClustersInput, setNClustersInput] = useState('3');
    const [maxIterations, setMaxIterations] = useState(300);
    const [tolerance, setTolerance] = useState(0.0001);
    const [randomState, setRandomState] = useState(42);
    const [distanceMetric, setDistanceMetric] = useState('euclidean');
    const [minkowskiP, setMinkowskiP] = useState(3);
    const [minkowskiPInput, setMinkowskiPInput] = useState('3');

    const [isRunning, setIsRunning] = useState(false);
    const [trainMsg, setTrainMsg] = useState('');
    const [results, setResults] = useState(null);

    // Get upstream data
    const upstreamData = useStore((store) => {
        const incoming = Array.from(store.edges.values()).filter((e) => e.target === id);
        for (const e of incoming) {
            const src = store.nodeInternals.get(e.source);
            if (src?.type === 'csvReader') {
                return {
                    type: 'csv',
                    headers: src.data?.headers || [],
                    file: src.data?.file
                };
            }
            if (src?.type === 'databaseReader') {
                return {
                    type: 'database',
                    headers: src.data?.headers || [],
                    rows: src.data?.rows || []
                };
            }
            if (src?.type === 'dataCleaner') {
                return {
                    type: 'cleaned',
                    headers: src.data?.headers || [],
                    cleanedRows: src.data?.cleanedRows || []
                };
            }
            if (src?.type === 'encoder') {
                return {
                    type: 'encoded',
                    headers: src.data?.headers || [],
                    encodedRows: src.data?.encodedRows || []
                };
            }
            if (src?.type === 'normalizer') {
                return {
                    type: 'normalized',
                    headers: src.data?.headers || [],
                    normalizedRows: src.data?.normalizedRows || []
                };
            }
            if (src?.type === 'featureSelector') {
                return {
                    type: 'featureSelector',
                    headers: src.data?.selectedHeaders || [],
                    selectedRows: src.data?.selectedRows || []
                };
            }
            if (src?.type === 'pca') {
                return {
                    type: 'pca',
                    headers: src.data?.pcaHeaders || [],
                    pcaRows: src.data?.pcaRows || []
                };
            }
            if (src?.type === 'dataTypeConverter') {
                return {
                    type: 'dataTypeConverter',
                    headers: src.data?.headers || [],
                    rows: src.data?.convertedRows || []
                };
            }
        }
        return null;
    });

    const headers = useMemo(() => upstreamData?.headers || [], [upstreamData]);

    // Toggle feature selection
    const toggleFeature = (h) => {
        setSelectedFeatures((prev) => (prev.includes(h) ? prev.filter((c) => c !== h) : [...prev, h]));
    };

    // Run K-Means clustering
    const onRun = async () => {
        setTrainMsg('');
        if (!upstreamData) {
            alert('Please connect a data source node.');
            return;
        }
        if (selectedFeatures.length === 0) {
            alert('Please select at least one feature column.');
            return;
        }
        if (nClusters < 2 || nClusters > 20) {
            alert('Number of clusters must be between 2 and 20.');
            return;
        }

        setIsRunning(true);

        try {
            let rows;

            if (upstreamData.type === 'csv') {
                const parsed = await parseFullTabularFile(upstreamData.file);
                rows = parsed.rows;
            } else if (upstreamData.type === 'database') {
                rows = upstreamData.rows;
            } else if (upstreamData.type === 'cleaned') {
                rows = upstreamData.cleanedRows;
            } else if (upstreamData.type === 'encoded') {
                rows = upstreamData.encodedRows;
            } else if (upstreamData.type === 'normalized') {
                rows = upstreamData.normalizedRows;
            } else if (upstreamData.type === 'featureSelector') {
                rows = upstreamData.selectedRows;
            } else if (upstreamData.type === 'pca') {
                rows = upstreamData.pcaRows;
            } else if (upstreamData.type === 'dataTypeConverter') {
                rows = upstreamData.rows;
            } else {
                throw new Error('Unknown data source type.');
            }

            if (nClusters >= rows.length) {
                alert(`Number of clusters (${nClusters}) must be less than number of samples (${rows.length}).`);
                setIsRunning(false);
                return;
            }

            // Get indices of selected features
            const featureIndices = selectedFeatures.map((c) => headers.indexOf(c));
            if (featureIndices.some((i) => i === -1)) throw new Error('Selected features not found.');

            // Extract feature matrix
            const X = [];
            for (const r of rows) {
                const xRow = [];
                let valid = true;
                for (const i of featureIndices) {
                    const v = parseFloat(r[i]);
                    if (!Number.isFinite(v)) { valid = false; break; }
                    xRow.push(v);
                }
                if (!valid) continue;
                X.push(xRow);
            }

            if (X.length < nClusters) {
                throw new Error(`Not enough valid rows. Need at least ${nClusters} rows for ${nClusters} clusters.`);
            }

            // Call backend API
            setTrainMsg(`Running K-Means clustering (K=${nClusters}, metric=${distanceMetric})...`);

            const result = await trainKMeans(X, nClusters, {
                max_iters: maxIterations,
                tol: tolerance,
                random_state: randomState,
                distance_metric: distanceMetric,
                minkowski_p: minkowskiP
            });

            if (result.success) {
                const clusterLabels = result.cluster_labels;
                const clusterCenters = result.cluster_centers;

                // Add cluster labels to dataset
                const clusteredData = rows.map((row, idx) => [...row, clusterLabels[idx]]);
                const clusteredHeaders = [...headers, 'cluster_label'];

                // Update node data
                setNodes((nds) => nds.map((n) => {
                    if (n.id !== id) return n;
                    return {
                        ...n,
                        data: {
                            ...n.data,
                            clusteredData: clusteredData,
                            clusteredHeaders: clusteredHeaders,
                            clusterLabels: clusterLabels,
                            clusterCenters: clusterCenters,
                            clusterSizes: result.cluster_sizes,
                            inertia: result.inertia,
                            nIterations: result.n_iterations,
                            converged: result.converged,
                            selectedFeatures: selectedFeatures,
                            nClusters: nClusters,
                            distanceMetric: distanceMetric,
                            // Store for Model Evaluator
                            model: {
                                selectedFeatures: selectedFeatures,
                                clusterCenters: clusterCenters,
                                nClusters: nClusters,
                                distanceMetric: distanceMetric,
                                X_mean: result.X_mean,
                                X_std: result.X_std,
                                minkowskiP: minkowskiP
                            },
                            X_mean: result.X_mean,
                            X_std: result.X_std,
                            minkowskiP: minkowskiP,
                            pcaData: result.pca_data
                        }
                    };
                }));

                setResults(result);
                setTrainMsg(`Clustering complete! ${result.n_clusters} clusters formed in ${result.n_iterations} iterations.`);
                alert('K-Means clustering finished.');
            } else {
                throw new Error(result.error || 'Clustering failed');
            }
        } catch (err) {
            setTrainMsg(err?.message || 'Clustering failed.');
            alert(err?.message || 'Clustering failed.');
        } finally {
            setIsRunning(false);
        }
    };

    const distanceMetricOptions = [
        { value: 'euclidean', label: 'Euclidean Distance' },
        { value: 'manhattan', label: 'Manhattan Distance' },
        { value: 'minkowski', label: 'Minkowski Distance' },
        { value: 'chebyshev', label: 'Chebyshev Distance' },
        { value: 'cosine', label: 'Cosine Similarity Distance' }
    ];

    // Cluster colors
    const clusterColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'];

    return (
        <div className="linear-regression-node">
            <InfoButton nodeType="kmeans" />
            <Handle type="target" position={Position.Top} isConnectable={isConnectable} style={{ background: '#555' }} />

            <div className="node-header">
                <FaProjectDiagram className="node-icon" />
                <span className="node-title">{data.label || 'K-Means Clustering'}</span>
            </div>

            {headers.length > 0 && (
                <div className="lr-selects">
                    <div className="lr-row">
                        <label>Feature Columns (X):</label>
                        <div className="mlr-columns">
                            {headers.map((h) => (
                                <label key={h} className="mlr-option">
                                    <input type="checkbox" checked={selectedFeatures.includes(h)} onChange={() => toggleFeature(h)} />
                                    <span>{h}</span>
                                </label>
                            ))}
                        </div>
                    </div>

                    <div className="lr-row">
                        <label>Number of Clusters (K):</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <button
                                type="button"
                                onClick={() => {
                                    if (nClusters > 2) {
                                        const newVal = nClusters - 1;
                                        setNClusters(newVal);
                                        setNClustersInput(newVal.toString());
                                    }
                                }}
                                disabled={nClusters <= 2}
                                style={{
                                    width: '24px',
                                    height: '24px',
                                    padding: '0',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    background: '#f8f9fa',
                                    cursor: nClusters <= 2 ? 'not-allowed' : 'pointer'
                                }}
                            >
                                -
                            </button>
                            <input
                                type="number"
                                min="2"
                                max="20"
                                step="1"
                                value={nClustersInput}
                                onChange={(e) => {
                                    const val = e.target.value;
                                    setNClustersInput(val);
                                    const numVal = parseInt(val, 10);
                                    if (!isNaN(numVal) && numVal >= 2 && numVal <= 20) {
                                        setNClusters(numVal);
                                    }
                                }}
                                onBlur={(e) => {
                                    const numVal = parseInt(e.target.value, 10);
                                    if (isNaN(numVal) || numVal < 2 || numVal > 20) {
                                        setNClustersInput(nClusters.toString());
                                    } else {
                                        setNClustersInput(numVal.toString());
                                        setNClusters(numVal);
                                    }
                                }}
                                style={{ width: '60px', padding: '4px', textAlign: 'center' }}
                            />
                            <button
                                type="button"
                                onClick={() => {
                                    if (nClusters < 20) {
                                        const newVal = nClusters + 1;
                                        setNClusters(newVal);
                                        setNClustersInput(newVal.toString());
                                    }
                                }}
                                disabled={nClusters >= 20}
                                style={{
                                    width: '24px',
                                    height: '24px',
                                    padding: '0',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    background: '#f8f9fa',
                                    cursor: nClusters >= 20 ? 'not-allowed' : 'pointer'
                                }}
                            >
                                +
                            </button>
                        </div>
                    </div>

                    <div className="lr-row">
                        <label>Distance Metric:</label>
                        <select value={distanceMetric} onChange={(e) => setDistanceMetric(e.target.value)}>
                            {distanceMetricOptions.map((option) => (
                                <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                        </select>
                    </div>

                    {distanceMetric === 'minkowski' && (
                        <div className="lr-row">
                            <label>Minkowski p-value:</label>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <button
                                    type="button"
                                    onClick={() => {
                                        if (minkowskiP > 1) {
                                            const newVal = minkowskiP - 0.5;
                                            setMinkowskiP(newVal);
                                            setMinkowskiPInput(newVal.toString());
                                        }
                                    }}
                                    disabled={minkowskiP <= 1}
                                    style={{
                                        width: '24px',
                                        height: '24px',
                                        padding: '0',
                                        border: '1px solid #ccc',
                                        borderRadius: '3px',
                                        background: '#f8f9fa',
                                        cursor: minkowskiP <= 1 ? 'not-allowed' : 'pointer'
                                    }}
                                >
                                    -
                                </button>
                                <input
                                    type="number"
                                    min="1"
                                    max="10"
                                    step="0.5"
                                    value={minkowskiPInput}
                                    onChange={(e) => {
                                        const val = e.target.value;
                                        setMinkowskiPInput(val);
                                        const numVal = parseFloat(val);
                                        if (!isNaN(numVal) && numVal >= 1 && numVal <= 10) {
                                            setMinkowskiP(numVal);
                                        }
                                    }}
                                    onBlur={(e) => {
                                        const numVal = parseFloat(e.target.value);
                                        if (isNaN(numVal) || numVal < 1 || numVal > 10) {
                                            setMinkowskiPInput(minkowskiP.toString());
                                        } else {
                                            setMinkowskiPInput(numVal.toString());
                                            setMinkowskiP(numVal);
                                        }
                                    }}
                                    style={{ width: '60px', padding: '4px', textAlign: 'center' }}
                                />
                                <button
                                    type="button"
                                    onClick={() => {
                                        if (minkowskiP < 10) {
                                            const newVal = minkowskiP + 0.5;
                                            setMinkowskiP(newVal);
                                            setMinkowskiPInput(newVal.toString());
                                        }
                                    }}
                                    disabled={minkowskiP >= 10}
                                    style={{
                                        width: '24px',
                                        height: '24px',
                                        padding: '0',
                                        border: '1px solid #ccc',
                                        borderRadius: '3px',
                                        background: '#f8f9fa',
                                        cursor: minkowskiP >= 10 ? 'not-allowed' : 'pointer'
                                    }}
                                >
                                    +
                                </button>
                                <span style={{ marginLeft: '4px', fontSize: '0.85em', color: '#666' }}>
                                    (p=1: Manhattan, p=2: Euclidean)
                                </span>
                            </div>
                        </div>
                    )}

                    <div className="lr-actions">
                        <button className="btn primary" onClick={onRun} disabled={isRunning}>
                            {isRunning ? 'Running...' : 'Run Clustering'}
                        </button>
                    </div>

                    {trainMsg && <div className="lr-msg">{trainMsg}</div>}

                    {results && (
                        <div className="model-results">
                            <div style={{ fontWeight: '700', color: '#1a202c', marginBottom: '8px', paddingBottom: '4px', borderBottom: '1px solid #e1e8f0' }}>
                                Clustering Results
                            </div>
                            <div>Clusters: <strong>{results.n_clusters}</strong></div>
                            <div>Iterations: <strong>{results.n_iterations}</strong></div>
                            <div>Inertia: <strong>{results.inertia.toFixed(2)}</strong></div>
                            <div>Converged: <strong>{results.converged ? '✓ Yes' : '✗ No'}</strong></div>
                            <div>Distance Metric: <strong>{distanceMetricOptions.find(o => o.value === results.distance_metric)?.label}</strong></div>

                            <div style={{ fontWeight: '700', color: '#1a202c', marginTop: '12px', marginBottom: '8px', paddingBottom: '4px', borderBottom: '1px solid #e1e8f0' }}>
                                Cluster Centers
                            </div>
                            {results.cluster_centers.map((center, idx) => (
                                <div key={idx} style={{
                                    marginBottom: '4px',
                                    padding: '4px',
                                    borderLeft: `3px solid ${clusterColors[idx % clusterColors.length]}`,
                                    paddingLeft: '8px',
                                    background: '#f7fafc'
                                }}>
                                    <strong>Cluster {idx}:</strong> [{center.map(v => v.toFixed(3)).join(', ')}]
                                    <span style={{ marginLeft: '8px', color: '#666', fontSize: '0.85em' }}>
                                        ({results.cluster_sizes[idx]} points)
                                    </span>
                                </div>
                            ))}

                            <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid #e1e8f0', fontSize: '0.8rem', color: '#718096' }}>
                                <div>Total Samples: <strong>{results.cluster_sizes.reduce((a, b) => a + b, 0)}</strong></div>
                                <div style={{ marginTop: '4px', fontSize: '0.75em', color: '#a0aec0' }}>
                                    💡 Tip: Normalizing features before clustering is recommended
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} style={{ background: '#555' }} />
        </div>
    );
};

export default KMeansNode;
