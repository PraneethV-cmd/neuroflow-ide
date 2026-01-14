import React, { useMemo, useState, useEffect } from 'react';
import { Handle, Position, useStore, useReactFlow } from 'reactflow';
import './LinearRegressionNode.css';
import { FaProjectDiagram } from 'react-icons/fa';
import { parseFullTabularFile } from '../../utils/parseTabularFile';
import { trainDBSCAN } from '../../utils/apiClient';
import InfoButton from '../ui/InfoButton';

const DBSCANNode = ({ id, data, isConnectable }) => {
    const { setNodes } = useReactFlow();

    // State management
    const [selectedFeatures, setSelectedFeatures] = useState([]);
    const [eps, setEps] = useState(0.5);
    const [epsInput, setEpsInput] = useState('0.5');
    const [minSamples, setMinSamples] = useState(5);
    const [minSamplesInput, setMinSamplesInput] = useState('5');
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

    const hasNormalizer = useStore((store) => {
        const incoming = Array.from(store.edges.values()).filter((e) => e.target === id);
        for (const e of incoming) {
            const src = store.nodeInternals.get(e.source);
            if (src?.type === 'normalizer') return true;
        }
        return false;
    });

    // Toggle feature selection
    const toggleFeature = (h) => {
        setSelectedFeatures((prev) => (prev.includes(h) ? prev.filter((c) => c !== h) : [...prev, h]));
    };

    // Run DBSCAN clustering
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
        if (eps <= 0) {
            alert('Epsilon must be positive.');
            return;
        }
        if (minSamples < 1) {
            alert('Minimum samples must be at least 1.');
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

            if (X.length < minSamples) {
                throw new Error(`Not enough valid rows. Need at least ${minSamples} rows.`);
            }

            // Call backend API
            setTrainMsg(`Running DBSCAN (eps=${eps}, min_samples=${minSamples})...`);

            const result = await trainDBSCAN(X, {
                eps: eps,
                min_samples: minSamples,
                distance_metric: distanceMetric,
                minkowski_p: minkowskiP
            });

            if (result.success) {
                const clusterLabels = result.cluster_labels;
                const coreSamples = result.core_samples;
                const coreSampleLabels = result.core_sample_labels;

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
                            clusterCenters: result.cluster_representatives,
                            clusterSizes: result.cluster_sizes,
                            nClusters: result.n_clusters,
                            nNoise: result.n_noise,
                            selectedFeatures: selectedFeatures,
                            eps: eps,
                            minSamples: minSamples,
                            distanceMetric: distanceMetric,
                            // Store for Model Evaluator
                            model: {
                                selectedFeatures: selectedFeatures,
                                eps: eps,
                                minSamples: minSamples,
                                distanceMetric: distanceMetric,
                                coreSamples: coreSamples,
                                coreSampleLabels: coreSampleLabels,
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
                setTrainMsg(`DBSCAN complete! ${result.n_clusters} clusters found, ${result.n_noise} noise points.`);
                alert('DBSCAN clustering finished.');
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
            <InfoButton nodeType="dbscan" />
            <Handle type="target" position={Position.Top} isConnectable={isConnectable} style={{ background: '#555' }} />

            <div className="node-header">
                <FaProjectDiagram className="node-icon" />
                <span className="node-title">{data.label || 'DBSCAN'}</span>
            </div>

            {headers.length > 0 && (
                <div className="lr-selects">
                    {!hasNormalizer && (
                        <div style={{
                            padding: '8px',
                            background: '#fff3cd',
                            color: '#856404',
                            borderRadius: '4px',
                            fontSize: '0.8em',
                            marginBottom: '10px',
                            border: '1px solid #ffeeba'
                        }}>
                            ⚠️ DBSCAN relies on distances. Normalization is strongly recommended.
                        </div>
                    )}

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
                        <label>Epsilon (ε):</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <button
                                type="button"
                                onClick={() => {
                                    if (eps > 0.1) {
                                        const newVal = Math.max(0.1, parseFloat((eps - 0.1).toFixed(1)));
                                        setEps(newVal);
                                        setEpsInput(newVal.toString());
                                    }
                                }}
                                disabled={eps <= 0.1}
                                style={{
                                    width: '24px',
                                    height: '24px',
                                    padding: '0',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    background: '#f8f9fa',
                                    cursor: eps <= 0.1 ? 'not-allowed' : 'pointer'
                                }}
                            >
                                -
                            </button>
                            <input
                                type="number"
                                step="0.1"
                                min="0.1"
                                value={epsInput}
                                onChange={(e) => {
                                    setEpsInput(e.target.value);
                                    const val = parseFloat(e.target.value);
                                    if (!isNaN(val) && val > 0) setEps(val);
                                }}
                                onBlur={(e) => {
                                    const val = parseFloat(e.target.value);
                                    if (isNaN(val) || val <= 0) {
                                        setEpsInput(eps.toString());
                                    } else {
                                        setEpsInput(val.toString());
                                        setEps(val);
                                    }
                                }}
                                style={{ width: '60px', padding: '4px', textAlign: 'center' }}
                            />
                            <button
                                type="button"
                                onClick={() => {
                                    const newVal = parseFloat((eps + 0.1).toFixed(1));
                                    setEps(newVal);
                                    setEpsInput(newVal.toString());
                                }}
                                style={{
                                    width: '24px',
                                    height: '24px',
                                    padding: '0',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    background: '#f8f9fa',
                                    cursor: 'pointer'
                                }}
                            >
                                +
                            </button>
                        </div>
                    </div>

                    <div className="lr-row">
                        <label>Min Samples:</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <button
                                type="button"
                                onClick={() => {
                                    if (minSamples > 1) {
                                        const newVal = minSamples - 1;
                                        setMinSamples(newVal);
                                        setMinSamplesInput(newVal.toString());
                                    }
                                }}
                                disabled={minSamples <= 1}
                                style={{
                                    width: '24px',
                                    height: '24px',
                                    padding: '0',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    background: '#f8f9fa',
                                    cursor: minSamples <= 1 ? 'not-allowed' : 'pointer'
                                }}
                            >
                                -
                            </button>
                            <input
                                type="number"
                                step="1"
                                min="1"
                                value={minSamplesInput}
                                onChange={(e) => {
                                    setMinSamplesInput(e.target.value);
                                    const val = parseInt(e.target.value);
                                    if (!isNaN(val) && val >= 1) setMinSamples(val);
                                }}
                                onBlur={(e) => {
                                    const val = parseInt(e.target.value);
                                    if (isNaN(val) || val < 1) {
                                        setMinSamplesInput(minSamples.toString());
                                    } else {
                                        setMinSamplesInput(val.toString());
                                        setMinSamples(val);
                                    }
                                }}
                                style={{ width: '60px', padding: '4px', textAlign: 'center' }}
                            />
                            <button
                                type="button"
                                onClick={() => {
                                    const newVal = minSamples + 1;
                                    setMinSamples(newVal);
                                    setMinSamplesInput(newVal.toString());
                                }}
                                style={{
                                    width: '24px',
                                    height: '24px',
                                    padding: '0',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    background: '#f8f9fa',
                                    cursor: 'pointer'
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
                            <label>Minkowski p:</label>
                            <input
                                type="number"
                                step="0.5"
                                value={minkowskiPInput}
                                onChange={(e) => {
                                    setMinkowskiPInput(e.target.value);
                                    const val = parseFloat(e.target.value);
                                    if (!isNaN(val) && val >= 1) setMinkowskiP(val);
                                }}
                                onBlur={() => setMinkowskiPInput(minkowskiP.toString())}
                                style={{ width: '60px', padding: '4px' }}
                            />
                        </div>
                    )}

                    <div className="lr-actions">
                        <button className="btn primary" onClick={onRun} disabled={isRunning}>
                            {isRunning ? 'Running...' : 'Run DBSCAN'}
                        </button>
                    </div>

                    {trainMsg && <div className="lr-msg">{trainMsg}</div>}

                    {results && (
                        <div className="model-results">
                            <div style={{ fontWeight: '700', color: '#1a202c', marginBottom: '8px', paddingBottom: '4px', borderBottom: '1px solid #e1e8f0' }}>
                                DBSCAN Results
                            </div>
                            <div>Clusters Found: <strong>{results.n_clusters}</strong></div>
                            <div>Noise Points: <strong style={{ color: '#e53e3e' }}>{results.n_noise}</strong></div>

                            <div style={{ fontWeight: '700', color: '#1a202c', marginTop: '12px', marginBottom: '8px', paddingBottom: '4px', borderBottom: '1px solid #e1e8f0' }}>
                                Cluster Sizes
                            </div>
                            <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
                                {results.cluster_sizes.map((size, idx) => (
                                    <div key={idx} style={{
                                        marginBottom: '4px',
                                        padding: '4px',
                                        borderLeft: `3px solid ${clusterColors[idx % clusterColors.length]}`,
                                        paddingLeft: '8px',
                                        background: '#f7fafc',
                                        fontSize: '0.9em'
                                    }}>
                                        <strong>Cluster {idx}:</strong> {size} samples
                                    </div>
                                ))}
                                {results.n_noise > 0 && (
                                    <div style={{
                                        marginBottom: '4px',
                                        padding: '4px',
                                        borderLeft: `3px solid #718096`,
                                        paddingLeft: '8px',
                                        background: '#edf2f7',
                                        fontSize: '0.9em',
                                        color: '#4a5568'
                                    }}>
                                        <strong>Noise (-1):</strong> {results.n_noise} samples
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}

            <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} style={{ background: '#555' }} />
        </div>
    );
};

export default DBSCANNode;
