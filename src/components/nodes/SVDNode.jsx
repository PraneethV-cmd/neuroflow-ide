import React, { useMemo, useState } from 'react';
import { Handle, Position, useStore, useReactFlow } from 'reactflow';
import './SVDNode.css';
import { parseFullTabularFile } from '../../utils/parseTabularFile';
import { applySVD } from '../../utils/apiClient';

const SVDNode = ({ id, data, isConnectable }) => {
    const [selectedColumns, setSelectedColumns] = useState([]);
    const [nComponents, setNComponents] = useState(2);
    const [isProcessing, setIsProcessing] = useState(false);
    const [svdData, setSvdData] = useState(null);
    const [error, setError] = useState('');
    const { setNodes } = useReactFlow();

    // Find upstream data source
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
                    encodedRows: src.data?.encodedRows || [],
                    encodingInfo: src.data?.encodingInfo || {}
                };
            }
            if (src?.type === 'normalizer') {
                return {
                    type: 'normalized',
                    headers: src.data?.headers || [],
                    normalizedRows: src.data?.normalizedRows || [],
                    normalizationInfo: src.data?.normalizationInfo || {}
                };
            }
            if (src?.type === 'featureSelector') {
                return {
                    type: 'featureSelector',
                    headers: src.data?.selectedHeaders || [],
                    selectedRows: src.data?.selectedRows || []
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

    const toggleColumn = (header) => {
        setSelectedColumns(prev =>
            prev.includes(header)
                ? prev.filter(col => col !== header)
                : [...prev, header]
        );
    };

    const onApplySVD = async () => {
        if (!upstreamData) {
            setError('Please connect a data source.');
            return;
        }
        if (selectedColumns.length === 0) {
            setError('Please select at least one column for SVD.');
            return;
        }

        setIsProcessing(true);
        setError('');

        try {
            let rows;

            // Get data from upstream source
            if (upstreamData.type === 'csv') {
                const parsed = await parseFullTabularFile(upstreamData.file);
                rows = parsed.rows;
            } else if (upstreamData.type === 'cleaned') {
                rows = upstreamData.cleanedRows;
            } else if (upstreamData.type === 'encoded') {
                rows = upstreamData.encodedRows;
            } else if (upstreamData.type === 'normalized') {
                rows = upstreamData.normalizedRows;
            } else if (upstreamData.type === 'featureSelector') {
                rows = upstreamData.selectedRows;
            } else if (upstreamData.type === 'dataTypeConverter') {
                rows = upstreamData.rows;
            } else {
                throw new Error('Unknown data source type.');
            }

            // Extract selected columns
            const selectedIndices = selectedColumns.map(col => headers.indexOf(col));
            const data = rows.map(row => selectedIndices.map(idx => parseFloat(row[idx])));

            // Validate numeric data
            const hasNonNumeric = data.some(row => row.some(val => !Number.isFinite(val)));
            if (hasNonNumeric) {
                throw new Error('Selected columns contain non-numeric values.');
            }

            const config = {
                n_components: nComponents
            };

            // Prepare full row data for propagating unselected columns
            // Convert all rows to numeric where possible, keep original values for non-selected columns
            const fullRowsNumeric = rows.map(row =>
                row.map(val => {
                    const num = parseFloat(val);
                    return Number.isFinite(num) ? num : val;
                })
            );

            // Call API
            const result = await applySVD(data, selectedColumns, config, fullRowsNumeric, headers, selectedIndices);

            if (result.success) {
                const transformedRows = result.transformed_data;

                setSvdData({
                    headers: result.component_headers,
                    rows: transformedRows,
                    singularValues: result.singular_values || [],
                    explainedVarianceRatio: result.explained_variance_ratio || [],
                    nComponents: result.n_components
                });

                // Store SVD data in node for downstream nodes
                setNodes((nds) => nds.map((n) => {
                    if (n.id !== id) return n;
                    return {
                        ...n,
                        data: {
                            ...n.data,
                            svdHeaders: result.component_headers,
                            svdRows: transformedRows,
                            svdInfo: {
                                singularValues: result.singular_values,
                                explainedVarianceRatio: result.explained_variance_ratio,
                                nComponents: result.n_components
                            }
                        }
                    };
                }));

            } else {
                throw new Error(result.error || 'SVD transformation failed');
            }
        } catch (err) {
            setError(err?.message || 'SVD transformation failed.');
        } finally {
            setIsProcessing(false);
        }
    };

    const onClear = () => {
        setSvdData(null);
        setSelectedColumns([]);
        setError('');
        setNodes((nds) => nds.map((n) =>
            n.id === id ? { ...n, data: { ...n.data, svdHeaders: [], svdRows: [], svdInfo: {} } } : n
        ));
    };

    return (
        <div className="svd-node">
            <Handle type="target" position={Position.Top} isConnectable={isConnectable} />

            <div className="svd-header">
                <span className="svd-title">{data.label || 'SVD'}</span>
            </div>

            {headers.length > 0 && (
                <div className="svd-content">
                    {/* Column Selection */}
                    <div className="columns-section">
                        <label>Select Numeric Columns:</label>
                        <div className="column-checkboxes">
                            {headers.map((header) => (
                                <label key={header} className="column-option">
                                    <input
                                        type="checkbox"
                                        checked={selectedColumns.includes(header)}
                                        onChange={() => toggleColumn(header)}
                                    />
                                    <span>{header}</span>
                                </label>
                            ))}
                        </div>
                    </div>

                    {/* Component Count */}
                    <div className="param-section">
                        <label>Number of Components:</label>
                        <input
                            type="number"
                            min="1"
                            max={selectedColumns.length || 10}
                            value={nComponents}
                            onChange={(e) => setNComponents(parseInt(e.target.value, 10))}
                        />
                    </div>

                    {/* Actions */}
                    <div className="svd-actions">
                        <button
                            className="btn primary"
                            onClick={onApplySVD}
                            disabled={isProcessing || selectedColumns.length === 0}
                        >
                            {isProcessing ? 'Applying SVD...' : 'Apply SVD'}
                        </button>
                        {svdData && (
                            <button className="btn secondary" onClick={onClear}>
                                Clear
                            </button>
                        )}
                    </div>

                    {/* Errors */}
                    {error && <div className="error-text">{error}</div>}

                    {/* SVD Results */}
                    {svdData && (
                        <div className="svd-preview">
                            <div className="preview-title">
                                SVD Results (Rank {svdData.nComponents})
                            </div>

                            {/* Singular Values & Explained Variance */}
                            {svdData.singularValues.length > 0 && (
                                <div className="singular-values-section">
                                    <div className="variance-title">Singular Values & Explained Variance:</div>
                                    {svdData.singularValues.map((sv, idx) => (
                                        <div key={idx} className="singular-value-item">
                                            <span className="variance-label">C{idx + 1}:</span>
                                            <span className="variance-value">σ={sv.toFixed(2)} ({((svdData.explainedVarianceRatio[idx] || 0) * 100).toFixed(2)}%)</span>
                                        </div>
                                    ))}
                                    {svdData.explainedVarianceRatio.length > 0 && (
                                        <div className="variance-total">
                                            Total Explained: <strong>{(svdData.explainedVarianceRatio.reduce((a, b) => a + b, 0) * 100).toFixed(2)}%</strong>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Data Preview */}
                            <div className="preview-title">
                                Transformed Data Preview (showing {Math.min(5, svdData.rows.length)} of {svdData.rows.length} rows)
                                {svdData.rows.length > 5 && (
                                    <span style={{ fontSize: '0.75em', color: '#2563eb', fontWeight: 500, marginLeft: '8px' }}>
                                        Right-click node to view all
                                    </span>
                                )}
                            </div>
                            <div className="table-scroll">
                                <table>
                                    <thead>
                                        <tr>
                                            {svdData.headers.map((header, idx) => (
                                                <th key={idx}>{header}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {svdData.rows.slice(0, 5).map((row, rIdx) => (
                                            <tr key={rIdx}>
                                                {row.map((val, cIdx) => (
                                                    <td key={cIdx}>
                                                        {typeof val === 'number' ? val.toFixed(4) : String(val ?? '')}
                                                    </td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            )}

            <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
        </div>
    );
};

export default SVDNode;
