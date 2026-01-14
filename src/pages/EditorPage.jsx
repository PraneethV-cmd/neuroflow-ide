import React, { useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, {
  MiniMap,
  useReactFlow,
  SmoothStepEdge,
  useViewport,
} from 'reactflow';
import 'reactflow/dist/style.css';

import TopToolbar from '../components/ui/TopToolbar';
import BottomToolbar from '../components/ui/BottomToolbar';
import Sidebar from '../components/ui/Sidebar';
import ContextMenu from '../components/ui/ContextMenu';
import DataViewModal from '../components/ui/DataViewModal';
import AdvancedDataAnalyticsModal from '../components/ui/AdvancedDataAnalyticsModal';
import CsvReaderNode from '../components/nodes/CsvReaderNode';
import DatabaseReaderNode from '../components/nodes/DatabaseReaderNode';
import LinearRegressionNode from '../components/nodes/LinearRegressionNode';
import MultiLinearRegressionNode from '../components/nodes/MultiLinearRegressionNode';
import PolynomialRegressionNode from '../components/nodes/PolynomialRegressionNode';
import KNNRegressionNode from '../components/nodes/KNNRegressionNode';
import KNNClassificationNode from '../components/nodes/KNNClassificationNode';
import DataCleanerNode from '../components/nodes/DataCleanerNode';
import BasicNode from '../components/nodes/BasicNode';
import StartNode from '../components/nodes/StartNode';
import ModelVisualizerNode from '../components/nodes/ModelVisualizerNode';
import EncoderNode from '../components/nodes/EncoderNode';
import NormalizerNode from '../components/nodes/NormalizerNode';
import LogisticRegressionNode from '../components/nodes/LogisticRegressionNode';
import NaiveBayesNode from '../components/nodes/NaiveBayesNode';
import DataVisualizerNode from '../components/nodes/DataVisualizerNode';
import ModelEvaluatorNode from '../components/nodes/ModelEvaluatorNode';
import HeatmapNode from '../components/nodes/HeatmapNode';
import FeatureSelectorNode from '../components/nodes/FeatureSelectorNode';
import PCANode from '../components/nodes/PCANode';
import SVDNode from '../components/nodes/SVDNode';
import DescribeNode from '../components/nodes/DescribeNode';
import DataTypeConverterNode from '../components/nodes/DataTypeConverterNode';
import KMeansNode from '../components/nodes/KMeansNode';
import DBSCANNode from '../components/nodes/DBSCANNode';
import HierarchicalClusteringNode from '../components/nodes/HierarchicalClusteringNode';
import FloatingEdge from '../components/edges/FloatingEdge';
import { MdVisibility } from 'react-icons/md';
import { FaProjectDiagram } from 'react-icons/fa';
import ClusterVisualizerModal from '../components/ui/ClusterVisualizerModal';
import DendrogramModal from '../components/ui/DendrogramModal';
import './EditorPage.css';
import { NodeInfoProvider } from '../context/NodeInfoContext';
import NodeInfoPanel from '../components/ui/NodeInfoPanel';
import useStore from '../store/store';

const nodeTypes = {
  // Existing specialized nodes
  start: StartNode,
  csvReader: CsvReaderNode,
  databaseReader: DatabaseReaderNode,
  linearRegression: LinearRegressionNode,
  multiLinearRegression: MultiLinearRegressionNode,
  polynomialRegression: PolynomialRegressionNode,
  knnRegression: KNNRegressionNode,
  knnClassification: KNNClassificationNode,
  dataCleaner: DataCleanerNode,
  modelVisualizer: ModelVisualizerNode,
  encoder: EncoderNode,
  normalizer: NormalizerNode,
  logisticRegression: LogisticRegressionNode,
  naiveBayes: NaiveBayesNode,
  dataVisualizer: DataVisualizerNode,
  heatmap: HeatmapNode,
  featureSelector: FeatureSelectorNode,
  pca: PCANode,
  svd: SVDNode,
  dataTypeConverter: DataTypeConverterNode,
  evaluator: ModelEvaluatorNode,
  // New specialized nodes
  kMeans: KMeansNode,
  hierarchicalClustering: HierarchicalClusteringNode,
  dbscan: DBSCANNode,
  // Generic/basic nodes
  mlp: BasicNode,
  cnn: BasicNode,
  rnn: BasicNode,
  transformer: BasicNode,
  visualizer: BasicNode,
  exporter: BasicNode,
  describeNode: DescribeNode,
};

const edgeTypes = {
  floating: FloatingEdge,
};

const EditorPage = () => {
  const reactFlowWrapper = useRef(null);
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    setNodes,
    setEdges,
    undo,
    redo,
    history,
    historyIndex,
    saveToHistory,
    getId,
    reactFlowInstance,
    setReactFlowInstance
  } = useStore();

  const [activeTool, setActiveTool] = useState('select');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarClosing, setSidebarClosing] = useState(false);

  // Context menu state
  const [contextMenu, setContextMenu] = useState(null);
  const [edgeContextMenu, setEdgeContextMenu] = useState(null);

  // Data view modal state
  const [dataViewModal, setDataViewModal] = useState(null);

  // Advanced analytics modal state
  const [advancedAnalyticsModal, setAdvancedAnalyticsModal] = useState(null);

  // Cluster visualizer modal state
  const [clusterVisualizerModal, setClusterVisualizerModal] = useState(null);

  // Dendrogram modal state (for hierarchical clustering)
  const [dendrogramModal, setDendrogramModal] = useState(null);

  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const { zoom } = useViewport();

  const canUndo = historyIndex > 0;
  const canRedo = historyIndex < history.length - 1;

  // Keyboard shortcuts for undo/redo
  useEffect(() => {
    const handleKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'z' && !event.shiftKey) {
        event.preventDefault();
        undo();
      } else if ((event.ctrlKey || event.metaKey) && (event.key === 'y' || (event.key === 'z' && event.shiftKey))) {
        event.preventDefault();
        redo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo]);

  const isValidConnection = useCallback((connection) => {
    if (connection.source === connection.target) return false;
    return true;
  }, []);

  const nodeColors = {
    csvReader: '#f59e0b',
    databaseReader: '#f59e0b',
    start: '#64748b',
    linearRegression: '#5a67d8',
    multiLinearRegression: '#5a67d8',
    knnRegression: '#5a67d8',
    polynomialRegression: '#5a67d8',
    logisticRegression: '#FF0080',
    naiveBayes: '#FF0080',
    knnClassification: '#FF0080',
    kMeans: '#FF0080',
    hierarchicalClustering: '#FF0080',
    dbscan: '#FF0080',
    mlp: '#FF0080',
    cnn: '#FF0080',
    rnn: '#FF0080',
    transformer: '#FF0080',
    dataCleaner: '#00b09b',
    normalizer: '#00b09b',
    encoder: '#00b09b',
    pca: '#8b5cf6',
    svd: '#8b5cf6',
    featureSelector: '#00b09b',
    heatmap: '#00b09b',
    dataTypeConverter: '#00b09b',
    modelVisualizer: '#fda085',
    dataVisualizer: '#fda085',
    visualizer: '#fda085',
    evaluator: '#f5576c',
    describeNode: '#718096',
    default: '#6a1b9a'
  };

  const handleConnect = useCallback((params) => {
    const sourceNode = nodes.find(n => n.id === params.source);
    const nodeType = sourceNode?.type || 'default';
    const edgeColor = nodeColors[nodeType] || nodeColors.default;

    onConnect({
      ...params,
      type: 'floating',
      style: { stroke: edgeColor, strokeWidth: 2 },
      markerEnd: { type: 'arrowclosed', color: edgeColor },
    });
    saveToHistory();
  }, [nodes, onConnect, saveToHistory]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      const type = event.dataTransfer.getData('application/reactflow');
      const nodeName = event.dataTransfer.getData('application/reactflow-name');
      if (!type || !reactFlowInstance) return;

      const position = reactFlowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const newNode = {
        id: getId(),
        type,
        position,
        data: {
          label: nodeName || `New Node`,
          nodeType: type
        }
      };
      setNodes([...nodes, newNode]);
      saveToHistory();
    },
    [reactFlowInstance, nodes, setNodes, getId, saveToHistory]
  );

  const addDefaultNode = useCallback(() => {
    const newNode = {
      id: getId(),
      type: 'csvReader',
      position: { x: 100 + Math.random() * 400, y: 100 + Math.random() * 400 },
      data: { label: `New Node` },
    };
    setNodes([...nodes, newNode]);
    saveToHistory();
    fitView();
  }, [nodes, setNodes, getId, saveToHistory, fitView]);

  const handleMenuClick = useCallback(() => {
    if (sidebarOpen) {
      setSidebarClosing(true);
      setTimeout(() => {
        setSidebarOpen(false);
        setSidebarClosing(false);
      }, 300);
    } else {
      setSidebarOpen(true);
    }
  }, [sidebarOpen]);

  // Node Context Menu
  const onNodeContextMenu = useCallback((event, node) => {
    event.preventDefault();
    if (node.type === 'start') return;
    setContextMenu({ x: event.clientX, y: event.clientY, node });
  }, []);

  // Edge Context Menu
  const onEdgeContextMenu = useCallback((event, edge) => {
    event.preventDefault();
    setEdgeContextMenu({ x: event.clientX, y: event.clientY, edgeId: edge.id });
  }, []);

  useEffect(() => {
    const handleClick = () => {
      setContextMenu(null);
      setEdgeContextMenu(null);
    };
    if (contextMenu || edgeContextMenu) {
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [contextMenu, edgeContextMenu]);

  const handleDeleteNode = useCallback((nodeId) => {
    const newNodes = nodes.filter((node) => node.id !== nodeId);
    const newEdges = edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId);
    setNodes(newNodes);
    setEdges(newEdges);
    saveToHistory();
    setContextMenu(null);
  }, [nodes, edges, setNodes, setEdges, saveToHistory]);

  const handleDeleteEdge = useCallback((edgeId) => {
    const newEdges = edges.filter((edge) => edge.id !== edgeId);
    setEdges(newEdges);
    saveToHistory();
    setEdgeContextMenu(null);
  }, [edges, setEdges, saveToHistory]);

  return (
    <NodeInfoProvider>
      <div className={`editor-container-new ${activeTool === 'pan' ? 'pan-active' : ''}`}>
        <TopToolbar activeTool={activeTool} setActiveTool={setActiveTool} onMenuClick={handleMenuClick} />
        {sidebarOpen && <Sidebar className={sidebarClosing ? 'slide-out' : ''} />}

        <div className="canvas-area">
          <div className="reactflow-wrapper-new" ref={reactFlowWrapper}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={handleConnect}
              onInit={setReactFlowInstance}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onNodeContextMenu={onNodeContextMenu}
              onEdgeContextMenu={onEdgeContextMenu}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              isValidConnection={(c) => c.source !== c.target}
              fitView
              fitViewOptions={{ maxZoom: 0.75 }}
              proOptions={{ hideAttribution: true }}
              connectionMode="loose"
              connectionLineComponent={SmoothStepEdge}
              defaultEdgeOptions={{
                type: 'floating',
                markerEnd: { type: 'arrowclosed', color: '#6a1b9a' },
                style: { stroke: '#6a1b9a', strokeWidth: 2 },
              }}
              panOnDrag={activeTool === 'pan'}
              selectionOnDrag={activeTool === 'select'}
            >
              <MiniMap
                style={{ position: 'absolute', bottom: '60px', right: '15px', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)' }}
                nodeColor="#888"
                maskColor="rgba(255, 255, 255, 0.7)"
              />
            </ReactFlow>

            <button className="add-node-button" onClick={addDefaultNode}>+</button>
            {contextMenu && (() => {
              const node = contextMenu.node;
              const customActions = [];

              // Define which node types can display full datasets and how to access their data
              const datasetNodes = {
                csvReader: {
                  condition: node.data?.headers && node.data?.file,
                  getData: async () => {
                    const { parseFullTabularFile } = await import('../utils/parseTabularFile');
                    const parsed = await parseFullTabularFile(node.data.file, true);
                    return {
                      headers: parsed.headers,
                      rows: parsed.rows,
                      fileName: node.data.file.name,
                    };
                  },
                  useAdvancedAnalytics: true, // Use advanced analytics modal for CSV Reader
                },
                databaseReader: {
                  condition: node.data?.headers && node.data?.rows && node.data.rows.length > 0,
                  getData: async () => ({
                    headers: node.data.headers,
                    rows: node.data.rows,
                    fileName: `Database Query (${node.data.rows.length} rows)`,
                  }),
                  useAdvancedAnalytics: true, // Use advanced analytics modal for Database Reader
                },
                dataCleaner: {
                  condition: node.data?.cleanedRows && node.data.cleanedRows.length > 0,
                  getData: async () => ({
                    headers: node.data.headers,
                    rows: node.data.cleanedRows,
                    fileName: `Cleaned Data (${node.data.cleanedRows.length} rows)`,
                  }),
                },
                encoder: {
                  condition: node.data?.encodedRows && node.data.encodedRows.length > 0,
                  getData: async () => ({
                    headers: node.data.headers,
                    rows: node.data.encodedRows,
                    fileName: `Encoded Data (${node.data.encodedRows.length} rows)`,
                  }),
                },
                normalizer: {
                  condition: node.data?.normalizedRows && node.data.normalizedRows.length > 0,
                  getData: async () => ({
                    headers: node.data.headers,
                    rows: node.data.normalizedRows,
                    fileName: `Normalized Data (${node.data.normalizedRows.length} rows)`,
                  }),
                },
                featureSelector: {
                  condition: node.data?.selectedHeaders && node.data?.selectedRows && node.data.selectedRows.length > 0,
                  getData: async () => ({
                    headers: node.data.selectedHeaders,
                    rows: node.data.selectedRows,
                    fileName: `Selected Features (${node.data.selectedRows.length} rows)`,
                  }),
                },
                pca: {
                  condition: node.data?.transformedHeaders && node.data?.transformedRows && node.data.transformedRows.length > 0,
                  getData: async () => ({
                    headers: node.data.transformedHeaders,
                    rows: node.data.transformedRows,
                    fileName: `PCA Transformed Data (${node.data.transformedRows.length} rows)`,
                  }),
                },
                svd: {
                  condition: node.data?.transformedHeaders && node.data?.transformedRows && node.data.transformedRows.length > 0,
                  getData: async () => ({
                    headers: node.data.transformedHeaders,
                    rows: node.data.transformedRows,
                    fileName: `SVD Transformed Data (${node.data.transformedRows.length} rows)`,
                  }),
                },
                dataTypeConverter: {
                  condition: node.data?.convertedRows && node.data.convertedRows.length > 0,
                  getData: async () => ({
                    headers: node.data.headers,
                    rows: node.data.convertedRows,
                    fileName: `Converted Data (${node.data.convertedRows.length} rows)`,
                  }),
                },
                kMeans: {
                  condition: node.data?.clusteredData && node.data.clusteredData.length > 0,
                  getData: async () => ({
                    headers: node.data.clusteredHeaders,
                    rows: node.data.clusteredData,
                    fileName: `Clustered Data (${node.data.clusteredData.length} rows)`,
                  }),
                },
                dbscan: {
                  condition: node.data?.clusteredData && node.data.clusteredData.length > 0,
                  getData: async () => ({
                    headers: node.data.clusteredHeaders,
                    rows: node.data.clusteredData,
                    fileName: `DBSCAN Clustered Data (${node.data.clusteredData.length} rows)`,
                  }),
                },
                hierarchicalClustering: {
                  condition: node.data?.clusteredData && node.data.clusteredData.length > 0,
                  getData: async () => ({
                    headers: node.data.clusteredHeaders,
                    rows: node.data.clusteredData,
                    fileName: `Hierarchical Clustered Data (${node.data.clusteredData.length} rows)`,
                  }),
                },
              };

              // Check if this node type supports dataset viewing
              const nodeConfig = datasetNodes[node.type];
              if (nodeConfig && nodeConfig.condition) {
                customActions.push({
                  label: 'View Full Dataset',
                  icon: <MdVisibility />,
                  className: 'view',
                  onClick: async () => {
                    try {
                      const data = await nodeConfig.getData();
                      // Use advanced analytics modal for CSV Reader and Database Reader, standard modal for others
                      if (nodeConfig.useAdvancedAnalytics) {
                        setAdvancedAnalyticsModal({
                          ...data,
                          nodeType: node.type,  // Store node type for filtering callback
                          nodeId: node.id       // Store node ID for precise matching
                        });
                      } else {
                        setDataViewModal(data);
                      }
                    } catch (err) {
                      console.error('Failed to load dataset for viewing:', err);
                    }
                  },
                });
              }

              // Cluster Graph Visualization
              if ((node.type === 'kMeans' || node.type === 'dbscan') && node.data?.clusteredData && node.data.clusteredData.length > 0) {
                customActions.push({
                  label: 'View Cluster Graph',
                  icon: <FaProjectDiagram />,
                  className: 'view',
                  onClick: () => {
                    setClusterVisualizerModal({
                      rows: node.data.clusteredData,
                      headers: node.data.clusteredHeaders,
                      features: node.data.selectedFeatures,
                      centers: node.data.clusterCenters
                    });
                  }
                });
              }

              // Dendrogram Visualization for Hierarchical Clustering
              if (node.type === 'hierarchicalClustering' && node.data?.dendrogram) {
                customActions.push({
                  label: 'View Dendrogram',
                  icon: <FaProjectDiagram />,
                  className: 'view',
                  onClick: () => {
                    setDendrogramModal({
                      dendrogram: node.data.dendrogram,
                      cutHeight: node.data.cutHeight,
                      nClusters: node.data.nClusters,
                      linkageMethod: node.data.linkageMethod,
                      distanceMetric: node.data.distanceMetric,
                      sampleData: node.data.sampleData,
                      headers: node.data.headers
                    });
                  }
                });
              }

              // Special handling for Data Cleaner node - add option to view removed rows
              if (node.type === 'dataCleaner' && node.data?.removedRows && node.data.removedRows.length > 0) {
                customActions.push({
                  label: 'View Removed Rows',
                  icon: <MdVisibility />,
                  className: 'view',
                  onClick: () => {
                    setDataViewModal({
                      headers: node.data.headers,
                      rows: node.data.removedRows,
                      fileName: `Removed Rows (${node.data.removedRows.length} rows)`,
                    });
                  },
                });
              }

              return (
                <ContextMenu
                  x={contextMenu.x}
                  y={contextMenu.y}
                  nodeId={node.id}
                  nodeType={node.type}
                  onDelete={handleDeleteNode}
                  onClose={() => setContextMenu(null)}
                  customActions={customActions}
                />
              );
            })()}
            {edgeContextMenu && (
              <div
                className="edge-context-menu"
                style={{
                  position: 'fixed',
                  top: edgeContextMenu.y,
                  left: edgeContextMenu.x,
                  background: 'white',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                  zIndex: 1000,
                  minWidth: '120px',
                }}
              >
                <button
                  onClick={() => handleDeleteEdge(edgeContextMenu.edgeId)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    border: 'none',
                    background: 'transparent',
                    textAlign: 'left',
                    cursor: 'pointer',
                    color: '#e53e3e',
                    fontSize: '14px',
                  }}
                  onMouseEnter={(e) => e.target.style.background = '#f7fafc'}
                  onMouseLeave={(e) => e.target.style.background = 'transparent'}
                >
                  🗑️ Delete Edge
                </button>
              </div>
            )}
          </div>
        </div>
        <BottomToolbar
          zoomIn={zoomIn}
          zoomOut={zoomOut}
          fitView={fitView}
          zoomLevel={zoom}
          onUndo={undo}
          onRedo={redo}
          canUndo={canUndo}
          canRedo={canRedo}
        />

        {/* Data View Modal */}
        {dataViewModal && (
          <DataViewModal
            isOpen={true}
            onClose={() => setDataViewModal(null)}
            headers={dataViewModal.headers}
            rows={dataViewModal.rows}
            fileName={dataViewModal.fileName}
          />
        )}

        {/* Advanced Analytics Modal (for CSV Reader) */}
        {advancedAnalyticsModal && (
          <AdvancedDataAnalyticsModal
            isOpen={true}
            onClose={() => setAdvancedAnalyticsModal(null)}
            headers={advancedAnalyticsModal.headers}
            rows={advancedAnalyticsModal.rows}
            fileName={advancedAnalyticsModal.fileName}
            onLoadFilteredData={(filteredData) => {
              // Find the node and update its data
              const newNodes = nodes.map((n) => {
                if (n.id === advancedAnalyticsModal.nodeId) {
                  return {
                    ...n,
                    data: {
                      ...n.data,
                      headers: filteredData.headers,
                      rows: filteredData.rows,
                      isFiltered: true,
                      originalRowCount: advancedAnalyticsModal.rows.length,
                      filteredRowCount: filteredData.rows.length
                    }
                  };
                }
                return n;
              });
              setNodes(newNodes);
              saveToHistory();
            }}
          />
        )}

        {/* Cluster Visualizer Modal */}
        {clusterVisualizerModal && (
          <ClusterVisualizerModal
            isOpen={true}
            onClose={() => setClusterVisualizerModal(null)}
            rows={clusterVisualizerModal.rows}
            headers={clusterVisualizerModal.headers}
            features={clusterVisualizerModal.features}
            centers={clusterVisualizerModal.centers}
          />
        )}

        {/* Dendrogram Modal */}
        {dendrogramModal && (
          <DendrogramModal
            isOpen={true}
            onClose={() => setDendrogramModal(null)}
            dendrogram={dendrogramModal.dendrogram}
            cutHeight={dendrogramModal.cutHeight}
            nClusters={dendrogramModal.nClusters}
            linkageMethod={dendrogramModal.linkageMethod}
            distanceMetric={dendrogramModal.distanceMetric}
            sampleData={dendrogramModal.sampleData}
            headers={dendrogramModal.headers}
          />
        )}

        <NodeInfoPanel />
      </div>
    </NodeInfoProvider>
  );
};

export default EditorPage;