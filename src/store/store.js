import { create } from 'zustand';
import {
    applyNodeChanges,
    applyEdgeChanges,
    addEdge,
} from 'reactflow';

// Initial state for nodes
const initialNodes = [
    {
        id: 'node_0',
        type: 'start',
        position: { x: 300, y: 300 },
        data: { label: 'Start' },
    },
];

let nodeId = 1;

const useStore = create((set, get) => ({
    nodes: initialNodes,
    edges: [],
    history: [{ nodes: initialNodes, edges: [] }],
    historyIndex: 0,
    reactFlowInstance: null,

    setReactFlowInstance: (instance) => set({ reactFlowInstance: instance }),

    getId: () => `node_${nodeId++}`,

    setNodes: (nds) => {
        set({ nodes: typeof nds === 'function' ? nds(get().nodes) : nds });
    },

    setEdges: (eds) => {
        set({ edges: typeof eds === 'function' ? eds(get().edges) : eds });
    },

    onNodesChange: (changes) => {
        set({
            nodes: applyNodeChanges(changes, get().nodes),
        });
    },

    onEdgesChange: (changes) => {
        set({
            edges: applyEdgeChanges(changes, get().edges),
        });
    },

    onConnect: (connection) => {
        set({
            edges: addEdge(connection, get().edges),
        });
    },

    // Optimized history management
    saveToHistory: () => {
        const { nodes, edges, history, historyIndex } = get();
        const newHistory = history.slice(0, historyIndex + 1);

        // Deep copy current state to prevent reference issues
        const newState = {
            nodes: JSON.parse(JSON.stringify(nodes)),
            edges: JSON.parse(JSON.stringify(edges)),
        };

        newHistory.push(newState);

        if (newHistory.length > 50) newHistory.shift();

        set({
            history: newHistory,
            historyIndex: newHistory.length - 1,
        });
    },

    undo: () => {
        const { history, historyIndex } = get();
        if (historyIndex > 0) {
            const prevIndex = historyIndex - 1;
            const prevState = history[prevIndex];
            set({
                nodes: JSON.parse(JSON.stringify(prevState.nodes)),
                edges: JSON.parse(JSON.stringify(prevState.edges)),
                historyIndex: prevIndex,
            });
        }
    },

    redo: () => {
        const { history, historyIndex } = get();
        if (historyIndex < history.length - 1) {
            const nextIndex = historyIndex + 1;
            const nextState = history[nextIndex];
            set({
                nodes: JSON.parse(JSON.stringify(nextState.nodes)),
                edges: JSON.parse(JSON.stringify(nextState.edges)),
                historyIndex: nextIndex,
            });
        }
    },
}));

export default useStore;
