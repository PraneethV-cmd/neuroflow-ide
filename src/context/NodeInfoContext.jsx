
import React, { createContext, useState, useContext } from 'react';
import { nodeInfo } from '../data/nodeInfo';

const NodeInfoContext = createContext();

export const NodeInfoProvider = ({ children }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [activeNodeInfo, setActiveNodeInfo] = useState(null);

    const openInfoPanel = (nodeType) => {
        const info = nodeInfo[nodeType];
        if (info) {
            setActiveNodeInfo(info);
            setIsOpen(true);
        } else {
            console.warn(`No info found for node type: ${nodeType}`);
            // Fallback or ignore
        }
    };

    const closeInfoPanel = () => {
        setIsOpen(false);
        setActiveNodeInfo(null);
    };

    return (
        <NodeInfoContext.Provider value={{ isOpen, activeNodeInfo, openInfoPanel, closeInfoPanel }}>
            {children}
        </NodeInfoContext.Provider>
    );
};

export const useNodeInfo = () => useContext(NodeInfoContext);
