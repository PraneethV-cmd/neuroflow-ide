
import React from 'react';
import { useNodeInfo } from '../../context/NodeInfoContext';
import { MdInfoOutline } from 'react-icons/md';
import './InfoButton.css';

const InfoButton = ({ nodeType }) => {
    const { openInfoPanel } = useNodeInfo();

    return (
        <button
            onClick={(e) => {
                e.stopPropagation();
                openInfoPanel(nodeType);
            }}
            className="node-info-button"
            title="What does this node do?"
        >
            <MdInfoOutline />
        </button>
    );
};

export default InfoButton;
