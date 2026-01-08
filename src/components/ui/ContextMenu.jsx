import React from 'react';
import './ContextMenu.css';
import { MdDelete } from 'react-icons/md';

const ContextMenu = ({ x, y, nodeId, nodeType, onDelete, onClose, customActions = [] }) => {
  if (!nodeId) return null;

  // Don't show delete option for start node
  const showDelete = nodeType !== 'start';

  const handleDelete = () => {
    if (showDelete && onDelete) {
      onDelete(nodeId);
    }
    onClose();
  };

  const handleCustomAction = (action) => {
    if (action.onClick) {
      action.onClick();
    }
    onClose();
  };

  // If no delete and no custom actions, don't show menu
  if (!showDelete && customActions.length === 0) return null;

  return (
    <div
      className="context-menu"
      style={{ left: `${x}px`, top: `${y}px` }}
      onClick={(e) => e.stopPropagation()}
    >
      {customActions.map((action, idx) => (
        <button
          key={idx}
          className={`context-menu-item ${action.className || ''}`}
          onClick={() => handleCustomAction(action)}
        >
          {action.icon}
          <span>{action.label}</span>
        </button>
      ))}

      {showDelete && (
        <button className="context-menu-item delete" onClick={handleDelete}>
          <MdDelete />
          <span>Delete</span>
        </button>
      )}
    </div>
  );
};

export default ContextMenu;

