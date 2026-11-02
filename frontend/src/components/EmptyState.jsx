import React from 'react';
import { Database, PlusCircle } from 'lucide-react';

export const EmptyState = ({
  icon: Icon = Database,
  title = 'No Records Found',
  description = 'No telemetry records or events are currently available.',
  actionLabel,
  onAction,
}) => {
  return (
    <div className="glass-panel rounded-xl p-10 text-center flex flex-col items-center justify-center">
      <div className="p-3 bg-slate-800 rounded-full text-slate-400 mb-3">
        <Icon className="w-8 h-8" />
      </div>
      <h4 className="text-base font-semibold text-slate-200">{title}</h4>
      <p className="text-xs text-slate-400 max-w-sm mt-1">{description}</p>
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="mt-4 inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 transition-colors"
        >
          <PlusCircle className="w-3.5 h-3.5" />
          <span>{actionLabel}</span>
        </button>
      )}
    </div>
  );
};

export default EmptyState;
