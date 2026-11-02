import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export const ErrorAlert = ({
  title = 'Service Communication Error',
  message,
  onRetry,
}) => {
  return (
    <div className="glass-panel border-rose-500/30 rounded-xl p-6 bg-rose-950/20 text-rose-200">
      <div className="flex items-start space-x-4">
        <div className="p-2.5 bg-rose-500/20 rounded-lg text-rose-400 shrink-0">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <div className="flex-1">
          <h4 className="text-base font-semibold text-rose-300">{title}</h4>
          <p className="text-sm text-rose-200/80 mt-1">
            {message || 'Unable to connect to the backend API service. Ensure the FastAPI backend server is running on http://127.0.0.1:8000.'}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-4 inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/40 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Connection</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ErrorAlert;
