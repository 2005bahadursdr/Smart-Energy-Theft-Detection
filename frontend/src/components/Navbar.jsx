import React, { useEffect, useState } from 'react';
import {
  Zap,
  Database,
  Link as LinkIcon,
  ShieldAlert,
  ShieldCheck,
  Activity,
  RefreshCw,
  Menu,
  X,
  Radio,
} from 'lucide-react';
import { apiService } from '../services/api';
import Badge from './Badge';

export const Navbar = ({
  sidebarOpen,
  setSidebarOpen,
  onRefreshAll,
  autoRefresh = false,
  setAutoRefresh,
}) => {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isOffline, setIsOffline] = useState(false);

  const checkHealth = async () => {
    try {
      setLoading(true);
      const data = await apiService.getHealth();
      setHealth(data);
      setIsOffline(false);
    } catch {
      setIsOffline(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 20000); // Check every 20s
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-40 bg-[#0B0F19]/90 backdrop-blur-md border-b border-slate-800/80 px-4 sm:px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Left: Mobile Menu Toggle & Brand */}
        <div className="flex items-center space-x-3">
          {/* Mobile Menu Button */}
          <button
            onClick={() => setSidebarOpen && setSidebarOpen(!sidebarOpen)}
            className="md:hidden p-2 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300 hover:text-white transition-colors"
            aria-label="Toggle navigation menu"
          >
            {sidebarOpen ? <X className="w-5 h-5 text-cyan-400" /> : <Menu className="w-5 h-5" />}
          </button>

          <div className="p-2 bg-gradient-to-tr from-cyan-500 to-indigo-600 rounded-lg text-white shadow-lg shadow-cyan-500/20 shrink-0">
            <Zap className="w-4 h-4 sm:w-5 sm:h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-sm sm:text-base font-bold text-white tracking-tight">
                Smart Grid Byzantine FL
              </h1>
              <span className="hidden sm:inline-block">
                <Badge variant="blockchain">Research Prototype</Badge>
              </span>
            </div>
            <p className="text-[11px] sm:text-xs text-slate-400 hidden xs:block">
              Decentralized Energy Theft Detection & On-Chain Audit
            </p>
          </div>
        </div>

        {/* Right: Status Indicators & Actions */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          {/* Live Auto-Refresh Toggle */}
          {setAutoRefresh && (
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`hidden sm:flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                autoRefresh
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-sm shadow-emerald-500/20'
                  : 'bg-slate-900/80 text-slate-400 border-slate-800 hover:text-white'
              }`}
              title="Toggle Live Real-Time Auto-Refresh"
            >
              <Radio className={`w-3.5 h-3.5 ${autoRefresh ? 'text-emerald-400 animate-pulse' : 'text-slate-500'}`} />
              <span className="text-[11px]">{autoRefresh ? 'Live Stream: ON' : 'Live Stream'}</span>
            </button>
          )}

          {/* Database Health (Desktop) */}
          <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs">
            <Database className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">DB:</span>
            {isOffline ? (
              <span className="text-rose-400 font-semibold">Offline</span>
            ) : health?.database ? (
              <span className="text-slate-200 font-medium">
                {health.database.provider === 'mongodb' ? 'MongoDB Atlas' : 'In-Memory'}
              </span>
            ) : (
              <span className="text-slate-500">Checking...</span>
            )}
          </div>

          {/* Blockchain Health (Tablet & Desktop) */}
          <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs">
            <LinkIcon className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-slate-400">EVM:</span>
            {isOffline ? (
              <span className="text-rose-400 font-semibold">Offline</span>
            ) : (
              <span className="text-slate-200 font-medium font-mono">
                Block #{health?.blockchain?.block_number ?? 2}
              </span>
            )}
          </div>

          {/* Privacy Firewall Indicator (Desktop) */}
          <div className="hidden xl:flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-[11px] text-emerald-300">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Privacy Firewall: ON</span>
          </div>

          {/* Manual Refresh Button */}
          <button
            onClick={() => {
              checkHealth();
              if (onRefreshAll) onRefreshAll();
            }}
            title="Refresh System Health & Dynamic Telemetry"
            className="p-1.5 sm:p-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-400 hover:text-white transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Production Warning Notice Bar (Responsive) */}
      <div className="mt-2.5 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20 flex flex-col sm:flex-row sm:items-center justify-between text-[11px] text-amber-300 gap-1.5 leading-normal">
        <div className="flex items-start sm:items-center space-x-1.5 flex-1 min-w-0">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5 sm:mt-0" />
          <span className="leading-snug">
            <strong className="text-amber-200">RESEARCH PROTOTYPE:</strong> Zero raw consumer readings exfiltrated. Production grid deployment requires mTLS & RBAC.
          </span>
        </div>
        <span className="text-[10px] font-mono text-amber-400/90 uppercase shrink-0 font-semibold self-start sm:self-auto bg-amber-500/20 px-1.5 py-0.5 rounded border border-amber-500/30">
          Firewall Active
        </span>
      </div>
    </header>
  );
};

export default Navbar;
