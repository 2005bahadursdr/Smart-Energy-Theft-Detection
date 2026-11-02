import React from 'react';
import {
  LayoutDashboard,
  Gauge,
  Network,
  Crosshair,
  ShieldAlert,
  Award,
  GitCompare,
  Link2,
  FileSpreadsheet,
  ChevronRight,
  X,
  Zap,
} from 'lucide-react';

export const navigationItems = [
  { id: 'overview', label: 'Executive Overview', icon: LayoutDashboard },
  { id: 'meters', label: 'Smart Meters & MDMS', icon: Gauge },
  { id: 'fl', label: 'FL Clients & Rounds', icon: Network },
  { id: 'predictions', label: 'Theft-Risk Predictions', icon: Crosshair },
  { id: 'byzantine', label: 'Byzantine Threat Monitor', icon: ShieldAlert },
  { id: 'trust', label: 'Trust & Reputation Engine', icon: Award },
  { id: 'comparison', label: 'FedAvg vs Trust Defense', icon: GitCompare },
  { id: 'blockchain', label: 'Blockchain Audit Trail', icon: Link2 },
  { id: 'experiments', label: 'Experiment Benchmarks', icon: FileSpreadsheet },
];

export const Sidebar = ({ activeTab, onSelectTab, sidebarOpen, setSidebarOpen }) => {
  const handleSelect = (id) => {
    onSelectTab(id);
    if (setSidebarOpen) {
      setSidebarOpen(false);
    }
  };

  const navContent = (
    <div className="flex flex-col justify-between h-full py-4 px-3">
      <div className="space-y-1">
        <div className="flex items-center justify-between px-3 mb-2">
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
            Navigation Modules
          </p>
          {/* Close button inside mobile drawer */}
          {setSidebarOpen && (
            <button
              onClick={() => setSidebarOpen(false)}
              className="md:hidden p-1 rounded-md text-slate-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;

          return (
            <button
              key={item.id}
              onClick={() => handleSelect(item.id)}
              className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                isActive
                  ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <div className="flex items-center space-x-3">
                <Icon
                  className={`w-4 h-4 ${
                    isActive ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-300'
                  }`}
                />
                <span className="truncate">{item.label}</span>
              </div>
              {isActive && <ChevronRight className="w-3.5 h-3.5 text-cyan-400 shrink-0" />}
            </button>
          );
        })}
      </div>

      {/* Footer Info Box */}
      <div className="p-3 m-1 rounded-lg bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400">
        <div className="flex items-center justify-between mb-1">
          <span className="font-semibold text-slate-300">Live Target Grid</span>
          <span className="text-cyan-400 font-mono text-[10px]">SGCC (42k) + Synth</span>
        </div>
        <p className="text-[10px] text-slate-500 leading-tight">
          Byzantine tolerance threshold: 40% malicious nodes.
        </p>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Persistent Sidebar */}
      <aside className="hidden md:flex w-64 bg-[#0F1422] border-r border-slate-800/80 min-h-[calc(100vh-80px)] flex-col justify-between shrink-0">
        {navContent}
      </aside>

      {/* Mobile Slide-In Drawer */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex">
          {/* Backdrop Overlay */}
          <div
            className="fixed inset-0 bg-black/75 backdrop-blur-sm animate-in fade-in duration-200"
            onClick={() => setSidebarOpen(false)}
          />

          {/* Drawer Panel */}
          <div className="relative w-72 max-w-[80vw] bg-[#0F1422] border-r border-slate-800 h-full z-10 shadow-2xl flex flex-col animate-in slide-in-from-left duration-200">
            {navContent}
          </div>
        </div>
      )}
    </>
  );
};

export default Sidebar;
