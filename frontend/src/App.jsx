import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';

// Pages
import OverviewPage from './pages/OverviewPage';
import MetersPage from './pages/MetersPage';
import FLClientsRoundsPage from './pages/FLClientsRoundsPage';
import PredictionsPage from './pages/PredictionsPage';
import ByzantineMonitorPage from './pages/ByzantineMonitorPage';
import TrustReputationPage from './pages/TrustReputationPage';
import ComparisonPage from './pages/ComparisonPage';
import BlockchainAuditPage from './pages/BlockchainAuditPage';
import ExperimentsPage from './pages/ExperimentsPage';

export const App = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleRefreshAll = () => {
    setRefreshTrigger((prev) => prev + 1);
  };

  const renderActivePage = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewPage onNavigate={(tab) => setActiveTab(tab)} autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      case 'meters':
        return <MetersPage autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      case 'fl':
        return <FLClientsRoundsPage autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      case 'predictions':
        return <PredictionsPage autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      case 'byzantine':
        return <ByzantineMonitorPage onRefreshAll={handleRefreshAll} autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      case 'trust':
        return <TrustReputationPage autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      case 'comparison':
        return <ComparisonPage autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      case 'blockchain':
        return <BlockchainAuditPage autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      case 'experiments':
        return <ExperimentsPage autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
      default:
        return <OverviewPage onNavigate={(tab) => setActiveTab(tab)} autoRefresh={autoRefresh} refreshTrigger={refreshTrigger} />;
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col font-sans">
      {/* Top Navigation */}
      <Navbar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        onRefreshAll={handleRefreshAll}
        autoRefresh={autoRefresh}
        setAutoRefresh={setAutoRefresh}
      />

      {/* Main Container */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Navigation Sidebar */}
        <Sidebar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
        />

        {/* Dynamic Page Content */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 max-w-7xl mx-auto w-full transition-all">
          {renderActivePage()}
        </main>
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#0B0F19] py-3 px-4 sm:px-6 text-center text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-slate-400">
            Blockchain-Based Byzantine-Resilient FL Smart Grid
          </span>
          <span>•</span>
          <span>MongoDB Atlas</span>
        </div>
        <div className="text-[11px] text-slate-500 font-mono">
          Research Prototyping Platform • Zero Raw Telemetry Exfiltration
        </div>
      </footer>
    </div>
  );
};

export default App;
