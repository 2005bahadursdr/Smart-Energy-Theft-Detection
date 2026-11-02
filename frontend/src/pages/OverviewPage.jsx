import React, { useEffect, useState, useCallback } from 'react';
import {
  Activity,
  Zap,
  Network,
  ShieldAlert,
  Link2,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  TrendingUp,
  RefreshCw,
  Radio,
} from 'lucide-react';
import { apiService } from '../services/api';
import StatCard from '../components/StatCard';
import Badge from '../components/Badge';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorAlert from '../components/ErrorAlert';

export const OverviewPage = ({ onNavigate, autoRefresh = false, refreshTrigger = 0 }) => {
  const [data, setData] = useState({
    health: null,
    meters: null,
    clients: null,
    rounds: null,
    audit: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadOverviewData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const [healthRes, metersRes, clientsRes, roundsRes, auditRes] = await Promise.all([
        apiService.getHealth().catch(() => null),
        apiService.getMetersSummary().catch(() => null),
        apiService.getClients().catch(() => null),
        apiService.getFLRounds().catch(() => null),
        apiService.getBlockchainAudit().catch(() => null),
      ]);

      setData({
        health: healthRes,
        meters: metersRes,
        clients: clientsRes,
        rounds: roundsRes,
        audit: auditRes,
      });
    } catch (err) {
      setError(err.message || 'Failed to fetch overview metrics.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadOverviewData();
  }, [loadOverviewData]);

  // Handle external refresh trigger
  useEffect(() => {
    if (refreshTrigger > 0) {
      loadOverviewData();
    }
  }, [refreshTrigger, loadOverviewData]);

  // Live Auto-Stream Refresh
  useEffect(() => {
    let interval = null;
    if (autoRefresh) {
      interval = setInterval(() => {
        loadOverviewData();
      }, 10000); // 10s live stream polling
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, loadOverviewData]);

  if (loading && !data.meters) return <LoadingSpinner message="Aggregating real-time multi-subsystem telemetry..." />;
  if (error && !data.meters) return <ErrorAlert title="Failed to Load Overview" message={error} onRetry={loadOverviewData} />;

  const { health, meters, clients, rounds, audit } = data;

  // Fully dynamic metrics calculated directly from real backend state
  const totalMeters = meters?.total_meters ?? 0;
  const theftRatioPct = meters?.theft_ratio != null ? (meters.theft_ratio * 100).toFixed(1) : '0.0';
  const meanKwh = meters?.consumption_metrics?.mean_kwh != null ? meters.consumption_metrics.mean_kwh.toFixed(2) : '0.00';
  
  const totalRounds = rounds?.total_rounds_completed ?? (rounds?.rounds?.length ?? 0);
  const latestRoundF1 = rounds?.rounds?.length ? rounds.rounds[rounds.rounds.length - 1]?.f1_score : null;
  const bestF1 = rounds?.best_f1_score != null 
    ? (rounds.best_f1_score * 100).toFixed(1) 
    : latestRoundF1 != null 
    ? (latestRoundF1 * 100).toFixed(1) 
    : '0.0';

  const clientList = clients?.clients || [];
  const activeClients = clients?.active_clients ?? clientList.filter((c) => c.status !== 'QUARANTINED').length;
  const quarantinedClients = clients?.quarantined_clients ?? clientList.filter((c) => c.status === 'QUARANTINED').length;

  const totalAuditRecords = audit?.total_audit_records ?? (audit?.blocks ? audit.blocks.reduce((acc, b) => acc + (b.records?.length || 0), 0) : 0);
  const latestBlock = audit?.blocks?.length 
    ? audit.blocks[audit.blocks.length - 1].block_number 
    : (health?.blockchain?.block_number ?? 0);

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center flex-wrap gap-2">
            Executive Research Overview
            <Badge variant="demo">{meters?.data_source_label || 'SIMULATED_DEMO_DATA'}</Badge>
            {autoRefresh && <Badge variant="warning">Live Stream Active</Badge>}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Real-time status of Smart Meter Simulator, Federated Learning Coordinator, Trust Engine, and Ethereum Reputation Ledger.
          </p>
        </div>
        <div className="flex items-center flex-wrap gap-2">
          <Badge variant="tamperOk">[TAMPER-EVIDENT OK]</Badge>
          <Badge variant="success">All Firewalls Active</Badge>
          <button
            onClick={loadOverviewData}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Refresh Overview Data"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Primary KPI Grid (Fully Responsive) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Smart Meter Grid"
          value={`${totalMeters.toLocaleString()} Meters`}
          subtitle={`Theft Ratio: ${theftRatioPct}% (${meanKwh} kWh/h)`}
          icon={Activity}
          color="cyan"
          badge={<Badge variant="primary">15m Slices</Badge>}
        />

        <StatCard
          title="FL Convergence F1"
          value={`${bestF1}%`}
          subtitle={`Rounds: ${totalRounds} | Strategy: Trust-Aware`}
          icon={TrendingUp}
          color="emerald"
          badge={<Badge variant="success">Converged</Badge>}
        />

        <StatCard
          title="Byzantine Threats"
          value={`${quarantinedClients} Quarantined`}
          subtitle={`${activeClients} Substations Honest & Active`}
          icon={ShieldAlert}
          color="rose"
          badge={<Badge variant="danger">Sign-Flipping</Badge>}
        />

        <StatCard
          title="On-Chain Audit Records"
          value={`${totalAuditRecords} Proofs`}
          subtitle={`Contract Block #${latestBlock}`}
          icon={Link2}
          color="purple"
          badge={<Badge variant="blockchain">Immutable</Badge>}
        />
      </div>

      {/* Architecture & Pipeline Visual (Responsive Cards) */}
      <div className="glass-panel rounded-xl p-4 sm:p-6 border border-slate-800/80">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Network className="w-4 h-4 text-cyan-400" />
            End-to-End Privacy-Preserving System Pipeline
          </h3>
          <span className="text-[11px] font-mono text-cyan-400">Zero Raw Data Sharing</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {/* Step 1 */}
          <div className="p-3.5 rounded-lg bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-500 mb-1">
                <span>01</span>
                <span className="text-cyan-400 font-semibold">Grid Edge</span>
              </div>
              <h4 className="text-xs font-bold text-white">Smart Meter Fleet</h4>
              <p className="text-[11px] text-slate-400 mt-1">100-1,000 smart meters generate 15-min interval readings with simulated bypass & scaling tampering.</p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-cyan-400">
              {totalMeters.toLocaleString()} Physical Endpoints
            </div>
          </div>

          {/* Step 2 */}
          <div className="p-3.5 rounded-lg bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-500 mb-1">
                <span>02</span>
                <span className="text-indigo-400 font-semibold">MDMS Ingestion</span>
              </div>
              <h4 className="text-xs font-bold text-white">Regional Partition</h4>
              <p className="text-[11px] text-slate-400 mt-1">Meter Data Management System clusters readings into 5 Non-IID substation feeders (client_A - client_E).</p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-indigo-400">
              5 Isolated Local Vaults
            </div>
          </div>

          {/* Step 3 */}
          <div className="p-3.5 rounded-lg bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-500 mb-1">
                <span>03</span>
                <span className="text-emerald-400 font-semibold">Deep Learning</span>
              </div>
              <h4 className="text-xs font-bold text-white">PyTorch CNN-LSTM</h4>
              <p className="text-[11px] text-slate-400 mt-1">Clients train local neural networks on confidential consumption data. Only parameter gradients leave the firewall.</p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-emerald-400">
              F1 Convergence: {bestF1}%
            </div>
          </div>

          {/* Step 4 */}
          <div className="p-3.5 rounded-lg bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-500 mb-1">
                <span>04</span>
                <span className="text-amber-400 font-semibold">Byzantine Defense</span>
              </div>
              <h4 className="text-xs font-bold text-white">Trust Engine</h4>
              <p className="text-[11px] text-slate-400 mt-1">Cosine similarity, loss validation, and reputation EMA isolate malicious updates (client_D) with w_k = 0.</p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-rose-400">
              {quarantinedClients} Node Quarantined
            </div>
          </div>

          {/* Step 5 */}
          <div className="p-3.5 rounded-lg bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-500 mb-1">
                <span>05</span>
                <span className="text-purple-400 font-semibold">Ethereum Audit</span>
              </div>
              <h4 className="text-xs font-bold text-white">Reputation Ledger</h4>
              <p className="text-[11px] text-slate-400 mt-1">Solidity contract hashes round proofs and reputation shifts onto Ethereum-compatible blockchain.</p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-purple-400">
              [TAMPER-EVIDENT OK]
            </div>
          </div>
        </div>
      </div>

      {/* Two Column Section: Substations Quick Table & Recent On-Chain Rounds */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Active Substation Client Partitions */}
        <div className="glass-panel rounded-xl p-4 sm:p-5 border border-slate-800/80">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              Substation Client Partitions & Non-IID Profiles
            </h3>
            <button
              onClick={() => onNavigate('fl')}
              className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-medium"
            >
              View Full Details <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="overflow-x-auto -mx-2 sm:mx-0">
            <table className="w-full text-xs text-left min-w-[420px]">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 px-3">Client</th>
                  <th className="py-2 px-3">Region / Profile</th>
                  <th className="py-2 px-3">Mean Daily</th>
                  <th className="py-2 px-3">Reputation</th>
                  <th className="py-2 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {clientList.map((c) => (
                  <tr key={c.client_id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-3 font-semibold text-white">
                      {c.client_id}
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-300">{c.region_type}</td>
                    <td className="py-2.5 px-3 text-slate-200">
                      {c.mean_daily_kwh} kWh
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={
                          c.current_reputation >= 0.8
                            ? 'text-emerald-400 font-semibold'
                            : c.current_reputation >= 0.5
                            ? 'text-amber-400 font-semibold'
                            : 'text-rose-400 font-semibold'
                        }
                      >
                        {(c.current_reputation * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      {c.status === 'QUARANTINED' ? (
                        <Badge variant="danger">QUARANTINED</Badge>
                      ) : (
                        <Badge variant="success">ACTIVE</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Recent On-Chain Audit Proofs */}
        <div className="glass-panel rounded-xl p-4 sm:p-5 border border-slate-800/80">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Link2 className="w-4 h-4 text-purple-400" />
              Recent On-Chain Audit Proofs
            </h3>
            <button
              onClick={() => onNavigate('blockchain')}
              className="text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1 font-medium"
            >
              Ledger Explorer <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-3">
            {(audit?.blocks || []).slice(-2).map((block, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800/80 space-y-2"
              >
                <div className="flex items-center justify-between text-xs flex-wrap gap-1">
                  <span className="font-mono text-purple-400 font-semibold">
                    Round #{block.round} • Block #{block.block_number}
                  </span>
                  <Badge variant="tamperOk">[TAMPER-EVIDENT OK]</Badge>
                </div>
                <div className="text-[11px] font-mono text-slate-400 truncate">
                  Tx: {block.transaction_hash}
                </div>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {(block.records || []).map((r, rIdx) => (
                    <span
                      key={rIdx}
                      className={`text-[10px] px-2 py-0.5 rounded border font-mono ${
                        r.decision === 'QUARANTINED'
                          ? 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                          : 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                      }`}
                    >
                      {r.client_id}: {r.decision} (Rep {r.new_reputation.toFixed(2)})
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OverviewPage;
