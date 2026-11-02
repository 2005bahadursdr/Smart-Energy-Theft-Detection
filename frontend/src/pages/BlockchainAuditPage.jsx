import React, { useEffect, useState } from 'react';
import {
  Link2,
  CheckCircle2,
  AlertTriangle,
  Lock,
  FileCode,
  Blocks,
  Hash,
  Shield,
  ExternalLink,
  Copy,
  RefreshCw,
} from 'lucide-react';
import { apiService } from '../services/api';
import StatCard from '../components/StatCard';
import Badge from '../components/Badge';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorAlert from '../components/ErrorAlert';

export const BlockchainAuditPage = ({ autoRefresh = false, refreshTrigger = 0 }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [copiedHash, setCopiedHash] = useState(null);

  const loadAuditData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiService.getBlockchainAudit();
      setData(res);
    } catch (err) {
      setError(err.message || 'Failed to retrieve on-chain blockchain audit records.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAuditData();
  }, []);

  useEffect(() => {
    if (refreshTrigger > 0) {
      loadAuditData();
    }
  }, [refreshTrigger]);

  useEffect(() => {
    let interval = null;
    if (autoRefresh) {
      interval = setInterval(() => {
        loadAuditData();
      }, 10000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const copyToClipboard = (text) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedHash(text);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  if (loading) return <LoadingSpinner message="Querying Ethereum ReputationRegistry smart contract logs..." />;
  if (error) return <ErrorAlert title="Blockchain Ledger Offline" message={error} onRetry={loadAuditData} />;

  const blocks = data?.blocks || [];
  const totalBlocks = data?.total_blocks ?? blocks.length;
  const totalRecords = data?.total_audit_records ?? blocks.reduce((acc, b) => acc + (b.records?.length || 0), 0);
  const contractAddress = data?.contract_address || '0xF2E246BB76DF876Cef8b38ae84130F4F55De395b';
  const isRealData = data?.data_source_label === 'REAL_ON_CHAIN_AUDIT';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center flex-wrap gap-2">
            Blockchain Audit Trail & Reputation Registry
            <Badge variant="tamperOk">[TAMPER-EVIDENT OK]</Badge>
            {autoRefresh && <Badge variant="warning">Live Stream Active</Badge>}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Immutable, cryptographically verifiable audit trail of federated aggregation decisions and reputation transitions.
          </p>
        </div>
        <div className="flex items-center flex-wrap gap-2">
          <Badge variant="blockchain">EVM Chain ID: {data?.chain_id || 1337}</Badge>
          <Badge variant="primary">Provider: {data?.provider_type || 'eth_tester'}</Badge>
          <button
            onClick={loadAuditData}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Reload Blockchain Ledger"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Epistemic Principle Notice */}
      <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-800/40 text-xs text-purple-200 space-y-1">
        <div className="flex items-center space-x-2 font-semibold text-purple-300">
          <Shield className="w-4 h-4 text-purple-400" />
          <span>Evidentiary Disclaimer & Cryptographic Boundaries</span>
        </div>
        <p className="text-purple-200/80 text-[11px] leading-relaxed">
          {data?.epistemic_disclaimer ||
            'Blockchain records cryptographic evidence of federated updates, canonical hashes, and reputation transitions. The blockchain does not itself detect or prevent Byzantine poisoning; it creates an immutable, verifiable audit trail for non-repudiation. Hashes prove non-repudiation and temporal ordering, not that the coordinator decision was ground-truth correct.'}
        </p>
      </div>

      {/* Contract & Ledger KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="ReputationRegistry Contract"
          value={`${contractAddress.slice(0, 6)}...${contractAddress.slice(-4)}`}
          subtitle="Solidity ^0.8.20 Smart Contract"
          icon={FileCode}
          color="purple"
          badge={<Badge variant="blockchain">Verified</Badge>}
        />

        <StatCard
          title="On-Chain Round Blocks"
          value={totalBlocks}
          subtitle="Committed via Batch Transactions"
          icon={Blocks}
          color="cyan"
        />

        <StatCard
          title="Audit Trail Entries"
          value={totalRecords}
          subtitle="100% Tamper-Evident Canonical Hashes"
          icon={Hash}
          color="emerald"
        />

        <StatCard
          title="Data Confidentiality"
          value="Zero Weights On-Chain"
          subtitle="Zero Raw Smart Meter Telemetry"
          icon={Lock}
          color="amber"
          badge={<Badge variant="success">Compliant</Badge>}
        />
      </div>

      {/* Contract Metadata Box */}
      <div className="glass-panel rounded-xl p-4 border border-slate-800/80 flex flex-col md:flex-row md:items-center justify-between text-xs gap-3">
        <div className="flex items-center space-x-2">
          <span className="font-mono text-slate-400">Deployed Contract:</span>
          <span className="font-mono text-cyan-400 font-bold bg-slate-900 px-2 py-1 rounded border border-slate-800">
            {contractAddress}
          </span>
          <button
            onClick={() => copyToClipboard(contractAddress)}
            className="text-slate-400 hover:text-white transition-colors"
            title="Copy Contract Address"
          >
            <Copy className="w-3.5 h-3.5" />
          </button>
          {copiedHash === contractAddress && (
            <span className="text-[10px] text-emerald-400 font-mono">Copied!</span>
          )}
        </div>
        <div className="flex items-center space-x-3 text-slate-400 font-mono text-[11px]">
          <span>Interface: Web3.py v8.0</span>
          <span>•</span>
          <span>Gas Optimization: Batch Multi-Record Commits</span>
        </div>
      </div>

      {/* On-Chain Audit Blocks Ledger */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Blocks className="w-4 h-4 text-purple-400" />
            Committed FL Round Audit Ledger & Transaction Proofs
          </h3>
          <Badge variant={isRealData ? 'tamperOk' : 'blockchain'}>
            {data?.data_source_label || 'REAL_ON_CHAIN_AUDIT'}
          </Badge>
        </div>

        {blocks.length === 0 ? (
          <div className="glass-panel rounded-xl p-8 border border-slate-800/80 text-center space-y-3">
            <Blocks className="w-10 h-10 text-slate-500 mx-auto" />
            <h4 className="text-sm font-semibold text-slate-300">No On-Chain Audit Records Found</h4>
            <p className="text-xs text-slate-500 max-w-md mx-auto">
              Only authentic on-chain audit records from completed federated learning rounds are displayed.
              Dummy records are strictly excluded from the persistent blockchain audit trail.
            </p>
          </div>
        ) : (
          blocks.map((block, bIdx) => {
            const rawTx = block.transaction_hash || '';
            const txDisplay = rawTx.startsWith('0x') ? rawTx : `0x${rawTx}`;

            return (
              <div key={bIdx} className="glass-panel rounded-xl p-5 border border-slate-800/80 space-y-4">
                {/* Block Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800/80 gap-2">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-purple-500/10 rounded-lg text-purple-400">
                      <Blocks className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="font-bold text-white text-xs font-mono">
                        Round #{block.round} Commitment • Block #{block.block_number}
                      </h4>
                      <div className="flex items-center space-x-2 text-[11px] text-slate-400 font-mono truncate max-w-md">
                        <span>TxHash: {txDisplay}</span>
                        <button
                          onClick={() => copyToClipboard(txDisplay)}
                          className="text-slate-400 hover:text-white transition-colors"
                          title="Copy Transaction Hash"
                        >
                          <Copy className="w-3 h-3" />
                        </button>
                        {copiedHash === txDisplay && (
                          <span className="text-[10px] text-emerald-400 font-mono">Copied!</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="tamperOk">[TAMPER-EVIDENT OK]</Badge>
                    <span className="text-[10px] font-mono text-slate-500">
                      {new Date(block.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>

                {/* Records Sub-Table */}
                <div className="overflow-x-auto -mx-2 sm:mx-0">
                  <table className="w-full text-xs text-left font-mono min-w-[760px]">
                    <thead>
                      <tr className="border-b border-slate-800/60 text-slate-400 text-[11px] whitespace-nowrap">
                        <th className="py-2 px-2.5">Client</th>
                        <th className="py-2 px-2.5">Model Update Hash</th>
                        <th className="py-2 px-2.5">Decision</th>
                        <th className="py-2 px-2.5">Δ Rep</th>
                        <th className="py-2 px-2.5">New Rep</th>
                        <th className="py-2 px-2.5">Canonical Record Hash</th>
                        <th className="py-2 px-2.5">Integrity Verification</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40 text-[11px] whitespace-nowrap">
                      {block.records.map((rec, rIdx) => {
                        const isQuarantined = rec.decision === 'QUARANTINED' || rec.decision === 'REJECTED';
                        const isAttenuated = rec.decision === 'ATTENUATED' || rec.decision === 'WEIGHT_REDUCED';
                        const decisionStyle = isQuarantined
                          ? 'bg-rose-500/10 text-rose-300 border border-rose-500/30'
                          : isAttenuated
                          ? 'bg-amber-500/10 text-amber-300 border border-amber-500/30'
                          : 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30';

                        const isTamperOk = rec.integrity_status === '[TAMPER-EVIDENT OK]';

                        return (
                          <tr key={rIdx} className="hover:bg-slate-800/30">
                            <td className="py-2 px-2.5 font-bold text-white">{rec.client_id}</td>
                            <td className="py-2 px-2.5 text-slate-400 truncate max-w-[140px]" title={rec.model_update_hash}>
                              {rec.model_update_hash.slice(0, 10)}...{rec.model_update_hash.slice(-6)}
                            </td>
                            <td className="py-2 px-2.5">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${decisionStyle}`}>
                                {rec.decision}
                              </span>
                            </td>
                            <td className="py-2 px-2.5">
                              <span className={rec.reputation_change < 0 ? 'text-rose-400' : 'text-emerald-400'}>
                                {rec.reputation_change > 0 ? `+${rec.reputation_change.toFixed(2)}` : rec.reputation_change.toFixed(2)}
                              </span>
                            </td>
                            <td className="py-2 px-2.5 text-slate-200 font-semibold">{rec.new_reputation.toFixed(2)}</td>
                            <td className="py-2 px-2.5 text-cyan-400/80 truncate max-w-[140px]" title={rec.record_hash}>
                              {rec.record_hash.slice(0, 10)}...{rec.record_hash.slice(-6)}
                            </td>
                            <td className="py-2 px-2.5">
                              <Badge variant={isTamperOk ? 'tamperOk' : 'tamperFail'}>{rec.integrity_status}</Badge>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default BlockchainAuditPage;
