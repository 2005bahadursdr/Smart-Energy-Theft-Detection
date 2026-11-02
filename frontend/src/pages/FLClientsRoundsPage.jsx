import React, { useEffect, useState } from 'react';
import {
  Network,
  TrendingUp,
  Cpu,
  ShieldAlert,
  Play,
  RotateCcw,
  CheckCircle2,
  Lock,
  Radio,
  FileCode,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { apiService } from '../services/api';
import StatCard from '../components/StatCard';
import Badge from '../components/Badge';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorAlert from '../components/ErrorAlert';

export const FLClientsRoundsPage = ({ autoRefresh = false, refreshTrigger = 0 }) => {
  const [clientsData, setClientsData] = useState(null);
  const [roundsData, setRoundsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // FL Experiment Form State
  const [rounds, setRounds] = useState(5);
  const [epochs, setEpochs] = useState(1);
  const [model, setModel] = useState('cnn_lstm');
  const [strategy, setStrategy] = useState('trust_aware_fedavg');
  const [attackType, setAttackType] = useState('none');
  const [maliciousFraction, setMaliciousFraction] = useState(0.2);
  const [enableBlockchain, setEnableBlockchain] = useState(true);
  const [seed, setSeed] = useState(42);

  const [flRunning, setFlRunning] = useState(false);
  const [flJob, setFlJob] = useState(null);
  const [flError, setFlError] = useState(null);

  const loadFLData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [cRes, rRes] = await Promise.all([
        apiService.getClients(),
        apiService.getFLRounds(),
      ]);
      setClientsData(cRes);
      setRoundsData(rRes);
    } catch (err) {
      setError(err.message || 'Failed to retrieve federated learning state.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFLData();
  }, []);

  useEffect(() => {
    if (refreshTrigger > 0) {
      loadFLData();
    }
  }, [refreshTrigger]);

  useEffect(() => {
    let interval = null;
    if (autoRefresh) {
      interval = setInterval(() => {
        loadFLData();
      }, 10000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const handleStartFL = async (e) => {
    e.preventDefault();
    try {
      setFlRunning(true);
      setFlError(null);
      setFlJob(null);

      const params = {
        rounds: Number(rounds),
        epochs: Number(epochs),
        model,
        strategy,
        attack_type: attackType,
        malicious_fraction: Number(maliciousFraction),
        enable_blockchain: Boolean(enableBlockchain),
        seed: Number(seed),
      };

      const result = await apiService.startFLSimulation(params);
      setFlJob(result);
      
      // Poll or reload data after 3 seconds
      setTimeout(async () => {
        await loadFLData();
        setFlRunning(false);
      }, 3500);
    } catch (err) {
      setFlError(err.message || 'Federated learning execution failed.');
      setFlRunning(false);
    }
  };

  if (loading) return <LoadingSpinner message="Querying federated learning substations & round telemetry..." />;
  if (error) return <ErrorAlert title="Federated Learning Subsystem Error" message={error} onRetry={loadFLData} />;

  const convergenceData = (roundsData?.rounds || []).map((r) => ({
    round: `R${r.round_number}`,
    Loss: Number(r.loss.toFixed(4)),
    F1: Number((r.f1_score * 100).toFixed(1)),
    Accuracy: Number((r.accuracy * 100).toFixed(1)),
    PRAUC: Number((r.pr_auc * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            Federated Learning Clients & Rounds Telemetry
            <Badge variant="primary">{roundsData?.active_strategy || 'trust_aware_fedavg'}</Badge>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Regional substation client partitions, non-IID load profiles, local training rounds, and convergence tracking.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="success">{clientsData?.active_clients ?? 4} Nodes Active</Badge>
          <Badge variant="danger">{clientsData?.quarantined_clients ?? 1} Quarantined</Badge>
        </div>
      </div>

      {/* KPI Cards (Responsive) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Rounds Completed"
          value={roundsData?.total_rounds_completed ?? 2}
          subtitle={`Strategy: ${roundsData?.active_strategy ?? 'Trust-Aware FedAvg'}`}
          icon={Network}
          color="cyan"
        />

        <StatCard
          title="Best F1 Score"
          value={`${((roundsData?.best_f1_score || 1.0) * 100).toFixed(1)}%`}
          subtitle={`Convergence: ${roundsData?.convergence_status ?? 'CONVERGED'}`}
          icon={TrendingUp}
          color="emerald"
        />

        <StatCard
          title="Total Communication"
          value={`${roundsData?.total_communication_mb?.toFixed(2) ?? '3.30'} MB`}
          subtitle="1.65 MB per 5-node round"
          icon={Cpu}
          color="purple"
        />

        <StatCard
          title="Regional Partitions"
          value={`${clientsData?.total_clients ?? 5} Substations`}
          subtitle="Non-IID Industrial to Rural range"
          icon={Radio}
          color="amber"
        />
      </div>

      {/* Epistemic Privacy Notice */}
      <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-300 space-y-1">
        <div className="flex items-center space-x-2 font-semibold text-cyan-400">
          <Lock className="w-4 h-4" />
          <span>Federated Learning Privacy Boundary</span>
        </div>
        <p className="text-slate-400 text-[11px] leading-relaxed">
          {roundsData?.epistemic_disclaimer ||
            'Transmitting model parameter updates preserves raw meter dataset isolation at local substations. However, FL alone does not guarantee differential privacy against reconstruction attacks without formal DP or secure aggregation.'}
        </p>
      </div>

      {/* Substation Partition Profiles Table */}
      <div className="glass-panel rounded-xl p-5 border border-slate-800/80">
        <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          Regional Substation Clients & Non-IID Load Profiles
        </h3>
        <p className="text-xs text-slate-400 mb-4">
          Extreme non-IID load heterogeneity across smart grid substations. The trust mechanism protects honest non-IID
          outliers (client_A, client_E) while isolating malicious poisoning.
        </p>

        <div className="overflow-x-auto -mx-2 sm:mx-0">
          <table className="w-full text-xs text-left min-w-[720px]">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 whitespace-nowrap">
                <th className="py-2.5 px-3">Substation</th>
                <th className="py-2.5 px-3">Feeder Description</th>
                <th className="py-2.5 px-3">Meters</th>
                <th className="py-2.5 px-3">Mean Daily Load</th>
                <th className="py-2.5 px-3">Trust Score</th>
                <th className="py-2.5 px-3">Reputation</th>
                <th className="py-2.5 px-3">Firewall</th>
                <th className="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 whitespace-nowrap">
              {(clientsData?.clients || []).map((client) => (
                <tr key={client.client_id} className="hover:bg-slate-800/40">
                  <td className="py-2.5 px-3 font-mono font-bold text-white flex items-center gap-2">
                    <FileCode className="w-3.5 h-3.5 text-cyan-400" />
                    {client.client_id}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">
                    <div>{client.substation_name}</div>
                    <span className="text-[10px] text-slate-500 font-mono">{client.region_type}</span>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">
                    {client.meters_count} ({client.theft_meters} theft)
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-200 font-semibold">
                    {client.mean_daily_kwh} kWh
                  </td>
                  <td className="py-2.5 px-3 font-mono font-semibold">
                    <span
                      className={
                        client.current_trust_score >= 0.8
                          ? 'text-cyan-400'
                          : client.current_trust_score >= 0.5
                          ? 'text-amber-400'
                          : 'text-rose-400'
                      }
                    >
                      {client.current_trust_score.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-semibold">
                    <span
                      className={
                        client.current_reputation >= 0.8
                          ? 'text-emerald-400'
                          : client.current_reputation >= 0.5
                          ? 'text-amber-400'
                          : 'text-rose-400'
                      }
                    >
                      {client.current_reputation.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
                      {client.privacy_firewall}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    {client.status === 'QUARANTINED' ? (
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

      {/* Convergence Charts */}
      <div className="glass-panel rounded-xl p-5 border border-slate-800/80">
        <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-emerald-400" />
          Global Test Loss & F1-Score Convergence Over Rounds
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={convergenceData} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="round" stroke="#64748B" fontSize={11} />
              <YAxis stroke="#64748B" fontSize={11} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0F172A',
                  borderColor: '#334155',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="F1" stroke="#10B981" strokeWidth={2.5} dot={{ r: 4 }} name="F1 Score (%)" />
              <Line type="monotone" dataKey="Accuracy" stroke="#06B6D4" strokeWidth={2} dot={{ r: 3 }} name="Accuracy (%)" />
              <Line type="monotone" dataKey="PRAUC" stroke="#8B5CF6" strokeWidth={1.5} strokeDasharray="4 4" dot={{ r: 2 }} name="PR-AUC (%)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* FL Experiment Dispatcher Form */}
      <div className="glass-panel rounded-xl p-6 border border-slate-800/80">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Play className="w-4 h-4 text-cyan-400" />
              Launch Safe Federated Learning Experiment
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Execute bounded training rounds with custom attack injections and robust aggregation strategies.
            </p>
          </div>
          <Badge variant="warning">Production Controls Require Auth</Badge>
        </div>

        <form onSubmit={handleStartFL} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Rounds */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                FL Rounds (1–20)
              </label>
              <input
                type="number"
                min="1"
                max="20"
                value={rounds}
                onChange={(e) => setRounds(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                required
              />
            </div>

            {/* Local Epochs */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Local Epochs (1–5)
              </label>
              <input
                type="number"
                min="1"
                max="5"
                value={epochs}
                onChange={(e) => setEpochs(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                required
              />
            </div>

            {/* Model Architecture */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Model Architecture
              </label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value="cnn_lstm">CNN-LSTM (Deep Sequence)</option>
                <option value="lstm">LSTM (Recurrent Baseline)</option>
              </select>
            </div>

            {/* Aggregation Strategy */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Aggregation Defense
              </label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value="trust_aware_fedavg">Trust-Aware FedAvg (Our Defense)</option>
                <option value="coordinate_median">Coordinate-Wise Median (Robust)</option>
                <option value="fedavg">Standard FedAvg (Vulnerable)</option>
              </select>
            </div>

            {/* Attack Modality */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Simulated Byzantine Attack
              </label>
              <select
                value={attackType}
                onChange={(e) => setAttackType(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value="none">None (Clean Benchmark)</option>
                <option value="sign_flipping">Sign-Flipping Gradient Inversion</option>
                <option value="model_poisoning">Extreme Model Poisoning (High Norm)</option>
                <option value="random_noise">Gaussian Random Noise Update</option>
                <option value="label_flipping">Adversarial Label Flipping</option>
              </select>
            </div>

            {/* Malicious Fraction */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Adversary Ratio (0–50%)
              </label>
              <input
                type="number"
                step="0.05"
                min="0.0"
                max="0.5"
                value={maliciousFraction}
                onChange={(e) => setMaliciousFraction(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                required
              />
            </div>

            {/* Blockchain Logging */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                On-Chain Audit Logging
              </label>
              <select
                value={enableBlockchain ? 'true' : 'false'}
                onChange={(e) => setEnableBlockchain(e.target.value === 'true')}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value="true">Enabled (ReputationRegistry)</option>
                <option value="false">Disabled (Off-Chain Only)</option>
              </select>
            </div>

            {/* Seed */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Random Seed
              </label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                required
              />
            </div>
          </div>

          {/* Submit */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2">
            <span className="text-[11px] text-amber-400 font-mono leading-tight">
              ⚠️ Simulated experiment execution. Bypasses live OAuth2 verification in development.
            </span>
            <button
              type="submit"
              disabled={flRunning}
              className={`inline-flex items-center space-x-2 px-5 py-2 rounded-lg text-xs font-semibold text-white transition-all ${
                flRunning
                  ? 'bg-slate-700 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 shadow-md shadow-cyan-500/20'
              }`}
            >
              {flRunning ? (
                <>
                  <RotateCcw className="w-4 h-4 animate-spin" />
                  <span>Coordinating Federated Training...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Start FL Experiment</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Job Output */}
        {flJob && (
          <div className="mt-4 p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/40 text-xs text-emerald-200 space-y-1">
            <div className="flex items-center space-x-2 font-bold text-emerald-300">
              <CheckCircle2 className="w-4 h-4" />
              <span>Job Initialized: {flJob.job_id} ({flJob.status})</span>
            </div>
            <p className="text-[11px] text-emerald-300/80">{flJob.message}</p>
          </div>
        )}

        {flError && (
          <div className="mt-4 p-3 rounded-lg bg-rose-950/40 border border-rose-500/40 text-xs text-rose-300">
            {flError}
          </div>
        )}
      </div>
    </div>
  );
};

export default FLClientsRoundsPage;
