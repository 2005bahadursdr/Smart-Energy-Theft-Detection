import React, { useEffect, useState } from 'react';
import {
  Gauge,
  Sliders,
  AlertTriangle,
  Zap,
  Play,
  RotateCcw,
  ShieldCheck,
  CheckCircle2,
  PieChart as PieIcon,
  BarChart2,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import { apiService } from '../services/api';
import StatCard from '../components/StatCard';
import Badge from '../components/Badge';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorAlert from '../components/ErrorAlert';

const COLORS = ['#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#64748B'];

export const MetersPage = ({ autoRefresh = false, refreshTrigger = 0 }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Simulation Form State
  const [numMeters, setNumMeters] = useState(100);
  const [days, setDays] = useState(30);
  const [samplingFreq, setSamplingFreq] = useState(15);
  const [theftRatio, setTheftRatio] = useState(0.15);
  const [seed, setSeed] = useState(42);

  const [simRunning, setSimRunning] = useState(false);
  const [simResult, setSimResult] = useState(null);
  const [simError, setSimError] = useState(null);

  const loadMetersData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiService.getMetersSummary();
      setData(res);
    } catch (err) {
      setError(err.message || 'Failed to load smart meter telemetry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMetersData();
  }, []);

  useEffect(() => {
    if (refreshTrigger > 0) {
      loadMetersData();
    }
  }, [refreshTrigger]);

  useEffect(() => {
    let interval = null;
    if (autoRefresh) {
      interval = setInterval(() => {
        loadMetersData();
      }, 10000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const handleRunSimulation = async (e) => {
    e.preventDefault();
    try {
      setSimRunning(true);
      setSimError(null);
      setSimResult(null);

      const params = {
        num_meters: Number(numMeters),
        days: Number(days),
        sampling_freq_minutes: Number(samplingFreq),
        theft_ratio: Number(theftRatio),
        seed: Number(seed),
      };

      const result = await apiService.runSimulation(params);
      setSimResult(result);
      await loadMetersData(); // Refresh summary data
    } catch (err) {
      setSimError(err.message || 'Simulation execution failed.');
    } finally {
      setSimRunning(false);
    }
  };

  if (loading) return <LoadingSpinner message="Querying aggregated smart meter telemetry..." />;
  if (error) return <ErrorAlert title="Meter Subsystem Offline" message={error} onRetry={loadMetersData} />;

  const metrics = data?.consumption_metrics || {};
  const statusData = Object.entries(data?.status_breakdown || {}).map(([name, value]) => ({
    name,
    value,
  }));

  const profileData = Object.entries(data?.consumer_profiles_breakdown || {}).map(([name, count]) => ({
    name,
    count,
  }));

  const eventsData = Object.entries(data?.event_breakdown || {})
    .slice(0, 6)
    .map(([name, value]) => ({
      name: name.replace('_', ' '),
      value,
    }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            Smart Meter Simulator & MDMS Telemetry
            <Badge variant="demo">{data?.data_source_label || 'SIMULATED_DEMO_DATA'}</Badge>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Aggregated consumption distribution and theft event statistics across smart grid meters.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="success">Privacy Firewall: ACTIVE</Badge>
          <Badge variant="primary">15-Min Granularity</Badge>
        </div>
      </div>

      {/* Privacy Notice Box */}
      <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-800/40 text-xs text-emerald-200 flex items-start space-x-3">
        <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
        <div>
          <h4 className="font-semibold text-emerald-300">Confidentiality Guarantee</h4>
          <p className="text-emerald-200/80 mt-0.5">
            Individual smart meter consumption time series and consumer PII are strictly redacted. Only privacy-preserving
            statistical moments, substation totals, and anomaly ratios are reported.
          </p>
        </div>
      </div>

      {/* Primary KPI Grid (Responsive & Strictly Dynamic) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Monitored Meters"
          value={data?.total_meters ? data.total_meters.toLocaleString() : '0'}
          subtitle={`Sampled Slices: ${data?.total_readings_sampled ? data.total_readings_sampled.toLocaleString() : '0'}`}
          icon={Gauge}
          color="cyan"
        />

        <StatCard
          title="Theft Ratio"
          value={data?.theft_ratio != null ? `${(data.theft_ratio * 100).toFixed(1)}%` : '0.0%'}
          subtitle={`Hardware Tamper Alerts: ${data?.hardware_tamper_alerts != null ? data.hardware_tamper_alerts.toLocaleString() : '0'}`}
          icon={AlertTriangle}
          color="amber"
        />

        <StatCard
          title="Mean Reported Load"
          value={metrics?.mean_kwh != null ? `${metrics.mean_kwh.toFixed(2)} kWh` : '0.00 kWh'}
          subtitle={`Std Dev: ${metrics?.std_kwh != null ? metrics.std_kwh.toFixed(2) : '0.00'} kWh | Median: ${metrics?.median_kwh != null ? metrics.median_kwh.toFixed(2) : '0.00'}`}
          icon={Zap}
          color="emerald"
        />

        <StatCard
          title="Unreported Stolen Energy"
          value={data?.estimated_unreported_theft_kwh != null ? `${data.estimated_unreported_theft_kwh.toLocaleString(undefined, { maximumFractionDigits: 0 })} kWh` : '0 kWh'}
          subtitle={`Total Reported: ${metrics?.total_consumed_kwh != null ? metrics.total_consumed_kwh.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '0'} kWh`}
          icon={AlertTriangle}
          color="rose"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Meter Status Breakdown */}
        <div className="glass-panel rounded-xl p-5 border border-slate-800/80">
          <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
            <PieIcon className="w-4 h-4 text-cyan-400" />
            Meter Operational Status Distribution
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={85}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend
                  wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                  formatter={(val) => <span className="text-slate-300">{val}</span>}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Consumer Profile Breakdown */}
        <div className="glass-panel rounded-xl p-5 border border-slate-800/80">
          <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-emerald-400" />
            Consumer Load Classification
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={profileData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="name" stroke="#64748B" fontSize={11} />
                <YAxis stroke="#64748B" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="count" fill="#10B981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Grid Events Breakdown Table */}
      <div className="glass-panel rounded-xl p-5 border border-slate-800/80">
        <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          MDMS Ingested Grid Events & Hardware Alerts
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {eventsData.map((ev, i) => (
            <div key={i} className="p-3 rounded-lg bg-slate-900/80 border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider">{ev.name}</span>
              <p className="text-lg font-bold text-white mt-0.5">{ev.value.toLocaleString()}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Simulation Controller Panel */}
      <div className="glass-panel rounded-xl p-6 border border-slate-800/80">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Smart Meter Grid Simulator Controller
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Re-generate synthetic grid telemetry with safe, bounded research parameters.
            </p>
          </div>
          <Badge variant="warning">Auth Bypassed (Dev)</Badge>
        </div>

        {/* Form */}
        <form onSubmit={handleRunSimulation} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* Num Meters */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Total Meters (10–2000)
              </label>
              <input
                type="number"
                min="10"
                max="2000"
                value={numMeters}
                onChange={(e) => setNumMeters(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                required
              />
            </div>

            {/* Duration Days */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Duration Days (1–90)
              </label>
              <input
                type="number"
                min="1"
                max="90"
                value={days}
                onChange={(e) => setDays(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                required
              />
            </div>

            {/* Sampling Frequency */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Sampling Slices
              </label>
              <select
                value={samplingFreq}
                onChange={(e) => setSamplingFreq(Number(e.target.value))}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value={15}>15 Minutes</option>
                <option value={30}>30 Minutes</option>
                <option value={60}>60 Minutes</option>
              </select>
            </div>

            {/* Theft Ratio */}
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Theft Ratio (0–50%)
              </label>
              <input
                type="number"
                step="0.01"
                min="0.0"
                max="0.5"
                value={theftRatio}
                onChange={(e) => setTheftRatio(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
                required
              />
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

          {/* Submit Button */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2">
            <span className="text-[11px] text-amber-400 font-mono leading-tight">
              ⚠️ In production grid deployment, simulation dispatch requires OAuth2/JWT authorization.
            </span>
            <button
              type="submit"
              disabled={simRunning}
              className={`inline-flex items-center space-x-2 px-5 py-2 rounded-lg text-xs font-semibold text-white transition-all ${
                simRunning
                  ? 'bg-slate-700 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 shadow-md shadow-cyan-500/20'
              }`}
            >
              {simRunning ? (
                <>
                  <RotateCcw className="w-4 h-4 animate-spin" />
                  <span>Synthesizing Grid Telemetry...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Execute Simulation Run</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Simulation Result Banner */}
        {simResult && (
          <div className="mt-4 p-4 rounded-xl bg-emerald-950/20 border border-emerald-500/40 text-xs text-emerald-200 space-y-2">
            <div className="flex items-center space-x-2 font-bold text-emerald-300">
              <CheckCircle2 className="w-4 h-4" />
              <span>Simulation Completed: {simResult.simulation_id}</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-[11px]">
              <div>Meters: {simResult.generated_summary?.total_meters}</div>
              <div>Theft Nodes: {simResult.generated_summary?.theft_meters}</div>
              <div>Sampled Readings: {simResult.generated_summary?.total_sampled_readings?.toLocaleString()}</div>
              <div>Tamper Alerts: {simResult.generated_summary?.hardware_tamper_alerts}</div>
            </div>
          </div>
        )}

        {/* Simulation Error Banner */}
        {simError && (
          <div className="mt-4 p-3 rounded-lg bg-rose-950/40 border border-rose-500/40 text-xs text-rose-300">
            {simError}
          </div>
        )}
      </div>
    </div>
  );
};

export default MetersPage;
