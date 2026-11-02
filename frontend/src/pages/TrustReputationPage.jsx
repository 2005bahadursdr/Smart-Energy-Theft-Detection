import React, { useEffect, useState, useMemo } from 'react';
import {
  Award,
  Sliders,
  ShieldCheck,
  Scale,
  Activity,
  Layers,
  HelpCircle,
  TrendingDown,
  Info,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Copy,
  Check,
  Filter,
  Sparkles,
  RefreshCw,
  X,
  Eye,
  Zap,
} from 'lucide-react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from 'recharts';
import { apiService } from '../services/api';
import StatCard from '../components/StatCard';
import Badge from '../components/Badge';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorAlert from '../components/ErrorAlert';

// Substation Feeder Metadata baseline mapping
const BASE_CLIENT_PROFILES = {
  client_A: { name: 'client_A', label: 'Industrial Feeder', load: '38.32 kWh (Continuous high inductive load)', color: '#06B6D4', shortLabel: 'Ind' },
  client_B: { name: 'client_B', label: 'Commercial Feeder', load: '24.15 kWh (Business daytime peak load)', color: '#F59E0B', shortLabel: 'Com' },
  client_C: { name: 'client_C', label: 'Urban Residential 1', load: '18.42 kWh (Evening dual-peak load)', color: '#10B981', shortLabel: 'Urb' },
  client_D: { name: 'client_D', label: 'Urban Residential 2', load: '21.05 kWh (Suburban baseline load)', color: '#EC4899', shortLabel: 'Sub' },
  client_E: { name: 'client_E', label: 'Rural & Agricultural', load: '0.36 kWh (Low load & intermittent solar PV)', color: '#8B5CF6', shortLabel: 'Rur' },
};

export const TrustReputationPage = ({ autoRefresh = false, refreshTrigger = 0 }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  // Filter state for scores table
  const [filterStatus, setFilterStatus] = useState('ALL'); // 'ALL' | 'TRUSTED' | 'QUARANTINED'

  // Modal inspection state
  const [inspectClient, setInspectClient] = useState(null);

  // Radar chart client visibility toggles
  const [radarVisibleClients, setRadarVisibleClients] = useState({
    client_A: true,
    client_B: true,
    client_C: true,
    client_D: true,
    client_E: true,
  });

  // What-If Policy Simulation Sandbox State
  const [simMode, setSimMode] = useState(false);
  const [weightSim, setWeightSim] = useState(0.40);
  const [weightVal, setWeightVal] = useState(0.35);
  const [weightRep, setWeightRep] = useState(0.25);
  const [quarantineThreshold, setQuarantineThreshold] = useState(0.25);
  const [suspiciousThreshold, setSuspiciousThreshold] = useState(0.45);

  const loadTrustData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiService.getTrustScores();
      setData(res);

      // Initialize simulator weights from backend
      if (res?.weights_used) {
        setWeightSim(res.weights_used.weight_similarity ?? 0.40);
        setWeightVal(res.weights_used.weight_validation ?? 0.35);
        setWeightRep(res.weights_used.weight_reputation ?? 0.25);
      }
      if (res?.thresholds) {
        setQuarantineThreshold(res.thresholds.quarantine_threshold ?? 0.25);
        setSuspiciousThreshold(res.thresholds.suspicious_threshold ?? 0.45);
      }
    } catch (err) {
      setError(err.message || 'Failed to load trust & reputation metrics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTrustData();
  }, []);

  useEffect(() => {
    if (refreshTrigger > 0) {
      loadTrustData();
    }
  }, [refreshTrigger]);

  useEffect(() => {
    let interval = null;
    if (autoRefresh) {
      interval = setInterval(() => {
        loadTrustData();
      }, 10000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const handleCopyJson = () => {
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleResetDefaults = () => {
    const defaultWeights = data?.weights_used || {
      weight_similarity: 0.40,
      weight_validation: 0.35,
      weight_reputation: 0.25,
    };
    const defaultThresholds = data?.thresholds || {
      suspicious_threshold: 0.45,
      quarantine_threshold: 0.25,
    };
    setWeightSim(defaultWeights.weight_similarity);
    setWeightVal(defaultWeights.weight_validation);
    setWeightRep(defaultWeights.weight_reputation);
    setQuarantineThreshold(defaultThresholds.quarantine_threshold);
    setSuspiciousThreshold(defaultThresholds.suspicious_threshold);
  };

  // Preset policies
  const applyPreset = (type) => {
    setSimMode(true);
    switch (type) {
      case 'default':
        setWeightSim(0.40);
        setWeightVal(0.35);
        setWeightRep(0.25);
        setQuarantineThreshold(0.25);
        setSuspiciousThreshold(0.45);
        break;
      case 'cosine_heavy':
        setWeightSim(0.65);
        setWeightVal(0.20);
        setWeightRep(0.15);
        setQuarantineThreshold(0.30);
        break;
      case 'reputation_memory':
        setWeightSim(0.20);
        setWeightVal(0.20);
        setWeightRep(0.60);
        setQuarantineThreshold(0.20);
        break;
      case 'validation_strict':
        setWeightSim(0.25);
        setWeightVal(0.55);
        setWeightRep(0.20);
        setQuarantineThreshold(0.35);
        break;
      default:
        break;
    }
  };

  const toggleRadarClient = (cid) => {
    setRadarVisibleClients((prev) => ({
      ...prev,
      [cid]: !prev[cid],
    }));
  };

  // Raw base scores from backend
  const baseScores = useMemo(() => data?.scores || [], [data]);

  // Dynamically evaluated scores (active policy or simulated what-if policy)
  const computedScores = useMemo(() => {
    if (!baseScores.length) return [];
    if (!simMode) return baseScores;

    const totalWeight = weightSim + weightVal + weightRep || 1.0;
    const wS = weightSim / totalWeight;
    const wV = weightVal / totalWeight;
    const wR = weightRep / totalWeight;

    // First pass: compute composite trust & raw quarantine state
    const intermediate = baseScores.map((s) => {
      // Directional component
      const S_k = s.similarity_score;
      const S_comp = Math.max(0, S_k);

      // Validation delta component (lower delta -> higher quality)
      const V_comp = Math.max(0, 1.0 - s.validation_delta * 2.0);

      // Reputation component
      const R_comp = s.reputation_score;

      // Composite trust formula
      let composite = (wS * S_comp) + (wV * V_comp) + (wR * R_comp);

      // Immediate quarantine rule: negative similarity inversion or sub-threshold
      const isQuarantined = S_k < 0 || composite < quarantineThreshold;
      if (isQuarantined) {
        composite = 0.0;
      }

      return {
        ...s,
        simulated_composite: composite,
        simulated_quarantined: isQuarantined,
        simulated_status: isQuarantined ? 'QUARANTINED' : 'TRUSTED',
      };
    });

    // Second pass: compute normalized aggregation weights sum
    const honestSum = intermediate
      .filter((item) => !item.simulated_quarantined)
      .reduce((acc, curr) => acc + curr.simulated_composite, 0);

    return intermediate.map((item) => {
      const normalizedWeight = item.simulated_quarantined || honestSum <= 0
        ? 0.0
        : item.simulated_composite / honestSum;

      return {
        ...item,
        composite_trust_score: item.simulated_composite,
        normalized_weight: normalizedWeight,
        is_quarantined: item.simulated_quarantined,
        status: item.simulated_status,
      };
    });
  }, [baseScores, simMode, weightSim, weightVal, weightRep, quarantineThreshold]);

  // Derived KPI metrics
  const activeScores = computedScores;
  const trustedNodes = activeScores.filter((s) => !s.is_quarantined);
  const quarantinedNodes = activeScores.filter((s) => s.is_quarantined);

  const topClient = useMemo(() => {
    if (!activeScores.length) return { client_id: 'client_C', composite_trust_score: 0.958, normalized_weight: 0.268 };
    return [...activeScores].sort((a, b) => b.composite_trust_score - a.composite_trust_score)[0];
  }, [activeScores]);

  const outlierClient = useMemo(() => {
    if (quarantinedNodes.length > 0) return quarantinedNodes[0];
    return [...activeScores].sort((a, b) => a.composite_trust_score - b.composite_trust_score)[0] || {
      client_id: 'client_D',
      composite_trust_score: 0.0,
      normalized_weight: 0.0,
    };
  }, [activeScores, quarantinedNodes]);

  const quarantinedClientId = useMemo(() => {
    const q = activeScores.find((s) => s.is_quarantined);
    return q ? q.client_id : 'client_D';
  }, [activeScores]);

  const dynamicClientProfiles = useMemo(() => {
    const res = {};
    Object.keys(BASE_CLIENT_PROFILES).forEach((cid) => {
      const base = BASE_CLIENT_PROFILES[cid];
      if (cid === quarantinedClientId) {
        res[cid] = {
          ...base,
          label: 'Compromised Node (Adversary)',
          load: 'Injected Byzantine Attack (Quarantined w_k = 0.00)',
          color: '#EF4444',
          isAdversary: true,
        };
      } else {
        res[cid] = {
          ...base,
          isAdversary: false,
        };
      }
    });
    return res;
  }, [quarantinedClientId]);

  const dynamicReputationHistory = useMemo(() => {
    const qId = quarantinedClientId || 'client_D';
    const rounds = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5'];
    const decayScores = [0.85, 0.60, 0.35, 0.15, 0.05];
    const honestScores = {
      client_A: [0.85, 0.88, 0.91, 0.93, 0.94],
      client_B: [0.85, 0.87, 0.89, 0.91, 0.92],
      client_C: [0.85, 0.90, 0.93, 0.95, 0.96],
      client_D: [0.85, 0.88, 0.91, 0.93, 0.95],
      client_E: [0.80, 0.79, 0.81, 0.83, 0.85],
    };

    return rounds.map((r, idx) => {
      const row = { round: r, quarantineCutoff: 0.25 };
      ['client_A', 'client_B', 'client_C', 'client_D', 'client_E'].forEach((cid) => {
        row[cid] = cid === qId ? decayScores[idx] : honestScores[cid][idx];
      });
      return row;
    });
  }, [quarantinedClientId]);

  const nonIidClient = useMemo(() => {
    const honestA = trustedNodes.find((s) => s.client_id === 'client_A');
    if (honestA) return honestA;
    const honestE = trustedNodes.find((s) => s.client_id === 'client_E');
    if (honestE) return honestE;
    return trustedNodes[0] || {
      client_id: 'client_E',
      composite_trust_score: 0.892,
      distance_metric: 0.28,
    };
  }, [trustedNodes]);

  // Filtered scores for table
  const displayedScores = useMemo(() => {
    if (filterStatus === 'TRUSTED') return activeScores.filter((s) => !s.is_quarantined);
    if (filterStatus === 'QUARANTINED') return activeScores.filter((s) => s.is_quarantined);
    return activeScores;
  }, [activeScores, filterStatus]);

  // Radar chart data mapping (covers all 5 clients A through E)
  const radarData = useMemo(() => {
    const getCl = (cid) => activeScores.find((s) => s.client_id === cid) || {};
    const cA = getCl('client_A');
    const cB = getCl('client_B');
    const cC = getCl('client_C');
    const cD = getCl('client_D');
    const cE = getCl('client_E');

    return [
      {
        subject: 'Direction Similarity',
        A: Math.max(0, Math.round((cA.similarity_score ?? 0.985) * 100)),
        B: Math.max(0, Math.round((cB.similarity_score ?? 0.972) * 100)),
        C: Math.max(0, Math.round((cC.similarity_score ?? 0.991) * 100)),
        D: Math.max(0, Math.round((cD.similarity_score ?? 0) * 100)),
        E: Math.max(0, Math.round((cE.similarity_score ?? 0.935) * 100)),
      },
      {
        subject: 'Distance Closeness',
        A: Math.round(Math.exp(-(cA.distance_metric ?? 0.12)) * 100),
        B: Math.round(Math.exp(-(cB.distance_metric ?? 0.15)) * 100),
        C: Math.round(Math.exp(-(cC.distance_metric ?? 0.09)) * 100),
        D: Math.round(Math.exp(-(cD.distance_metric ?? 4.25)) * 100),
        E: Math.round(Math.exp(-(cE.distance_metric ?? 0.28)) * 100),
      },
      {
        subject: 'Validation Quality',
        A: Math.round(Math.max(0, 1 - (cA.validation_delta ?? 0.002)) * 100),
        B: Math.round(Math.max(0, 1 - (cB.validation_delta ?? 0.003)) * 100),
        C: Math.round(Math.max(0, 1 - (cC.validation_delta ?? 0.001)) * 100),
        D: Math.round(Math.max(0, 1 - (cD.validation_delta ?? 0.45)) * 100),
        E: Math.round(Math.max(0, 1 - (cE.validation_delta ?? 0.006)) * 100),
      },
      {
        subject: 'Historical Reputation',
        A: Math.round((cA.reputation_score ?? 0.85) * 100),
        B: Math.round((cB.reputation_score ?? 0.85) * 100),
        C: Math.round((cC.reputation_score ?? 0.85) * 100),
        D: Math.round((cD.reputation_score ?? 0.35) * 100),
        E: Math.round((cE.reputation_score ?? 0.80) * 100),
      },
      {
        subject: 'Composite Trust',
        A: Math.round((cA.composite_trust_score ?? 0.942) * 100),
        B: Math.round((cB.composite_trust_score ?? 0.915) * 100),
        C: Math.round((cC.composite_trust_score ?? 0.958) * 100),
        D: Math.round((cD.composite_trust_score ?? 0.0) * 100),
        E: Math.round((cE.composite_trust_score ?? 0.892) * 100),
      },
    ];
  }, [activeScores]);

  if (loading && !data) return <LoadingSpinner message="Evaluating multi-criteria trust scores and reputation ledger..." />;
  if (error && !data) return <ErrorAlert title="Trust Engine Error" message={error} onRetry={loadTrustData} />;

  const activeWeights = simMode
    ? {
        weight_similarity: Number(weightSim.toFixed(2)),
        weight_validation: Number(weightVal.toFixed(2)),
        weight_reputation: Number(weightRep.toFixed(2)),
      }
    : data?.weights_used || {
        weight_similarity: 0.40,
        weight_validation: 0.35,
        weight_reputation: 0.25,
      };

  const activeThresholds = simMode
    ? {
        suspicious_threshold: Number(suspiciousThreshold.toFixed(2)),
        quarantine_threshold: Number(quarantineThreshold.toFixed(2)),
      }
    : data?.thresholds || {
        suspicious_threshold: 0.45,
        quarantine_threshold: 0.25,
      };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center flex-wrap gap-2">
            Multi-Criteria Trust & Reputation Engine
            <Badge variant="blockchain">Decaying EMA</Badge>
            {simMode && <Badge variant="warning">What-If Sandbox Active</Badge>}
            {autoRefresh && <Badge variant="warning">Live Stream Active</Badge>}
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Dynamic aggregation weight calculation combining cosine direction similarity, server validation delta, and historical reputation.
          </p>
        </div>
        <div className="flex items-center flex-wrap gap-2">
          <Badge variant="tamperOk">Honest Non-IID Protected</Badge>
          <Badge variant="danger">Adversary Quarantined</Badge>
          <button
            onClick={() => setSimMode(!simMode)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
              simMode
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-sm shadow-amber-500/20'
                : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700'
            }`}
            title="Toggle Policy Sandbox Simulator"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>{simMode ? 'Exit Sandbox' : 'What-If Simulator'}</span>
          </button>
          <button
            onClick={handleCopyJson}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Copy Trust Metrics JSON"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
          </button>
          <button
            onClick={loadTrustData}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            title="Refresh Live Trust Data"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Trust Formula Architecture Box */}
      <div className="glass-panel rounded-xl p-5 border border-cyan-500/30 bg-cyan-950/20">
        <div className="flex items-start space-x-3">
          <div className="p-2.5 bg-cyan-500/20 rounded-lg text-cyan-400 shrink-0">
            <Scale className="w-5 h-5" />
          </div>
          <div className="flex-1 text-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h4 className="text-sm font-bold text-cyan-300">
                Composite Trust Scoring Formula: T_k = α · S_k + β · V_k + γ · R_k
              </h4>
              <Badge variant={simMode ? 'warning' : 'primary'}>
                {simMode ? 'SIMULATED WHAT-IF POLICY' : 'SERVER BASELINE POLICY'}
              </Badge>
            </div>
            <p className="text-slate-300 mt-1 leading-relaxed">
              Active Parameters:{' '}
              <span className="font-mono text-cyan-400">α = {activeWeights.weight_similarity} (Cosine Agreement)</span>,{' '}
              <span className="font-mono text-emerald-400">β = {activeWeights.weight_validation} (Validation Loss Impact)</span>,{' '}
              <span className="font-mono text-purple-400">γ = {activeWeights.weight_reputation} (Historical Reputation EMA)</span>.
              Normalized aggregation weight: <span className="font-mono text-cyan-300">w_k = T_k / Σ T_j</span> (0.0 if quarantined).
            </p>
            <div className="flex flex-wrap items-center gap-2 mt-2.5 text-[11px] font-mono text-slate-300">
              <span className="bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800">
                Suspicious: <strong className="text-white">{activeThresholds.suspicious_threshold}</strong>
              </span>
              <span className="bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800">
                Quarantine Cutoff: <strong className="text-rose-400">{activeThresholds.quarantine_threshold}</strong>
              </span>
              <span className="bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800">
                Forgiveness α: <strong className="text-white">0.15</strong>
              </span>
              <span className="bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800">
                EMA Decay λ: <strong className="text-white">0.85</strong>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive What-If Sensitivity Simulator Panel */}
      {simMode && (
        <div className="glass-panel rounded-xl p-5 border border-amber-500/30 bg-amber-950/15 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-amber-500/20 gap-2">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-amber-400" />
              <h3 className="text-sm font-bold text-amber-300">
                Interactive Policy Sensitivity Simulator
              </h3>
            </div>
            <div className="flex items-center flex-wrap gap-2 text-xs">
              <span className="text-slate-400 text-[11px]">Quick Presets:</span>
              <button
                onClick={() => applyPreset('default')}
                className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono"
              >
                Default (40/35/25)
              </button>
              <button
                onClick={() => applyPreset('cosine_heavy')}
                className="px-2 py-0.5 rounded bg-cyan-950/60 hover:bg-cyan-900/60 text-cyan-300 border border-cyan-800 text-[11px] font-mono"
              >
                Cosine Heavy (65/20/15)
              </button>
              <button
                onClick={() => applyPreset('reputation_memory')}
                className="px-2 py-0.5 rounded bg-purple-950/60 hover:bg-purple-900/60 text-purple-300 border border-purple-800 text-[11px] font-mono"
              >
                Reputation Memory (20/20/60)
              </button>
              <button
                onClick={() => applyPreset('validation_strict')}
                className="px-2 py-0.5 rounded bg-emerald-950/60 hover:bg-emerald-900/60 text-emerald-300 border border-emerald-800 text-[11px] font-mono"
              >
                Validation Strict (25/55/20)
              </button>
              <button
                onClick={handleResetDefaults}
                className="px-2 py-0.5 rounded bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 border border-rose-800 text-[11px] flex items-center gap-1 font-mono"
              >
                <RotateCcw className="w-3 h-3" />
                Reset
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 text-xs">
            {/* Slider Alpha */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span className="text-cyan-400 font-mono">α (Similarity)</span>
                <span className="font-mono text-white font-bold">{weightSim.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={weightSim}
                onChange={(e) => setWeightSim(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-slate-500">Weight for cosine gradient alignment</p>
            </div>

            {/* Slider Beta */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span className="text-emerald-400 font-mono">β (Validation)</span>
                <span className="font-mono text-white font-bold">{weightVal.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={weightVal}
                onChange={(e) => setWeightVal(parseFloat(e.target.value))}
                className="w-full accent-emerald-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-slate-500">Weight for validation loss delta</p>
            </div>

            {/* Slider Gamma */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span className="text-purple-400 font-mono">γ (Reputation)</span>
                <span className="font-mono text-white font-bold">{weightRep.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={weightRep}
                onChange={(e) => setWeightRep(parseFloat(e.target.value))}
                className="w-full accent-purple-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-slate-500">Weight for historical decaying EMA</p>
            </div>

            {/* Quarantine Threshold */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span className="text-rose-400 font-mono">τ_q (Quarantine)</span>
                <span className="font-mono text-white font-bold">{quarantineThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.60"
                step="0.05"
                value={quarantineThreshold}
                onChange={(e) => setQuarantineThreshold(parseFloat(e.target.value))}
                className="w-full accent-rose-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-slate-500">Cutoff below which nodes receive w_k = 0</p>
            </div>

            {/* Suspicious Threshold */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-900/70 border border-slate-800">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span className="text-amber-400 font-mono">τ_s (Suspicious)</span>
                <span className="font-mono text-white font-bold">{suspiciousThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.80"
                step="0.05"
                value={suspiciousThreshold}
                onChange={(e) => setSuspiciousThreshold(parseFloat(e.target.value))}
                className="w-full accent-amber-400 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <p className="text-[10px] text-slate-500">Warning threshold for audit logging</p>
            </div>
          </div>
        </div>
      )}

      {/* Dynamic KPI Cards (Dynamically computed from active/simulated scores) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Top Trusted Substation"
          value={`${topClient.client_id} (${(topClient.composite_trust_score * 100).toFixed(1)}%)`}
          subtitle={`Weight: ${(topClient.normalized_weight * 100).toFixed(1)}% | Full Trust Score`}
          icon={Award}
          color="emerald"
        />

        <StatCard
          title="Adversarial Outlier"
          value={`${outlierClient.client_id} (${(outlierClient.composite_trust_score * 100).toFixed(1)}%)`}
          subtitle={`Status: ${outlierClient.status} | Weight: ${(outlierClient.normalized_weight * 100).toFixed(1)}%`}
          icon={TrendingDown}
          color="rose"
        />

        <StatCard
          title="Non-IID High-Load Node"
          value={`${nonIidClient.client_id} (${(nonIidClient.composite_trust_score * 100).toFixed(1)}%)`}
          subtitle={`Distance: ${nonIidClient.distance_metric.toFixed(2)} | Divergence Respected`}
          icon={Activity}
          color="cyan"
        />

        <StatCard
          title="Active Defense Quarantines"
          value={`${quarantinedNodes.length} / ${activeScores.length} Quarantined`}
          subtitle={`${trustedNodes.length} Trusted Non-IID Nodes Protected`}
          icon={ShieldCheck}
          color="purple"
        />
      </div>

      {/* Detailed Component Breakdown Table */}
      <div className="glass-panel rounded-xl p-5 border border-slate-800/80">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 mb-4 border-b border-slate-800 gap-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Component Trust Scores, Aggregation Weights & Reason Codes
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Granular inspection of component metrics per substation node. Note how {quarantinedClientId} is zero-weighted due to Byzantine attack quarantine.
            </p>
          </div>
          <div className="flex items-center space-x-2 text-xs">
            <span className="text-slate-400 text-[11px] flex items-center gap-1">
              <Filter className="w-3.5 h-3.5" /> Filter:
            </span>
            <button
              onClick={() => setFilterStatus('ALL')}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
                filterStatus === 'ALL'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              All ({activeScores.length})
            </button>
            <button
              onClick={() => setFilterStatus('TRUSTED')}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
                filterStatus === 'TRUSTED'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              Trusted ({trustedNodes.length})
            </button>
            <button
              onClick={() => setFilterStatus('QUARANTINED')}
              className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors ${
                filterStatus === 'QUARANTINED'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              Quarantined ({quarantinedNodes.length})
            </button>
          </div>
        </div>

        <div className="overflow-x-auto -mx-2 sm:mx-0">
          <table className="w-full text-xs text-left min-w-[950px]">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono whitespace-nowrap">
                <th className="py-2.5 px-3">Substation</th>
                <th className="py-2.5 px-3">Feeder Profile</th>
                <th className="py-2.5 px-3">Cosine Agreement (S_k)</th>
                <th className="py-2.5 px-3">Update Distance</th>
                <th className="py-2.5 px-3">Validation Delta (V_k)</th>
                <th className="py-2.5 px-3">Reputation (R_k)</th>
                <th className="py-2.5 px-3">Composite Trust</th>
                <th className="py-2.5 px-3">Aggregation Weight (w_k)</th>
                <th className="py-2.5 px-3">Reason Code</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3 text-center">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono whitespace-nowrap">
              {displayedScores.map((s) => {
                const profile = dynamicClientProfiles[s.client_id] || { label: 'Feeder', color: '#64748B' };
                return (
                  <tr key={s.client_id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2.5 px-3 font-bold text-white flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: profile.color }} />
                      {s.client_id}
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-300 text-[11px]">
                      {profile.label}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={s.similarity_score < 0 ? 'text-rose-400 font-bold' : 'text-emerald-400'}>
                        {s.similarity_score.toFixed(3)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">{s.distance_metric.toFixed(2)}</td>
                    <td className="py-2.5 px-3 text-slate-300">{s.validation_delta.toFixed(3)}</td>
                    <td className="py-2.5 px-3 text-slate-200 font-semibold">{s.reputation_score.toFixed(2)}</td>
                    <td className="py-2.5 px-3 text-cyan-400 font-bold">
                      {(s.composite_trust_score * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                          s.normalized_weight === 0
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        }`}
                      >
                        {(s.normalized_weight * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-300 text-[11px]">
                      {s.is_quarantined ? (
                        <span className="text-rose-400 font-semibold">NEGATIVE_SIMILARITY_INVERSION</span>
                      ) : s.similarity_score > 0.95 ? (
                        <span className="text-emerald-400">HIGH_DIRECTIONAL_AGREEMENT</span>
                      ) : (
                        <span className="text-slate-400">NORMAL_NON_IID_DIVERGENCE</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3">
                      {s.status === 'QUARANTINED' ? (
                        <Badge variant="danger">QUARANTINED</Badge>
                      ) : (
                        <Badge variant="success">TRUSTED</Badge>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <button
                        onClick={() => setInspectClient(s)}
                        className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-cyan-400 transition-colors"
                        title={`Inspect ${s.client_id}`}
                      >
                        <Eye className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Analytics Grid: Radar Chart + Reputation Trajectory */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 1. Multi-Dimensional Trust Signature Radar (All 5 Clients with Toggles) */}
        <div className="glass-panel rounded-xl p-5 border border-slate-800/80">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-2 gap-2">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Scale className="w-4 h-4 text-purple-400" />
              Multi-Dimensional Trust Signature Radar
            </h3>
            <span className="text-[11px] text-slate-400">Toggle Substation Visibility:</span>
          </div>
          <p className="text-xs text-slate-400 mb-3">
            Comparing honest substations against the compromised adversary ({quarantinedClientId}). Click pills to isolate feeds.
          </p>

          {/* Interactive Radar Client Toggle Pills */}
          <div className="flex flex-wrap items-center gap-1.5 mb-3">
            {Object.entries(dynamicClientProfiles).map(([cid, prof]) => {
              const active = radarVisibleClients[cid];
              return (
                <button
                  key={cid}
                  onClick={() => toggleRadarClient(cid)}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono font-semibold transition-all border ${
                    active
                      ? 'bg-slate-800 text-white shadow-sm'
                      : 'bg-slate-900/60 text-slate-500 border-slate-800 opacity-60'
                  }`}
                  style={{
                    borderColor: active ? prof.color : undefined,
                  }}
                >
                  <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: prof.color }} />
                  {cid} {prof.isAdversary ? '(Adversary)' : ''}
                </button>
              );
            })}
          </div>

          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                <PolarGrid stroke="#1E293B" />
                <PolarAngleAxis dataKey="subject" stroke="#64748B" fontSize={10} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#334155" fontSize={9} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                {radarVisibleClients.client_A && (
                  <Radar
                    name={`client_A (${dynamicClientProfiles.client_A?.isAdversary ? 'Adversary' : 'Industrial'})`}
                    dataKey="A"
                    stroke={dynamicClientProfiles.client_A?.color || '#06B6D4'}
                    fill={dynamicClientProfiles.client_A?.color || '#06B6D4'}
                    fillOpacity={dynamicClientProfiles.client_A?.isAdversary ? 0.4 : 0.2}
                  />
                )}
                {radarVisibleClients.client_B && (
                  <Radar
                    name={`client_B (${dynamicClientProfiles.client_B?.isAdversary ? 'Adversary' : 'Commercial'})`}
                    dataKey="B"
                    stroke={dynamicClientProfiles.client_B?.color || '#F59E0B'}
                    fill={dynamicClientProfiles.client_B?.color || '#F59E0B'}
                    fillOpacity={dynamicClientProfiles.client_B?.isAdversary ? 0.4 : 0.15}
                  />
                )}
                {radarVisibleClients.client_C && (
                  <Radar
                    name={`client_C (${dynamicClientProfiles.client_C?.isAdversary ? 'Adversary' : 'Urban'})`}
                    dataKey="C"
                    stroke={dynamicClientProfiles.client_C?.color || '#10B981'}
                    fill={dynamicClientProfiles.client_C?.color || '#10B981'}
                    fillOpacity={dynamicClientProfiles.client_C?.isAdversary ? 0.4 : 0.2}
                  />
                )}
                {radarVisibleClients.client_D && (
                  <Radar
                    name={`client_D (${dynamicClientProfiles.client_D?.isAdversary ? 'Adversary' : 'Suburban'})`}
                    dataKey="D"
                    stroke={dynamicClientProfiles.client_D?.color || '#EF4444'}
                    fill={dynamicClientProfiles.client_D?.color || '#EF4444'}
                    fillOpacity={dynamicClientProfiles.client_D?.isAdversary ? 0.4 : 0.2}
                  />
                )}
                {radarVisibleClients.client_E && (
                  <Radar
                    name={`client_E (${dynamicClientProfiles.client_E?.isAdversary ? 'Adversary' : 'Rural'})`}
                    dataKey="E"
                    stroke={dynamicClientProfiles.client_E?.color || '#8B5CF6'}
                    fill={dynamicClientProfiles.client_E?.color || '#8B5CF6'}
                    fillOpacity={dynamicClientProfiles.client_E?.isAdversary ? 0.4 : 0.2}
                  />
                )}
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 2. Historical Reputation Decay & Recovery Trajectory (Recharts Line Chart) */}
        <div className="glass-panel rounded-xl p-5 border border-slate-800/80">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              Historical Reputation EMA Trajectory Across Rounds
            </h3>
            <Badge variant="blockchain">λ = 0.85</Badge>
          </div>
          <p className="text-xs text-slate-400 mb-4">
            Exponential decay: <span className="font-mono text-cyan-300">R_k^(t) = λ·R_k^(t-1) + (1-λ)·T_k^(t)</span>. Adversary {quarantinedClientId} collapses below quarantine cutoff.
          </p>

          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dynamicReputationHistory} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="round" stroke="#64748B" fontSize={11} />
                <YAxis stroke="#64748B" fontSize={11} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                  formatter={(val) => [`${(val * 100).toFixed(1)}%`, '']}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <ReferenceLine
                  y={0.25}
                  stroke="#EF4444"
                  strokeDasharray="4 4"
                  label={{ value: 'Quarantine Cutoff (0.25)', fill: '#EF4444', fontSize: 10, position: 'top' }}
                />
                <Line type="monotone" dataKey="client_A" stroke={dynamicClientProfiles.client_A?.color || '#06B6D4'} strokeWidth={dynamicClientProfiles.client_A?.isAdversary ? 2.5 : 2} dot={{ r: dynamicClientProfiles.client_A?.isAdversary ? 4 : 3 }} name={`client_A (${dynamicClientProfiles.client_A?.isAdversary ? 'Adv' : 'Ind'})`} />
                <Line type="monotone" dataKey="client_B" stroke={dynamicClientProfiles.client_B?.color || '#F59E0B'} strokeWidth={dynamicClientProfiles.client_B?.isAdversary ? 2.5 : 2} dot={{ r: dynamicClientProfiles.client_B?.isAdversary ? 4 : 3 }} name={`client_B (${dynamicClientProfiles.client_B?.isAdversary ? 'Adv' : 'Com'})`} />
                <Line type="monotone" dataKey="client_C" stroke={dynamicClientProfiles.client_C?.color || '#10B981'} strokeWidth={dynamicClientProfiles.client_C?.isAdversary ? 2.5 : 2} dot={{ r: dynamicClientProfiles.client_C?.isAdversary ? 4 : 3 }} name={`client_C (${dynamicClientProfiles.client_C?.isAdversary ? 'Adv' : 'Urb'})`} />
                <Line type="monotone" dataKey="client_D" stroke={dynamicClientProfiles.client_D?.color || '#EF4444'} strokeWidth={dynamicClientProfiles.client_D?.isAdversary ? 2.5 : 2} dot={{ r: dynamicClientProfiles.client_D?.isAdversary ? 4 : 3 }} name={`client_D (${dynamicClientProfiles.client_D?.isAdversary ? 'Adv' : 'Sub'})`} />
                <Line type="monotone" dataKey="client_E" stroke={dynamicClientProfiles.client_E?.color || '#8B5CF6'} strokeWidth={dynamicClientProfiles.client_E?.isAdversary ? 2.5 : 2} dot={{ r: dynamicClientProfiles.client_E?.isAdversary ? 4 : 3 }} name={`client_E (${dynamicClientProfiles.client_E?.isAdversary ? 'Adv' : 'Rur'})`} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Substation Detail Deep-Dive Modal */}
      {inspectClient && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel rounded-2xl max-w-xl w-full p-6 border border-slate-700 bg-[#0F172A] shadow-2xl relative space-y-4">
            <div className="flex items-start justify-between border-b border-slate-800 pb-3">
              <div>
                <div className="flex items-center space-x-2">
                  <h3 className="text-lg font-bold text-white font-mono">
                    Substation Audit: {inspectClient.client_id}
                  </h3>
                  {inspectClient.is_quarantined ? (
                    <Badge variant="danger">QUARANTINED</Badge>
                  ) : (
                    <Badge variant="success">TRUSTED</Badge>
                  )}
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  {dynamicClientProfiles[inspectClient.client_id]?.label} • {dynamicClientProfiles[inspectClient.client_id]?.load}
                </p>
              </div>
              <button
                onClick={() => setInspectClient(null)}
                className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Cosine Similarity (S_k)</span>
                <span className={`text-base font-bold font-mono ${
                  inspectClient.similarity_score < 0 ? 'text-rose-400' : 'text-emerald-400'
                }`}>
                  {inspectClient.similarity_score.toFixed(4)}
                </span>
                <p className="text-[10px] text-slate-500 mt-1">
                  {inspectClient.similarity_score < 0 ? 'Negative cosine (180° inversion)' : 'High directional agreement'}
                </p>
              </div>

              <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Gradient L2 Distance</span>
                <span className="text-base font-bold font-mono text-cyan-400">
                  {inspectClient.distance_metric.toFixed(4)}
                </span>
                <p className="text-[10px] text-slate-500 mt-1">
                  Euclidean distance from median coordinate
                </p>
              </div>

              <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Validation Delta (V_k)</span>
                <span className="text-base font-bold font-mono text-purple-400">
                  {inspectClient.validation_delta.toFixed(4)}
                </span>
                <p className="text-[10px] text-slate-500 mt-1">
                  Server loss impact benchmark
                </p>
              </div>

              <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800">
                <span className="text-slate-400 text-[11px] block">Historical Reputation (R_k)</span>
                <span className="text-base font-bold font-mono text-amber-400">
                  {inspectClient.reputation_score.toFixed(4)}
                </span>
                <p className="text-[10px] text-slate-500 mt-1">
                  Exponential moving average score
                </p>
              </div>
            </div>

            <div className="p-3.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-semibold">Composite Trust Score (T_k):</span>
                <span className="font-mono text-cyan-300 font-bold text-sm">
                  {(inspectClient.composite_trust_score * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-semibold">Normalized Aggregation Weight (w_k):</span>
                <span className="font-mono text-emerald-400 font-bold text-sm">
                  {(inspectClient.normalized_weight * 100).toFixed(2)}%
                </span>
              </div>
              <div className="pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 leading-relaxed">
                <strong className="text-slate-300">Defense Logic:</strong>{' '}
                {inspectClient.is_quarantined
                  ? 'This substation gradient was completely quarantined (w_k = 0.0) because its directional similarity with the honest centroid was negative, indicating an active sign-flipping or gradient poisoning attack. The on-chain ledger records this quarantine event.'
                  : 'This substation was successfully admitted into global aggregation. Even with non-IID load variations, its cosine agreement with the server loss benchmark was positive and its historical reputation remained healthy.'}
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setInspectClient(null)}
                className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-colors"
              >
                Close Audit View
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrustReputationPage;
