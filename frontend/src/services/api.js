/**
 * Backend API Client & Service Layer.
 * Connects to FastAPI backend on http://127.0.0.1:8000 (via Vite proxy or direct URL).
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';
const ROOT_URL = '';

class ApiService {
  async _request(endpoint, options = {}) {
    const url = endpoint.startsWith('http') ? endpoint : `${ROOT_URL}${endpoint}`;
    const defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };

    const config = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      if (!response.ok) {
        let errMessage = `HTTP Error ${response.status}: ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData.message) errMessage = errData.message;
          else if (errData.detail) {
            errMessage = typeof errData.detail === 'string' 
              ? errData.detail 
              : JSON.stringify(errData.detail);
          }
        } catch {
          // Response body was not JSON
        }
        throw new Error(errMessage);
      }
      return await response.json();
    } catch (error) {
      console.warn(`[API Error on ${url}]:`, error.message);
      throw error;
    }
  }

  // 1. Health Probe
  async getHealth() {
    return this._request('/health');
  }

  // 2. Meters Summary
  async getMetersSummary() {
    return this._request('/api/meters/summary');
  }

  // 3. Substation Clients
  async getClients() {
    return this._request('/api/clients');
  }

  // 4. FL Rounds Telemetry
  async getFLRounds() {
    return this._request('/api/fl/rounds');
  }

  // 5. Trigger FL Simulation
  async startFLSimulation(params) {
    return this._request('/api/fl/start', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // 6. Theft Risk Predictions
  async getPredictions() {
    return this._request('/api/predictions');
  }

  // 6b. Live Smart Meter Risk Inference
  async evaluateMeterRisk(params) {
    return this._request('/api/predictions/evaluate', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // 6c. Real Dataset Sample Smart Meters
  async getPredictionSamples() {
    return this._request('/api/predictions/samples');
  }

  // 7. Byzantine Attack Events
  async getByzantineEvents() {
    return this._request('/api/byzantine/events');
  }

  // 7b. Inject Byzantine Attack System-Wide
  async injectByzantineAttack(params) {
    return this._request('/api/byzantine/inject', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // 7c. Reset Byzantine Attack State
  async resetByzantineAttack() {
    return this._request('/api/byzantine/reset', {
      method: 'POST',
    });
  }

  // 8. Trust & Reputation Scores
  async getTrustScores() {
    return this._request('/api/trust/scores');
  }

  // 9. Blockchain Audit Ledger
  async getBlockchainAudit() {
    return this._request('/api/blockchain/audit');
  }

  // 10. Run Meter Simulation
  async runSimulation(params) {
    return this._request('/api/simulation/run', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // 11. Experiments Summary
  async getExperimentsSummary() {
    return this._request('/api/experiments/summary');
  }

  // 12. Benchmarks Comparison
  async getBenchmarksComparison() {
    return this._request('/api/benchmarks/comparison');
  }
}

export const apiService = new ApiService();
export default apiService;
