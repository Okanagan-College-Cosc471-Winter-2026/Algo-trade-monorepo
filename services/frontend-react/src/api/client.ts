import axios from 'axios';
import { 
  Stock, OHLC, Prediction, SnapshotsResponse, 
  BuildSnapshotPayload, BuildSnapshotResponse,
  SimSession, SimStepPrediction, OpsStatus, AirflowStatus, PipelineLog
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const api = {
  healthCheck: () => apiClient.get<boolean>('/utils/health-check/').then(res => res.data),
  
  listStocks: () => apiClient.get<Stock[]>('/market/stocks').then(res => res.data),
  
  getStock: (symbol: string) => apiClient.get<Stock>(`/market/stocks/${symbol.toUpperCase()}`).then(res => res.data),
  
  getOHLC: (symbol: string, days: number = 365) => 
    apiClient.get<OHLC[]>(`/market/stocks/${symbol.toUpperCase()}/ohlc`, { params: { days } }).then(res => res.data),
    
  predict: (symbol: string) => 
    apiClient.get<Prediction>(`/inference/predict/${symbol.toUpperCase()}`).then(res => res.data),
    
  predictBase: (symbol: string) => 
    apiClient.get<Prediction>(`/inference/predict-base/${symbol.toUpperCase()}`).then(res => res.data),
    
  buildSnapshot: (payload: BuildSnapshotPayload) => 
    apiClient.post<BuildSnapshotResponse>('/data/build-snapshot', payload).then(res => res.data),
    
  listSnapshots: () => apiClient.get<SnapshotsResponse>('/data/snapshots').then(res => res.data),
  
  downloadSnapshot: (filename: string) => 
    apiClient.get(`/data/snapshots/download/${filename}`, { responseType: 'blob' }).then(res => res.data),

  // Simulation
  simSymbols: () => apiClient.get<string[]>('/simulation/symbols').then(res => res.data),
  
  simSession: () => apiClient.get<SimSession>('/simulation/session').then(res => res.data),
  
  simBase: (symbol: string) => 
    apiClient.get<SimStepPrediction>(`/simulation/base/${symbol.toUpperCase()}`).then(res => res.data),
    
  simStep: (symbol: string, step: number) => 
    apiClient.get<SimStepPrediction>(`/simulation/step/${symbol.toUpperCase()}/${step}`).then(res => res.data),
    
  simHistory: (symbol: string) => 
    apiClient.get<OHLC[]>(`/simulation/history/${symbol.toUpperCase()}`).then(res => res.data),
    
  simOHLC: (symbol: string) => 
    apiClient.get<OHLC[]>(`/simulation/ohlc/${symbol.toUpperCase()}`).then(res => res.data),

  // Ops
  opsStatus: () => apiClient.get<OpsStatus>('/ops/status').then(res => res.data),
  
  opsNibiSsh: () => apiClient.get<{ alive: boolean }>('/ops/nibi/ssh').then(res => res.data),
  
  opsNibiExec: (command: string) => 
    apiClient.post<{ rc: number; stdout: string; stderr: string }>('/ops/nibi/exec', { command }).then(res => res.data),
    
  opsPipelineLogs: (limit: number = 50) => 
    apiClient.get<PipelineLog[]>('/ops/pipeline/logs', { params: { limit } }).then(res => res.data),
    
  opsDataFreshness: () => apiClient.get<any>('/ops/data/freshness').then(res => res.data),
  
  opsAirflow: () => apiClient.get<AirflowStatus>('/ops/airflow').then(res => res.data),
  
  opsLogTail: (logName: string, lines: number = 80) => 
    apiClient.get<{ exists: boolean; path: string; lines: string[] }>(`/ops/logs/${logName}`, { params: { lines } }).then(res => res.data),
    
  opsNibiRelogin: () => 
    apiClient.post<{ already_alive: boolean; mode: string }>('/ops/nibi/relogin').then(res => res.data),
};
