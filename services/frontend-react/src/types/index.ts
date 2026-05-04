export interface Stock {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  exchange: string;
}

export interface OHLC {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionPath {
  bar_time: string;
  pred_close: number;
}

export interface Prediction {
  symbol: string;
  current_price: number;
  prediction_date: string;
  model_version: string;
  predicted_direction: 'up' | 'down';
  predicted_full_day_return: number;
  path: PredictionPath[];
  refreshed_at?: string;
  warm_refreshed_at?: string;
  as_of?: string;
}

export interface Snapshot {
  filename: string;
  size_mb: number;
}

export interface SnapshotsResponse {
  directory: string;
  snapshots: Snapshot[];
}

export interface BuildSnapshotPayload {
  ticker: string;
  start_date?: string;
  end_date?: string;
  format: 'parquet' | 'csv' | 'both';
}

export interface BuildSnapshotResponse {
  tickers_processed: number;
  total_rows_extracted: number;
  files_created: string[];
}

export interface SimSession {
  replay_date: string;
  effective_as_of_date: string;
  steps_completed: number;
  step_labels: string[];
  base_trees: number;
  warm_trees_per_step: number;
}

export interface SimStepPrediction {
  predicted_full_day_return: number;
  predicted_direction: string;
  bars: Array<{
    pred_log_return: number;
  }>;
}

export interface OpsStatus {
  generated_at: string;
  ssh_socket: {
    alive: boolean;
  };
  collector: {
    collector_state: string;
    age_min?: number;
    last_stage?: string;
    last_status?: string;
    error?: string;
  };
  data: {
    staleness_min?: number;
    freshness_state: string;
    total_rows: number;
    freshness_reason?: string;
  };
  nibi_job: {
    job_id?: string;
    live_state?: string;
    status?: string;
    submitted_at?: string;
    sim_date?: string;
  };
  live_job_primary?: {
    job_id: string;
    name: string;
    state: string;
    elapsed: string;
    time_lim: string;
  };
  model: {
    name: string;
    train_end_date: string;
    n_estimators: number;
    promoted_at: string;
    active: boolean;
    windows_ok: number;
    windows_total: number;
    windows_error: number;
    windows_steps: Array<{
      step: number;
      et_label: string;
      status: string;
      train_sec: number;
      total_sec: number;
    }>;
  };
  machine: {
    hostname: string;
    uptime_hrs: number;
    load_avg: {
      '1m': number;
      '5m': number;
      '15m': number;
    };
    process_count: number;
    os: string;
    cpu_pct: number;
    cpu_model: string;
    cpu_cores: number;
    cpu_per_core: number[];
    ram_pct: number;
    ram_used_gb: number;
    ram_total_gb: number;
    swap_pct: number;
    swap_used_gb: number;
    swap_total_gb: number;
    disk: Record<string, { pct: number; used_gb: number; total_gb: number }>;
    net_sent_gb: number;
    net_recv_gb: number;
    net_pkts_sent: number;
    net_pkts_recv: number;
  };
  nibi_gpu?: {
    name: string;
    mem_used: number;
    mem_total: number;
    util_pct: number;
    temp_c: number;
  };
  nibi_jobs: {
    available: boolean;
    user?: string;
    queued: Array<{
      job_id: string;
      name: string;
      state: string;
      elapsed: string;
      time_lim: string;
      start: string;
    }>;
    history: Array<{
      job_id: string;
      name: string;
      state: string;
      exit: string;
      elapsed: string;
      start: string;
    }>;
    quota_raw?: string;
  };
  training_flow: {
    cutoff_date?: string;
    sim_date?: string;
    current_stage?: string;
    message?: string;
    stages: Array<{
      label: string;
      status: string;
      detail: string;
    }>;
    snapshot?: {
      cutoff_symbols: number;
      open_bar_symbols: number;
      close_bar_symbols: number;
      validation_ok: boolean;
    };
  };
}

export interface AirflowStatus {
  dags: Array<{
    dag_id: string;
    is_paused: boolean;
    last_state: string;
    last_start: string;
    last_end: string;
    schedule: string;
    next_run: string;
  }>;
  recent_runs: Array<{
    dag_id: string;
    state: string;
    run_type: string;
    started: string;
    ended: string;
    duration_s: number;
    run_id: string;
  }>;
  error?: string;
}

export interface PipelineLog {
  ts: string;
  stage: string;
  status: string;
  message: string;
}
