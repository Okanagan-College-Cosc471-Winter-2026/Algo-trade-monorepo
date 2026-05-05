import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import { useApi, useInterval } from '../hooks';
import { 
  StatusBadge, 
  ProgressBar, 
  LogViewer, 
  DataTable, 
  MetricCard,
  Column
} from '../components';
import { PipelineLog, AirflowStatus } from '../types';
import styles from './Ops.module.css';

export const Ops: React.FC = () => {
  const { data: status, loading, refresh: refreshStatus } = useApi(() => api.opsStatus());
  const [logs, setLogs] = useState<string[]>([]);
  const [logType, setLogType] = useState<'pipeline_15m' | 'nibi_usage'>('pipeline_15m');
  const [nibiCmd, setNibiCmd] = useState('');
  const [nibiResult, setNibiResult] = useState<{ rc: number; stdout: string; stderr: string } | null>(null);

  // Poll for status every 30 seconds
  useInterval(refreshStatus, 30000);

  // Fetch logs separately
  const fetchLogs = useCallback(async () => {
    try {
      const resp = await api.opsLogTail(logType, 60);
      if (resp.lines) setLogs(resp.lines);
    } catch (err) {
      console.error('Failed to fetch logs', err);
    }
  }, [logType]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useInterval(fetchLogs, 10000);

  if (loading && !status) return <div>Loading Ops Status...</div>;
  if (!status) return <div>Error loading status.</div>;

  const handleRunNibiCmd = async () => {
    if (!nibiCmd.trim()) return;
    try {
      const res = await api.opsNibiExec(nibiCmd);
      setNibiResult(res);
    } catch (err) {
      console.error('Failed to run NIBI command', err);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>System Operations</h1>
        <span className={styles.timestamp}>
          Snapshot at {status.generated_at.replace('T', ' ').slice(0, 19)} UTC
        </span>
      </header>

      {!status.ssh_socket.alive && (
        <div className={styles.alert}>
          <strong>NIBI SSH socket is DOWN</strong> — model training is unavailable.
          Run <code>morning_login.sh</code> on the host machine.
        </div>
      )}

      <section className={styles.section}>
        <h2>Service Health</h2>
        <div className={styles.grid}>
          <MetricCard 
            label="Backend API" 
            value={<StatusBadge status="online" />} 
          />
          <MetricCard 
            label="NIBI SSH Socket" 
            value={<StatusBadge status={status.ssh_socket.alive ? 'online' : 'offline'} label={status.ssh_socket.alive ? 'Socket Alive' : 'Socket Dead'} />} 
          />
          <MetricCard 
            label="Collector Pipeline" 
            value={<StatusBadge status={status.collector.collector_state} />} 
            help={status.collector.error}
          />
          <MetricCard 
            label="Market Data" 
            value={<StatusBadge status={status.data.freshness_state} />} 
            help={status.data.freshness_reason}
          />
        </div>
      </section>

      <section className={styles.section}>
        <h2>NIBI Training Job</h2>
        <div className={styles.jobInfo}>
          <div className={styles.grid}>
            <MetricCard label="Job ID" value={status.live_job_primary?.job_id || status.nibi_job.job_id || '—'} />
            <MetricCard label="Job Name" value={status.live_job_primary?.name || '—'} />
            <MetricCard label="Status" value={<StatusBadge status={status.live_job_primary?.state || status.nibi_job.live_state || 'unknown'} />} />
            <MetricCard label="Elapsed/Limit" value={status.live_job_primary ? `${status.live_job_primary.elapsed} / ${status.live_job_primary.time_lim}` : '—'} />
          </div>
          
          <div className={styles.progressSection}>
            <ProgressBar 
              value={status.model.windows_ok / (status.model.windows_total || 1)} 
              label={`Warm-refresh windows: ${status.model.windows_ok} / ${status.model.windows_total} completed`}
              subLabel={status.model.windows_error ? `${status.model.windows_error} errors` : ''}
              variant="primary"
            />
          </div>
        </div>
      </section>

      <section className={styles.section}>
        <h2>Live Logs</h2>
        <div className={styles.logControls}>
          <button 
            className={logType === 'pipeline_15m' ? styles.activeTab : ''} 
            onClick={() => setLogType('pipeline_15m')}
          >
            Data Pipeline
          </button>
          <button 
            className={logType === 'nibi_usage' ? styles.activeTab : ''} 
            onClick={() => setLogType('nibi_usage')}
          >
            NIBI Usage
          </button>
        </div>
        <LogViewer lines={logs} />
      </section>

      <section className={styles.section}>
        <h2>NIBI Remote Commands</h2>
        <div className={styles.nibiConsole}>
          <input 
            type="text" 
            value={nibiCmd} 
            onChange={(e) => setNibiCmd(e.target.value)}
            placeholder="squeue -u harshsaw"
            className={styles.cmdInput}
          />
          <button 
            onClick={handleRunNibiCmd}
            disabled={!status.ssh_socket.alive}
            className={styles.runBtn}
          >
            Run on NIBI
          </button>
        </div>
        {nibiResult && (
          <div className={styles.consoleOutput}>
            <div className={styles.rc}>Exit code: {nibiResult.rc}</div>
            <pre>{nibiResult.stdout}</pre>
            {nibiResult.stderr && <pre className={styles.stderr}>{nibiResult.stderr}</pre>}
          </div>
        )}
      </section>
    </div>
  );
};
